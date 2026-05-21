"""
向量存储模块
使用 ChromaDB + sentence-transformers 实现本地向量检索
"""
import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.api import EmbeddingFunction
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class SentenceTransformerEmbedding(EmbeddingFunction):
    """自定义 Embedding 函数"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            print(f"本地未找到 Embedding 模型，尝试下载: {model_name}")
            self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class VectorStore:
    """向量存储管理器"""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name or config.collection_name
        self.persist_directory = persist_directory or config.chroma_persist_dir
        self.embedding_function = SentenceTransformerEmbedding(embedding_model)

        # 初始化 ChromaDB (持久化模式)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def _file_hash(self, file_path: str) -> str:
        """计算知识库文件内容哈希，用于判断是否需要重建索引。"""
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _knowledge_sources(self, file_path: str = None) -> List[str]:
        if file_path:
            return [file_path]
        return [config.knowledge_path, config.qa_path]

    def _load_sources(self, file_path: str = None) -> Tuple[List[dict], str]:
        documents = []
        manifest_parts = []

        for source_path in self._knowledge_sources(file_path):
            if not os.path.exists(source_path):
                print(f"知识文件不存在，跳过: {source_path}")
                continue

            source_hash = self._file_hash(source_path)
            manifest_parts.append(f"{source_path}:{source_hash}")

            with open(source_path, "r", encoding="utf-8") as f:
                content = f.read()

            if content.strip():
                documents.append({
                    "path": source_path,
                    "hash": source_hash,
                    "content": content,
                })

        manifest_hash = hashlib.sha256("\n".join(manifest_parts).encode("utf-8")).hexdigest()
        return documents, manifest_hash

    def _rewrite_query(self, query: str) -> List[str]:
        normalized = query.lower()
        expansions = []

        rules = [
            (
                ("motivation", "motivated", "why do", "why did", "why this work", "why this problem"),
                "research motivation problem importance why this work matters research gap background",
            ),
            (
                ("novelty", "novel", "innovation", "innovative", "different", "existing work", "prior work"),
                "novelty innovation contribution compared with existing work prior work research gap limitation",
            ),
            (
                ("contribution", "contributions", "main contribution"),
                "main contributions research contribution novelty method evaluation impact",
            ),
            (
                ("proposal", "phd", "research"),
                "phd proposal research question motivation novelty existing work method contribution",
            ),
            (
                ("challenge", "difficult", "hard", "problem you solved"),
                "technical challenge difficulty solution tradeoff impact project experience",
            ),
            (
                ("impact", "result", "outcome", "achievement"),
                "impact result outcome metric achievement evaluation improvement",
            ),
            (
                ("limitation", "weakness", "risk", "future work"),
                "limitation risk weakness future work threat validity next step",
            ),
        ]

        for triggers, expansion in rules:
            if any(trigger in normalized for trigger in triggers):
                expansions.append(expansion)

        queries = [query.strip()]
        queries.extend(expansions)
        seen = set()
        unique_queries = []
        for item in queries:
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                unique_queries.append(item)
        return unique_queries[:5]

    def _source_weight(self, source: str) -> float:
        name = os.path.basename(source).lower()
        if "qa" in name:
            return 1.5
        if "research" in name or "proposal" in name:
            return 1.3
        if "resume" in name or "cv" in name:
            return 1.0
        if "transcript" in name:
            return 0.85
        return 1.0

    def _keyword_score(self, query: str, document: str) -> float:
        query_terms = {
            term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query.lower())
            if term not in {"the", "and", "for", "with", "what", "why", "how", "your", "you", "are"}
        }
        if not query_terms:
            return 0.0

        doc_lower = document.lower()
        hits = sum(1 for term in query_terms if term in doc_lower)
        return hits / max(len(query_terms), 1)

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        将文本分割成块

        Args:
            text: 原始文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数

        Returns:
            文本块列表
        """
        chunk_size = chunk_size or config.chunk_size
        overlap = overlap or config.chunk_overlap

        # 按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落不超过限制，则合并
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)

                # 处理超长段落
                if len(para) > chunk_size:
                    # 按句子或固定长度分割
                    for i in range(0, len(para), chunk_size - overlap):
                        chunk = para[i:i + chunk_size]
                        if chunk.strip():
                            chunks.append(chunk)
                    current_chunk = ""
                else:
                    current_chunk = para

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def load_knowledge(self, file_path: str = None) -> int:
        """
        从文件加载知识库

        Args:
            file_path: 知识文件路径

        Returns:
            加载的文档数量
        """
        documents, manifest_hash = self._load_sources(file_path)
        if not documents:
            print("没有可加载的知识库内容")
            if self.collection.count() > 0:
                self.clear()
            return 0

        # 知识库文件未变化时复用现有索引；变化时重建，避免检索到旧简历内容。
        existing = self.collection.get(limit=1, include=["metadatas"])
        if existing["ids"]:
            metadata = existing.get("metadatas", [{}])[0] or {}
            if metadata.get("manifest_hash") == manifest_hash:
                print("知识库未变化，复用现有索引")
                return self.collection.count()

            print("知识库文件已变化，重建向量索引")
            self.clear()

        all_chunks = []
        ids = []
        metadatas = []

        for source_index, document in enumerate(documents):
            chunks = self.chunk_text(document["content"])
            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                ids.append(f"doc_{source_index}_{chunk_index}")
                metadatas.append({
                    "source": document["path"],
                    "source_hash": document["hash"],
                    "manifest_hash": manifest_hash,
                    "chunk_index": chunk_index,
                })

        if not all_chunks:
            return 0

        self.collection.add(
            documents=all_chunks,
            ids=ids,
            metadatas=metadatas
        )

        print(f"已加载 {len(all_chunks)} 个知识块到向量数据库")
        return len(all_chunks)

    def search(self, query: str, top_k: int = None) -> List[str]:
        """
        相似度检索

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            相似文档列表
        """
        top_k = top_k or config.top_k
        count = self.collection.count()
        if count == 0:
            return []

        query_texts = self._rewrite_query(query)
        candidate_top_k = min(getattr(config, "candidate_top_k", 12), count)
        results = self.collection.query(
            query_texts=query_texts,
            n_results=candidate_top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked: Dict[str, dict] = {}
        result_documents = results.get("documents", [])
        result_metadatas = results.get("metadatas", [])
        result_distances = results.get("distances", [])

        for query_index, documents in enumerate(result_documents):
            metadatas = result_metadatas[query_index] if query_index < len(result_metadatas) else []
            distances = result_distances[query_index] if query_index < len(result_distances) else []
            for doc_index, document in enumerate(documents):
                metadata = metadatas[doc_index] if doc_index < len(metadatas) else {}
                distance = distances[doc_index] if doc_index < len(distances) else 1.0
                source = metadata.get("source", "")
                base_score = 1.0 / (1.0 + max(distance, 0.0))
                score = base_score * self._source_weight(source)
                score += 0.15 * self._keyword_score(query, document)
                score += 0.04 * max(len(query_texts) - query_index, 0)

                previous = ranked.get(document)
                if previous is None or score > previous["score"]:
                    ranked[document] = {"score": score, "document": document}

        return [
            item["document"]
            for item in sorted(ranked.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        ]

    def clear(self) -> None:
        """清空向量数据库"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def get_document_count(self) -> int:
        """获取文档数量"""
        return self.collection.count()


# 全局实例
vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取全局向量存储实例"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store
