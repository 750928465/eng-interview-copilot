"""
向量存储模块
使用 ChromaDB + sentence-transformers 实现本地向量检索
"""
import os
from typing import List, Optional
import chromadb
from chromadb.api import EmbeddingFunction
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class SentenceTransformerEmbedding(EmbeddingFunction):
    """自定义 Embedding 函数"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
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
        file_path = file_path or config.knowledge_path

        if not os.path.exists(file_path):
            print(f"知识文件不存在: {file_path}")
            return 0

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否已有数据
        existing = self.collection.get()
        if existing["ids"]:
            print("知识库已存在，跳过加载")
            return len(existing["ids"])

        # 分块并存储
        chunks = self.chunk_text(content)
        if not chunks:
            return 0

        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        print(f"已加载 {len(chunks)} 个知识块到向量数据库")
        return len(chunks)

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

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = results.get("documents", [[]])
        return documents[0] if documents else []

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