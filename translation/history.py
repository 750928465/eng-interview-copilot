"""
Translation history persistence.

Only ASR translation pairs are stored here. RAG/LLM knowledge-base Q&A stays out
of this file by design.
"""
import json
import os
from datetime import datetime
from typing import Dict, List

from config import config


def append_translation_history(
    source_text: str,
    translated_text: str,
    asr_ms: float = 0.0,
    translate_ms: float = 0.0,
    request_id: str = "",
    request_started_at: str = "",
    round_num: int = 0,
) -> bool:
    source_text = (source_text or "").strip()
    translated_text = (translated_text or "").strip()
    if not source_text or not translated_text:
        return False

    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_text": source_text,
        "translated_text": translated_text,
        "asr_ms": round(float(asr_ms or 0.0), 2),
        "translate_ms": round(float(translate_ms or 0.0), 2),
        "request_id": request_id,
        "request_started_at": request_started_at,
        "round_num": int(round_num or 0),
    }

    os.makedirs(os.path.dirname(config.translation_history_path), exist_ok=True)
    with open(config.translation_history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return True


def load_translation_history(limit: int = 200) -> List[Dict[str, object]]:
    if not os.path.exists(config.translation_history_path):
        return []

    entries: List[Dict[str, object]] = []
    with open(config.translation_history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if limit > 0:
        entries = entries[-limit:]
    return list(reversed(entries))


def load_translation_requests(limit: int = 100) -> List[Dict[str, object]]:
    entries = load_translation_history(limit=0)
    requests: Dict[str, Dict[str, object]] = {}

    for index, entry in enumerate(reversed(entries)):
        request_id = str(entry.get("request_id") or "")
        if not request_id:
            request_id = f"legacy-{entry.get('timestamp', index)}"

        timestamp = str(entry.get("request_started_at") or entry.get("timestamp") or "")
        if request_id not in requests:
            requests[request_id] = {
                "request_id": request_id,
                "started_at": timestamp,
                "rounds": [],
            }

        requests[request_id]["rounds"].append(entry)
        if timestamp and (
            not requests[request_id]["started_at"]
            or timestamp < requests[request_id]["started_at"]
        ):
            requests[request_id]["started_at"] = timestamp

    grouped = list(requests.values())
    grouped.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    if limit > 0:
        grouped = grouped[:limit]
    return grouped


def clear_translation_history() -> None:
    if os.path.exists(config.translation_history_path):
        os.remove(config.translation_history_path)
