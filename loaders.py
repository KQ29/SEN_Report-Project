# loaders.py
import json
import re
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd


# ---------- Utilities ----------
def parse_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, (datetime, pd.Timestamp)):
        try:
            return pd.to_datetime(ts).to_pydatetime()
        except Exception:
            return None
    s = str(ts).strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None


def extract_user_id_and_audience(query: str) -> Tuple[Optional[int], str]:
    m = re.search(r'user[_\s]?id\s*[:=]?\s*(\d+)', query, flags=re.IGNORECASE)
    user_id = int(m.group(1)) if m else None
    audience = "parent"
    if re.search(r'\bteacher\b', query, flags=re.IGNORECASE):
        audience = "teacher"
    return user_id, audience


# ---------- JSON loading ----------
def load_json_raw(path_or_bytes) -> Any:
    if hasattr(path_or_bytes, "read"):
        return json.load(path_or_bytes)
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return json.loads(path_or_bytes.decode("utf-8"))
    if isinstance(path_or_bytes, str):
        try:
            with open(path_or_bytes, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return json.loads(path_or_bytes)
    return json.loads(path_or_bytes)


def _ensure_lists(d: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "user", "users", "enrollment", "daily_activity_log", "topic_session",
        "chapter_session", "activity_performance", "lesson_session",
        "topics", "Topics", "messages", "ai_literacy_assessment",
        "emotional_regulation_log", "support_usage", "communication_events",
        "avatar_change", "background_change", "section_contents",
        "lesson_sections", "mcqs", "mcq_options", "open_questions",
    ]
    out = dict(d)
    for k in keys:
        if k in out and not isinstance(out[k], list):
            out[k] = [out[k]]
    return out


def _looks_like_new_schema(raw: Any) -> bool:
    if not isinstance(raw, list):
        return False
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "lesson_sections (chapter)" in item:
            return True
        users = item.get("user") or item.get("users")
        if isinstance(users, list):
            for u in users:
                if isinstance(u, dict) and ("userid" in u or "DOB" in u):
                    return True
    return False


def _convert_new_schema(raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(raw, list):
        return raw

    global_once = {"org", "org_user", "role", "item_master"}
    seen_globals = set()

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        user_records = [u for u in entry.get("user", []) if isinstance(u, dict)]
        if not user_records:
            continue
        user = dict(user_records[0])
        user_id = user.get("user_id") or user.get("userid") or user.get("id")
        if user_id is None:
            continue
        user["user_id"] = user_id
        if "dob" not in user and "DOB" in user:
            user["dob"] = user["DOB"]
        if "device_type" not in user and "deviceType" in user:
            user["device_type"] = user["deviceType"]
        if "class_level" not in user and user.get("grade_level"):
            user["class_level"] = user.get("grade_level")
        merged.setdefault("user", []).append(user)

        def _copy_records(key: str, target_key: Optional[str] = None, *, inject_user: bool = False, mutator=None):
            records = entry.get(key)
            if records is None:
                return []
            if not isinstance(records, list):
                records = [records]
            copied: List[Dict[str, Any]] = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                new_rec = dict(rec)
                if inject_user and "user_id" not in new_rec:
                    new_rec["user_id"] = user_id
                if mutator:
                    mutator(new_rec)
                copied.append(new_rec)
            if not copied:
                return []
            merged.setdefault(target_key or key, []).extend(copied)
            return copied

        _copy_records("topics", inject_user=True)

        lesson_sections = _copy_records("lesson_sections (chapter)", target_key="lesson_sections", inject_user=True)
        lesson_topic = {}
        if lesson_sections:
            for rec in lesson_sections:
                lesson_id = rec.get("lesson_sections_id") or rec.get("lesson_section_id")
                topic_id = rec.get("topic_id") or rec.get("topicId")
                if lesson_id is not None and topic_id:
                    lesson_topic[lesson_id] = topic_id

        section_topic: Dict[Any, Any] = {}

        def _section_mutator(rec: Dict[str, Any]):
            lesson_id = rec.get("lesson_sections_id") or rec.get("lesson_section_id")
            topic_id = None
            if lesson_id in lesson_topic:
                topic_id = lesson_topic.get(lesson_id)
            if topic_id:
                rec.setdefault("topic_id", topic_id)
            sec_id = rec.get("section_content_id")
            if sec_id and topic_id:
                section_topic[sec_id] = topic_id

        _copy_records("section_contents", inject_user=True, mutator=_section_mutator)

        topic_session_topics: Dict[Any, Any] = {}

        def _chapter_mutator(rec: Dict[str, Any]):
            sec_id = rec.get("section_content_id")
            topic_id = None
            if sec_id and sec_id in section_topic:
                topic_id = section_topic.get(sec_id)
            if topic_id:
                rec.setdefault("topic_id", topic_id)
            ts_id = rec.get("topic_session_id")
            if ts_id and topic_id and ts_id not in topic_session_topics:
                topic_session_topics[ts_id] = topic_id

        _copy_records("chapter_session", inject_user=True, mutator=_chapter_mutator)

        def _topic_session_mutator(rec: Dict[str, Any]):
            ts_id = rec.get("topic_session_id")
            topic_id = topic_session_topics.get(ts_id)
            if topic_id:
                rec.setdefault("topic_id", topic_id)

        _copy_records("topic_session", inject_user=True, mutator=_topic_session_mutator)

        _copy_records("activity_performance", inject_user=True)
        _copy_records("ai_literacy_assessment", inject_user=True)

        def _daily_mutator(rec: Dict[str, Any]):
            if "login_timestamp" not in rec and rec.get("login"):
                rec["login_timestamp"] = rec["login"]
            if "time_spent" not in rec and rec.get("total_time_spent") is not None:
                rec["time_spent"] = rec.get("total_time_spent")

        _copy_records("daily_session_log", target_key="daily_activity_log", inject_user=True, mutator=_daily_mutator)

        for gkey in global_once:
            if gkey in entry and gkey not in seen_globals:
                seen_globals.add(gkey)
                _copy_records(gkey)

    return merged


def normalize_union(raw: Any) -> Dict[str, Any]:
    canon_keys = [
        "user", "enrollment", "daily_activity_log", "topic_session",
        "chapter_session", "activity_performance", "lesson_session",
        "topics", "messages", "ai_literacy_assessment",
        "emotional_regulation_log", "support_usage", "communication_events",
        "avatar_change", "background_change", "section_contents",
        "lesson_sections", "mcqs", "mcq_options", "open_questions",
    ]

    buckets: Dict[str, List[Any]] = {k: [] for k in canon_keys}

    def _ingest_dict(d: Dict[str, Any]):
        d = _ensure_lists(d)
        if "users" in d:
            buckets["user"].extend(d.get("users", []))
        if "user" in d:
            buckets["user"].extend(d.get("user", []))
        if "Topics" in d:
            buckets["topics"].extend(d.get("Topics", []))
        if "topics" in d:
            buckets["topics"].extend(d.get("topics", []))
        for k in [
            "enrollment", "daily_activity_log", "topic_session",
            "chapter_session", "activity_performance", "lesson_session",
            "messages", "ai_literacy_assessment", "emotional_regulation_log",
            "support_usage", "communication_events", "avatar_change",
            "background_change", "section_contents", "lesson_sections",
            "mcqs", "mcq_options", "open_questions",
        ]:
            if k in d:
                buckets[k].extend(d.get(k, []))
        for v in d.values():
            if isinstance(v, dict):
                _ingest_dict(v)

    if isinstance(raw, dict):
        if "students" in raw and isinstance(raw["students"], list):
            for s in raw["students"]:
                if isinstance(s, dict):
                    _ingest_dict(s)
        else:
            _ingest_dict(raw)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                _ingest_dict(item)

    for k in canon_keys:
        buckets.setdefault(k, [])
    return buckets


def load_json(path_or_bytes) -> Dict[str, Any]:
    raw = load_json_raw(path_or_bytes)
    if _looks_like_new_schema(raw):
        raw = _convert_new_schema(raw)
    return normalize_union(raw)


# ---------- NEW: merge multiple normalized datasets ----------
def merge_data(*datasets: Dict[str, Any]) -> Dict[str, Any]:
    if not datasets:
        return {}
    merged: Dict[str, List[Any]] = {}
    keys = set().union(*[set(ds.keys()) for ds in datasets])
    for k in keys:
        merged[k] = []
        for ds in datasets:
            merged[k].extend(ds.get(k, []))

    # ✅ Deduplicate ai_literacy_assessment
    if "ai_literacy_assessment" in merged:
        seen = set()
        uniq = []
        for r in merged["ai_literacy_assessment"]:
            if not isinstance(r, dict):
                continue
            uid = r.get("user_id")
            t_at = str(r.get("taken_at", ""))
            typ = str(r.get("type", "")).lower()
            key = (uid, typ, t_at)
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        merged["ai_literacy_assessment"] = uniq

    return merged
