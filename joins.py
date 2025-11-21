# joins.py
from typing import Any, Dict, List, Optional
from utils_time import get_time_spent


def build_indexes(data: Dict[str, Any]):
    enrollment_by_id = {e["enrollment_id"]: e for e in data.get("enrollment", []) if isinstance(e, dict) and "enrollment_id" in e}
    topics_by_id = {t["topic_id"]: t for t in data.get("topics", []) if isinstance(t, dict) and "topic_id" in t}
    topic_session_by_id = {t["topic_session_id"]: t for t in data.get("topic_session", []) if isinstance(t, dict) and "topic_session_id" in t}
    chapter_sessions = [c for c in data.get("chapter_session", []) if isinstance(c, dict)]
    chapter_session_by_id = {c["chapter_session_id"]: c for c in chapter_sessions if "chapter_session_id" in c}
    lesson_sessions = [l for l in data.get("lesson_session", []) if isinstance(l, dict)]
    activity_perf = [a for a in data.get("activity_performance", []) if isinstance(a, dict)]
    daily_logs = [d for d in data.get("daily_activity_log", []) if isinstance(d, dict)]
    users = {u["user_id"]: u for u in data.get("user", []) if isinstance(u, dict) and "user_id" in u}

    lesson_sections = [ls for ls in data.get("lesson_sections", []) if isinstance(ls, dict)]
    lesson_sections_by_id = {}
    for ls in lesson_sections:
        key = ls.get("lesson_sections_id") or ls.get("lesson_section_id")
        if key is not None:
            lesson_sections_by_id[key] = ls

    section_contents = [sc for sc in data.get("section_contents", []) if isinstance(sc, dict)]
    section_contents_by_id = {}
    for sc in section_contents:
        key = sc.get("section_content_id")
        if key is not None:
            section_contents_by_id[key] = sc

    return {
        "enrollment_by_id": enrollment_by_id,
        "topics_by_id": topics_by_id,
        "topic_session_by_id": topic_session_by_id,
        "chapter_sessions": chapter_sessions,
        "chapter_session_by_id": chapter_session_by_id,
        "lesson_sessions": lesson_sessions,
        "activity_perf": activity_perf,
        "daily_logs": daily_logs,
        "users": users,
        "lesson_sections_by_id": lesson_sections_by_id,
        "section_contents_by_id": section_contents_by_id,
    }

def chapter_session_user_id(cs: Dict[str, Any], idx: Dict[str, Any]) -> Optional[int]:
    if cs.get("user_id") is not None:
        return cs.get("user_id")
    ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
    if not ts:
        return None
    return topic_session_user_id(ts, idx)

def topic_session_user_id(ts: Dict[str, Any], idx: Dict[str, Any]) -> Optional[int]:
    if ts.get("user_id") is not None:
        return ts.get("user_id")
    en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
    return en.get("user_id") if en else None

def perf_user_id(ap: Dict[str, Any], idx: Dict[str, Any]) -> Optional[int]:
    if ap.get("user_id") is not None:
        return ap.get("user_id")
    cs_id = ap.get("chapter_session_id")
    cs = idx.get("chapter_session_by_id", {}).get(cs_id)
    if not cs:
        return None
    return chapter_session_user_id(cs, idx)

def perf_subject(ap: Dict[str, Any], idx: Dict[str, Any]) -> Optional[str]:
    cs_id = ap.get("chapter_session_id")
    cs = idx.get("chapter_session_by_id", {}).get(cs_id)
    if not cs:
        return None
    topic_id = cs.get("topic_id")
    if topic_id is None:
        sec_id = cs.get("section_content_id")
        if sec_id is not None:
            section = idx.get("section_contents_by_id", {}).get(sec_id)
            if section:
                lesson_id = section.get("lesson_sections_id") or section.get("lesson_section_id")
                lesson = idx.get("lesson_sections_by_id", {}).get(lesson_id)
                if lesson:
                    topic_id = lesson.get("topic_id") or lesson.get("topicId")
    if topic_id is None:
        ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
        if ts:
            topic_id = ts.get("topic_id")
            if topic_id is None:
                en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
                if en:
                    topic_id = en.get("topic_id")
    if topic_id is None:
        return None
    topic = idx["topics_by_id"].get(topic_id)
    return topic.get("subject") if topic else None
