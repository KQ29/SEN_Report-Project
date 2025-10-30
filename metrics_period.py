"""
Period metrics and derived KPIs for the Student Report app.

Exports (backward compatible):
- available_date_range_for_user(data, user_id, idx)
- period_stats(data, user_id, start_dt, end_dt)
- compute_trend(curr, prev)
- compute_focus_score(completion_pct, avg_session_mins)

Advanced Learning KPIs:
- accuracy_and_mastery(data, user_id, start_dt, end_dt)
- response_time_stats(data, user_id, start_dt, end_dt)
- engagement_consistency(data, user_id, start_dt, end_dt)
- independence_support(data, user_id, start_dt, end_dt)
- communication_social(data, user_id, start_dt, end_dt)
- emotional_regulation_summary(data, user_id, start_dt, end_dt)
- activity_attempt_profile(data, user_id, start_dt, end_dt)

NEW:
- ai_literacy_stats(data, user_id, start_dt, end_dt)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from statistics import mean
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from joins import (
    build_indexes,
    topic_session_user_id,
    chapter_session_user_id,
    perf_user_id,
    perf_subject,
)
from utils_time import parse_ts


# -----------------------------
# Helpers
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _pick_first_ts(record: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
    for k in keys:
        if k in record and record[k] not in (None, "", "Unknown"):
            dt = parse_ts(record[k])
            if dt:
                return dt
    return None


def _get_time_spent(r: Dict[str, Any]) -> float:
    """Use either total_time_spent or time_spent if present (minutes or seconds, depending on source)."""
    v = r.get("total_time_spent", None)
    if v is None:
        v = r.get("time_spent", 0)
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


# -----------------------------
# Public: available date range
# -----------------------------
def available_date_range_for_user(
    data: Dict[str, Any], user_id: int, idx: Dict[str, Any]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    dates: List[datetime] = []

    # daily_activity_log
    for r in data.get("daily_activity_log", []):
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("login_timestamp"))
            if dt:
                dates.append(dt)

    # lesson_session
    for r in data.get("lesson_session", []):
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("created_at"))
            if dt:
                dates.append(dt)

    # topic_session
    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid == user_id:
            for k in ("started_at", "completed_at"):
                dt = parse_ts(ts.get(k))
                if dt:
                    dates.append(dt)

    # chapter_session
    for cs in data.get("chapter_session", []):
        uid = chapter_session_user_id(cs, idx)
        if uid == user_id:
            for k in ("started_at", "completed_at"):
                dt = parse_ts(cs.get(k))
                if dt:
                    dates.append(dt)

    # activity_performance
    for ap in data.get("activity_performance", []):
        uid = perf_user_id(ap, idx)
        if uid == user_id:
            dt = parse_ts(ap.get("submitted_at"))
            if dt:
                dates.append(dt)

    if not dates:
        return None, None
    return min(dates), max(dates)


# -----------------------------
# Public: period stats
# -----------------------------
def _filter_records_by_period(
    records: List[Dict[str, Any]], start_dt: datetime, end_dt: datetime, ts_keys: List[str]
) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        ts = _pick_first_ts(r, ts_keys)
        if ts is None:
            continue
        if start_dt <= ts <= end_dt:
            out.append(r)
    return out


def period_stats(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    ts_keys_sessions = ["started_at", "completed_at", "created_at", "timestamp", "date"]
    ts_keys_perf = ["submitted_at"]
    ts_keys_dailies = ["login_timestamp", "created_at", "timestamp", "date"]

    idx = build_indexes(data)

    topic_sessions_all = [t for t in data.get("topic_session", []) if topic_session_user_id(t, idx) == user_id]
    chapter_sessions_all = [c for c in data.get("chapter_session", []) if chapter_session_user_id(c, idx) == user_id]
    lesson_sessions_all = [l for l in data.get("lesson_session", []) if l.get("user_id") == user_id]
    perf_all = [ap for ap in data.get("activity_performance", []) if perf_user_id(ap, idx) == user_id]
    daily_logs_all = [d for d in data.get("daily_activity_log", []) if d.get("user_id") == user_id]

    topic_sessions = _filter_records_by_period(topic_sessions_all, start_dt, end_dt, ts_keys_sessions)
    chapter_sessions = _filter_records_by_period(chapter_sessions_all, start_dt, end_dt, ts_keys_sessions)
    lesson_sessions = _filter_records_by_period(lesson_sessions_all, start_dt, end_dt, ts_keys_sessions)
    perf_rows = _filter_records_by_period(perf_all, start_dt, end_dt, ts_keys_perf)
    daily_logs = _filter_records_by_period(daily_logs_all, start_dt, end_dt, ts_keys_dailies)

    had_ts = (
        any([topic_sessions, chapter_sessions, lesson_sessions, perf_rows, daily_logs])
        or any(_pick_first_ts(x, ts_keys_sessions) for x in (topic_sessions_all + chapter_sessions_all + lesson_sessions_all))
        or any(_pick_first_ts(x, ts_keys_perf) for x in perf_all)
        or any(_pick_first_ts(x, ts_keys_dailies) for x in daily_logs_all)
    )

    total_time = sum(_get_time_spent(r) for r in topic_sessions) + \
                 sum(_get_time_spent(r) for r in chapter_sessions) + \
                 sum(float(r.get("time_spent", 0) or 0) for r in daily_logs) + \
                 sum(_get_time_spent(r) for r in lesson_sessions)

    session_lengths = [*[_get_time_spent(r) for r in topic_sessions],
                       *[_get_time_spent(r) for r in chapter_sessions],
                       *[float(r.get("time_spent", 0) or 0) for r in daily_logs],
                       *[_get_time_spent(r) for r in lesson_sessions]]
    session_lengths = [s for s in session_lengths if s and s > 0]
    sessions_count = len(session_lengths)
    avg_session_len = mean(session_lengths) if session_lengths else 0.0

    completed = sum(1 for t in topic_sessions if float(t.get("completion_percent") or 0) >= 80)
    total_lessons = len(topic_sessions)
    completion_pct = _pct(completed, total_lessons)

    # unique active days across all collections
    day_set = set()

    def _collect_dates(coll, keys):
        for r in coll:
            ts = _pick_first_ts(r, keys)
            if ts:
                day_set.add(ts.date())

    _collect_dates(topic_sessions, ts_keys_sessions)
    _collect_dates(chapter_sessions, ts_keys_sessions)
    _collect_dates(lesson_sessions, ts_keys_sessions)
    _collect_dates(daily_logs, ts_keys_dailies)
    _collect_dates(perf_rows, ts_keys_perf)
    active_days = len(day_set)

    scores = []
    for p in perf_rows:
        sv = p.get("score", None)
        if sv in (None, ""):
            sv = 100.0 if p.get("is_right") else 0.0
        scores.append(float(sv))
    avg_score = mean(scores) if scores else 0.0

    return {
        "had_ts": had_ts,
        "total_time_mins": round(float(total_time), 1),
        "sessions": sessions_count,
        "avg_session_mins": round(float(avg_session_len), 1),
        "lessons_done": completed,
        "lessons_total": total_lessons,
        "completion_pct": round(float(completion_pct), 1),
        "active_days": active_days,
        "avg_score": round(float(avg_score), 1),
    }


# -----------------------------
# Public: trends & focus score
# -----------------------------
def compute_trend(curr: float, prev: float) -> int:
    if prev <= 0:
        return 0
    return int(round(100 * (curr - prev) / prev))


def compute_focus_score(completion_pct: float, avg_session_mins: float) -> int:
    base = 50.0
    base += (completion_pct - 50.0) * 0.4
    base += (avg_session_mins - 10.0) * 1.2
    return int(_clamp(base, 0, 100))


# ============================================================
# Advanced Learning KPIs
# ============================================================
def accuracy_and_mastery(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Returns:
      {
        "overall": float,             # overall accuracy %
        "subjects": {subject: avg%},  # dict for bar chart
        "subjects_meta": [            # optional richer rows
            {"subject": str, "avg": float, "band": str, "attempts": int}
        ]
      }
    """
    idx = build_indexes(data)
    rows: List[Dict[str, Any]] = []
    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        score = ap.get("score")
        if score in (None, ""):
            score = 100.0 if ap.get("is_right") else 0.0
        subj = perf_subject(ap, idx) or "Math"
        rows.append({"subject": subj, "score": float(score)})

    if not rows:
        return {"overall": 0.0, "subjects": {}, "subjects_meta": []}

    df = pd.DataFrame(rows)
    overall = float(df["score"].mean())

    def band(x: float) -> str:
        return "Proficient" if x >= 75 else ("Developing" if x >= 50 else "Emerging")

    grp = (
        df.groupby("subject")
          .agg(avg=("score", "mean"), attempts=("score", "size"))
          .reset_index()
    )
    grp["avg"] = grp["avg"].round(1)
    grp["band"] = grp["avg"].apply(band)

    subjects_dict = {r["subject"]: float(r["avg"]) for r in grp.to_dict(orient="records")}
    subjects_meta = grp[["subject", "avg", "band", "attempts"]].to_dict(orient="records")

    return {"overall": overall, "subjects": subjects_dict, "subjects_meta": subjects_meta}


def response_time_stats(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Compute processing speed using TOTAL time actually spent on attempts (time_spent)
    within the window. Returns dicts suitable for bar charts.

    Returns:
      {
        "mean": float, "median": float, "p90": float, "attempts": int,
        "per_subject": {subject: avg_time},
        "per_subject_meta": [
            {"subject": str, "avg_time": float, "total_time": float, "attempts": int}
        ]
      }

    NOTE: If your time_spent is in minutes, these are minutes. If in seconds, adjust your labels in charts.
    """
    idx = build_indexes(data)
    rows: List[Dict[str, Any]] = []
    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        t = ap.get("time_spent", 0) or 0
        try:
            t = float(t)
        except Exception:
            t = 0.0
        subj = perf_subject(ap, idx) or "Math"
        if t > 0:
            rows.append({"subject": subj, "time_spent": t})

    if not rows:
        return {
            "mean": 0.0, "median": 0.0, "p90": 0.0, "attempts": 0,
            "per_subject": {}, "per_subject_meta": []
        }

    df = pd.DataFrame(rows)
    arr = df["time_spent"].to_numpy(dtype=float)
    mean_t = float(np.mean(arr))
    med_t  = float(np.median(arr))
    p90_t  = float(np.quantile(arr, 0.9))

    per_subj = (
        df.groupby("subject")
          .agg(total_time=("time_spent", "sum"), attempts=("time_spent", "size"), avg_time=("time_spent", "mean"))
          .reset_index()
    )
    per_subj["total_time"] = per_subj["total_time"].round(1)
    per_subj["avg_time"]   = per_subj["avg_time"].round(1)

    per_subject_dict = {r["subject"]: float(r["avg_time"]) for r in per_subj.to_dict(orient="records")}
    per_subject_meta = per_subj[["subject", "avg_time", "total_time", "attempts"]].to_dict(orient="records")

    return {
        "mean": round(mean_t, 1),
        "median": round(med_t, 1),
        "p90": round(p90_t, 1),
        "attempts": int(len(arr)),
        "per_subject": per_subject_dict,
        "per_subject_meta": per_subject_meta,
    }


def engagement_consistency(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """Active-days, sessions total, longest streak, weekly heat matrix (Mon..Sun x weeks)."""
    stamps: List[datetime] = []

    def add_ts(ts):
        if ts:
            stamps.append(ts)

    # Daily log
    for r in data.get("daily_activity_log", []):
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("login_timestamp"))
            if dt and start_dt <= dt <= end_dt:
                add_ts(dt)

    # Lesson / Topic / Chapter sessions
    for coll, keys in [
        (data.get("lesson_session", []), ["created_at", "started_at", "completed_at"]),
        (data.get("topic_session", []), ["started_at", "completed_at", "created_at"]),
        (data.get("chapter_session", []), ["started_at", "completed_at"]),
    ]:
        for r in coll:
            for k in keys:
                dt = parse_ts(r.get(k))
                if dt and start_dt <= dt <= end_dt:
                    add_ts(dt)

    if not stamps:
        return {"active_days": 0, "sessions_total": 0, "streak": 0, "weeks": [], "heat": []}

    # Active days & longest streak
    days = sorted(set([d.date() for d in stamps]))
    active_days = len(days)
    streak = best = 1
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1

    # Sessions per day for heatmap
    by_day = defaultdict(int)
    for dt in stamps:
        by_day[dt.date()] += 1

    # Build continuous weeks between start_dt..end_dt (Monday starts)
    start_w = start_dt.date() - timedelta(days=start_dt.weekday())
    end_d = end_dt.date()
    weeks = []
    cur = start_w
    while cur <= end_d:
        weeks.append(cur)
        cur += timedelta(days=7)

    heat = []  # list of columns; each is list[7] counts Mon..Sun
    for w in weeks:
        col = []
        for d in range(7):
            day = w + timedelta(days=d)
            col.append(by_day.get(day, 0))
        heat.append(col)

    return {
        "active_days": active_days,
        "sessions_total": len(stamps),
        "streak": best,
        "weeks": [w.isoformat() for w in weeks],
        "heat": heat,
    }


def independence_support(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, float]:
    """Blend hint usage, retry behaviour, and support feature usage into an independence index."""
    idx = build_indexes(data)
    hints = 0
    questions_attempted = 0
    retries = 0

    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        questions_attempted += 1
        if ap.get("used_hint") in (True, 1, "true", "True"):
            hints += 1
        attempt_count = 0
        try:
            attempt_count = int(ap.get("attempts", 0) or 0)
        except Exception:
            attempt_count = 0
        if attempt_count <= 0:
            attempt_count = 1
        if attempt_count > 1:
            retries += attempt_count - 1

    support_events: List[Dict[str, Any]] = []
    for evt in data.get("support_usage", []):
        if evt.get("user_id") != user_id:
            continue
        dt = parse_ts(evt.get("recorded_at") or evt.get("created_at") or evt.get("timestamp"))
        if dt and start_dt <= dt <= end_dt:
            support_events.append(evt)

    help_requests = 0
    assistive_feature_uses = 0
    accessibility_uses = 0
    for evt in support_events:
        count = evt.get("count", 1)
        try:
            count = int(count)
        except Exception:
            count = 1
        event_type = str(evt.get("event_type", "")).lower()
        feature = str(evt.get("feature", "")).lower()
        if event_type == "help_request":
            help_requests += count
        else:
            assistive_feature_uses += count
        if any(token in feature for token in ("accessibility", "contrast", "font", "reader", "text_to_speech")):
            accessibility_uses += count

    if questions_attempted == 0:
        return {
            "hint_rate": 0.0,
            "retry_rate": 0.0,
            "support_rate": 0.0,
            "help_requests": float(help_requests),
            "support_feature_uses": float(assistive_feature_uses),
            "accessibility_uses": float(accessibility_uses),
            "independence": 0.0,
        }

    hint_rate = 100.0 * hints / questions_attempted
    retry_rate = 100.0 * retries / questions_attempted
    support_rate = 0.0
    if questions_attempted:
        support_rate = 100.0 * (help_requests + assistive_feature_uses) / questions_attempted

    independence = max(0.0, 100.0 - (0.5 * hint_rate + 0.3 * retry_rate + 0.2 * support_rate))
    return {
        "hint_rate": round(hint_rate, 1),
        "retry_rate": round(retry_rate, 1),
        "support_rate": round(support_rate, 1),
        "help_requests": float(help_requests),
        "support_feature_uses": float(assistive_feature_uses),
        "accessibility_uses": float(accessibility_uses),
        "independence": round(independence, 1),
    }


def communication_social(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, float]:
    """
    If your dataset has 'messages', count them within the window.
    As a fallback, count personalisation variants used in the window as a proxy.
    """
    msgs = 0
    if "messages" in data:
        for m in data["messages"]:
            if m.get("user_id") == user_id:
                dt = parse_ts(m.get("created_at") or m.get("timestamp"))
                if dt and start_dt <= dt <= end_dt:
                    msgs += 1

    # Proxy via personalisation usage
    try:
        from ui_personalisation import collect_personalisation_usage
        usage = collect_personalisation_usage(data, user_id, start_dt, end_dt)
        changes = sum(1 for d in usage.values() for _ in d.keys())
    except Exception:
        changes = 0

    comm_events: List[Dict[str, Any]] = []
    for evt in data.get("communication_events", []):
        if evt.get("user_id") != user_id:
            continue
        dt = parse_ts(evt.get("timestamp") or evt.get("created_at"))
        if dt and start_dt <= dt <= end_dt:
            comm_events.append({**evt, "_dt": dt})

    interactions = len(comm_events)
    student_initiated = sum(
        1 for evt in comm_events if str(evt.get("initiated_by", "")).lower() == "student"
    )
    avg_length = mean(
        [float(evt.get("message_length", 0) or 0.0) for evt in comm_events]
    ) if comm_events else 0.0
    avg_turns = mean(
        [float(evt.get("turns", 0) or 0.0) for evt in comm_events]
    ) if comm_events else 0.0
    longest = max(comm_events, key=lambda e: float(e.get("message_length") or 0.0), default=None)
    last_evt = max(comm_events, key=lambda e: e["_dt"], default=None)

    return {
        "messages": float(msgs),
        "personalisation_changes": float(changes),
        "interactions": float(interactions),
        "student_initiated": float(student_initiated),
        "avg_length": round(avg_length, 1),
        "avg_turns": round(avg_turns, 1),
        "last_interaction_type": last_evt.get("interaction_type") if last_evt else None,
        "longest_message_length": float(longest.get("message_length")) if longest else 0.0,
    }


def emotional_regulation_summary(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Summarise mood and sensory regulation data captured as Zones of Regulation entries.
    """
    entries: List[Dict[str, Any]] = []
    for rec in data.get("emotional_regulation_log", []):
        if rec.get("user_id") != user_id:
            continue
        dt = parse_ts(rec.get("recorded_at") or rec.get("timestamp") or rec.get("created_at"))
        if dt and start_dt <= dt <= end_dt:
            entries.append({**rec, "_dt": dt})

    if not entries:
        return {
            "records": 0,
            "zone_counts": {},
            "green_pct": 0.0,
            "stability_index": 0.0,
            "latest_zone": None,
            "latest_mood": None,
            "top_adjustments": [],
            "timeline": [],
        }

    zone_counts = Counter(
        (str(e.get("zone", "Unknown")).strip().title() or "Unknown") for e in entries
    )
    total = sum(zone_counts.values())
    green_pct = 100.0 * zone_counts.get("Green", 0) / total if total else 0.0

    weight_map = {"Green": 1.0, "Yellow": 0.6, "Blue": 0.8, "Red": 0.2}
    weighted_sum = sum(weight_map.get(zone, 0.5) * count for zone, count in zone_counts.items())
    stability_index = 100.0 * weighted_sum / total if total else 0.0

    adjustments_counter: Counter[str] = Counter()
    for e in entries:
        adjustments = e.get("sensory_adjustments")
        if isinstance(adjustments, dict):
            for key, val in adjustments.items():
                adjustments_counter[f"{key}:{val}"] += 1
        elif isinstance(adjustments, list):
            for item in adjustments:
                adjustments_counter[str(item)] += 1
        elif adjustments:
            adjustments_counter[str(adjustments)] += 1

    top_adjustments = [label for label, _ in adjustments_counter.most_common(3)]
    entries.sort(key=lambda r: r["_dt"])
    latest = entries[-1]
    timeline = [
        {"date": e["_dt"].date().isoformat(), "zone": str(e.get("zone"))}
        for e in entries[-5:]
    ]

    return {
        "records": float(total),
        "zone_counts": dict(zone_counts),
        "green_pct": round(green_pct, 1),
        "stability_index": round(stability_index, 1),
        "latest_zone": latest.get("zone"),
        "latest_mood": latest.get("mood_indicator"),
        "top_adjustments": top_adjustments,
        "timeline": timeline,
    }


def activity_attempt_profile(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Track attempts per activity, focusing on MCQs for now while remaining future-proof.
    """
    idx = build_indexes(data)
    attempts: List[Tuple[datetime, Dict[str, Any]]] = []
    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if dt and start_dt <= dt <= end_dt:
            attempts.append((dt, ap))

    if not attempts:
        return {
            "activities_recorded": 0,
            "mcq_attempted": 0,
            "mcq_correct_pct": 0.0,
            "mcq_avg_attempts": 0.0,
            "mcq_first_try_success_pct": 0.0,
            "attempt_details": [],
        }

    attempts.sort(key=lambda tup: tup[0])
    mcq_rows = [row for row in attempts if str(row[1].get("activity_type", "")).lower() == "mcq"]

    def _attempts_count(ap: Dict[str, Any]) -> int:
        try:
            val = int(ap.get("attempts", 0) or 0)
        except Exception:
            val = 0
        return max(val, 1)

    mcq_correct = sum(1 for _, ap in mcq_rows if ap.get("is_right") in (True, 1, "true", "True"))
    mcq_first_try = sum(
        1 for _, ap in mcq_rows if _attempts_count(ap) == 1 and ap.get("is_right") in (True, 1, "true", "True")
    )
    mcq_attempt_counts = [_attempts_count(ap) for _, ap in mcq_rows]

    details = []
    for dt, ap in attempts:
        attempts_taken = _attempts_count(ap)
        details.append(
            {
                "activity_id": ap.get("activity_id"),
                "activity_type": ap.get("activity_type"),
                "attempts": attempts_taken,
                "final_answer": ap.get("user_answer"),
                "is_right": bool(ap.get("is_right")),
                "submitted_at": dt.isoformat(),
            }
        )

    return {
        "activities_recorded": len(attempts),
        "mcq_attempted": len(mcq_rows),
        "mcq_correct_pct": round(_pct(mcq_correct, len(mcq_rows)), 1) if mcq_rows else 0.0,
        "mcq_avg_attempts": round(mean(mcq_attempt_counts), 2) if mcq_attempt_counts else 0.0,
        "mcq_first_try_success_pct": round(_pct(mcq_first_try, len(mcq_rows)), 1) if mcq_rows else 0.0,
        "attempt_details": details[:12],
    }

# ============================================================
# NEW: AI Literacy & Learning Gain
# ============================================================
def ai_literacy_stats(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Reads assessments from data['ai_literacy_assessment'].

    Expected minimal schema per row:
      {
        "user_id": int,
        "taken_at": ISO string,
        "type": "pre" | "post",
        "score": number,
        "max_score": number,
        "concepts_mastered": [str],    # optional
        "applications": [str]          # optional
      }

    Returns (gracefully empty if none):
      {
        "available": bool,
        "pre_score": float|None,
        "post_score": float|None,
        "max_score": float|None,
        "learning_gain": float|None,   # (%)
        "level_before": str|None,      # Beginner/Developing/Proficient
        "level_after": str|None,
        "concepts_mastered": [str],
        "applications": [str]
      }
    """
    rows = data.get("ai_literacy_assessment", [])
    if not rows:
        return {
            "available": False,
            "pre_score": None,
            "post_score": None,
            "max_score": None,
            "learning_gain": None,
            "level_before": None,
            "level_after": None,
            "concepts_mastered": [],
            "applications": [],
        }

    # Filter for this user and this window
    def _dt_ok(r):
        dt = parse_ts(r.get("taken_at"))
        return dt and (start_dt <= dt <= end_dt)

    user_rows = [r for r in rows if r.get("user_id") == user_id and _dt_ok(r)]
    if not user_rows:
        # Fall back to most recent pre/post if window filtering yields none
        user_rows = [r for r in rows if r.get("user_id") == user_id]

    if not user_rows:
        return {
            "available": False,
            "pre_score": None,
            "post_score": None,
            "max_score": None,
            "learning_gain": None,
            "level_before": None,
            "level_after": None,
            "concepts_mastered": [],
            "applications": [],
        }

    # Pick the latest pre and post by taken_at
    pre = None
    post = None
    for r in user_rows:
        t = (r.get("type") or "").lower()
        ts = parse_ts(r.get("taken_at"))
        if t == "pre":
            if (pre is None) or (ts and parse_ts(pre.get("taken_at") or "1970-01-01") and ts > parse_ts(pre["taken_at"])):
                pre = r
        elif t == "post":
            if (post is None) or (ts and parse_ts(post.get("taken_at") or "1970-01-01") and ts > parse_ts(post["taken_at"])):
                post = r

    def _score(r):  # (score, max)
        if not r:
            return None, None
        try:
            s = float(r.get("score", 0))
            m = float(r.get("max_score", 100) or 100)
        except Exception:
            return None, None
        return s, m

    pre_s, pre_m = _score(pre)
    post_s, post_m = _score(post)
    max_score = post_m or pre_m or 100.0

    def _level(score: Optional[float], maximum: float) -> Optional[str]:
        if score is None:
            return None
        pct = 100.0 * score / max(1.0, maximum)
        if pct >= 75:
            return "Proficient"
        if pct >= 50:
            return "Developing"
        return "Beginner"

    learning_gain = None
    if (pre_s is not None) and (post_s is not None) and (max_score and max_score > 0):
        learning_gain = round((post_s - pre_s) / max_score * 100.0, 1)

    concepts = (post or {}).get("concepts_mastered") or []
    if not isinstance(concepts, list):
        concepts = []
    apps = (post or {}).get("applications") or []
    if not isinstance(apps, list):
        apps = []

    return {
        "available": True if (pre or post) else False,
        "pre_score": pre_s,
        "post_score": post_s,
        "max_score": float(max_score) if max_score is not None else None,
        "learning_gain": learning_gain,
        "level_before": _level(pre_s, max_score) if pre_s is not None else None,
        "level_after": _level(post_s, max_score) if post_s is not None else None,
        "concepts_mastered": concepts,
        "applications": apps,
    }
