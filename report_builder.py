# report_builder.py
from datetime import datetime
from typing import Dict, Any, List


def _fmt(value, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return f"{value}{suffix}"
    if num.is_integer():
        num_str = f"{int(num)}"
    else:
        num_str = f"{num:.1f}"
    return f"{num_str}{suffix}"


def _fmt_delta(delta, suffix: str = "") -> str:
    if delta in (None, "", "—"):
        return "—"
    try:
        val = float(delta)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if val >= 0 else "-"
    mag = abs(val)
    mag_str = f"{int(mag)}" if mag.is_integer() else f"{mag:.1f}"
    return f"{sign}{mag_str}{suffix}"


def _section(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title) + "\n"


def _info_line(icon: str, label: str, value: Any, width: int = 24) -> str:
    return f"{icon} {label:<{width}}: {value}\n"


def _list_line(icon: str, text: str) -> str:
    return f"{icon} {text}\n"


def _bullet(items: List[str], icon: str = "•") -> str:
    return "".join(f"{icon} {item}\n" for item in items)


def _status_icon(value, thresholds=(55, 80)) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "⚪"
    if val >= thresholds[1]:
        return "🟢"
    if val >= thresholds[0]:
        return "🟡"
    return "🔴"


def _trend_icon(change) -> str:
    try:
        val = float(change)
    except (TypeError, ValueError):
        return "⚪"
    if val >= 5:
        return "🟢"
    if val <= -5:
        return "🔴"
    return "🟡"


def _risk_icon(risk: Any) -> str:
    if risk is None:
        return "⚪"
    lookup = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    return lookup.get(str(risk).lower(), "⚪")


def build_report(d: Dict[str, Any]) -> str:
    student = d.get("student", {})
    period = d.get("period", {})
    usage = d.get("usage", {})
    focus = d.get("focus", {})
    learning = d.get("learning", {})
    routine = d.get("routine", {})
    acc = d.get("accuracy_mastery", {})
    spd = d.get("processing_speed", {})
    ai = d.get("ai_literacy", {})

    focus_score = focus.get("focus_score")
    focus_delta = focus.get("focus_score_delta")
    completion_pct = usage.get("completion_pct")
    lessons_done = usage.get("lessons_done", 0)
    lessons_total = usage.get("lessons_total", 0)
    total_time = usage.get("total_time_mins")
    trend_vs_prev = usage.get("trend_vs_prev_pct")
    avg_session = usage.get("avg_session_mins")
    active_days = usage.get("active_days", "—")
    sessions = usage.get("sessions", "—")
    dropoff_risk = routine.get("dropoff_risk", "—")

    focus_badge = _status_icon(focus_score)
    completion_badge = _status_icon(completion_pct)
    trend_badge = _trend_icon(trend_vs_prev)
    risk_badge = _risk_icon(dropoff_risk)

    subs_meta = acc.get("subjects_meta") or []
    spd_attempts = spd.get("attempts", 0) if spd else 0
    spd_meta = spd.get("per_subject_meta", []) if spd else []

    out: List[str] = []

    title = "Student Learning Report (SEN)"
    out.append(title + "\n" + "=" * len(title) + "\n\n")

    out.append(_section("STUDENT SNAPSHOT"))
    out.append(_info_line("👤", "Student", student.get("name", "—")))
    out.append(_info_line("🆔", "Student ID", student.get("id", "—")))
    out.append(_info_line("🏫", "Class / Year", f"{student.get('class','—')} / {student.get('year','—')}"))
    out.append(_info_line("🗓️", "Reporting window", f"{period.get('start','—')} → {period.get('end','—')}"))
    out.append(_info_line("🎯", "Prepared for", d.get("prepared_for", "—")))
    out.append(
        _info_line(
            "🕒",
            "Generated on",
            period.get("generated_on") or datetime.today().date().isoformat(),
        )
    )
    out.append(
        _info_line(
            "🗂️",
            "Data sources",
            "activity_performance, chapter_session, topic_session, lesson_session, daily_activity_log, topics, enrollment",
        )
    )

    out.append(_section("EXECUTIVE SUMMARY"))
    out.append(_list_line("🟦", "Steady learning momentum with clear opportunities to deepen mastery."))
    out.append(
        _info_line(
            focus_badge,
            "Focus score",
            f"{_fmt(focus_score)} (Δ {_fmt_delta(focus_delta)})",
        )
    )
    out.append(
        _info_line(
            completion_badge,
            "Completion",
            f"{lessons_done}/{lessons_total} ({_fmt(completion_pct, '%')})",
        )
    )
    out.append(
        _info_line(
            trend_badge,
            "Time on task",
            f"{_fmt(total_time, ' mins')} (trend {_fmt_delta(trend_vs_prev, '%')})",
        )
    )

    out.append(_section("ENGAGEMENT & ROUTINE"))
    out.append(_info_line("📅", "Active days", active_days))
    out.append(
        _info_line(
            "⏱️",
            "Sessions",
            f"{sessions} (avg {_fmt(avg_session, ' mins')})",
        )
    )
    out.append(
        _info_line(
            risk_badge,
            "Drop-off risk",
            str(dropoff_risk).capitalize() if isinstance(dropoff_risk, str) else _fmt(dropoff_risk),
        )
    )

    out.append(_section("FOCUS & ATTENTION"))
    out.append(
        _list_line(
            "🎯",
            f"Focus score { _fmt(focus_score) } compared with class median { _fmt(focus.get('class_median')) }.",
        )
    )
    out.append(
        _info_line(
            "🧠",
            "Avg sustained block",
            _fmt(focus.get("avg_sustained_block_mins"), " mins"),
        )
    )

    out.append(_section("LEARNING PROGRESS & MASTERY"))
    if subs_meta:
        band_icon = {"Proficient": "🟢", "Developing": "🟡", "Emerging": "🔴"}
        for row in subs_meta:
            icon = band_icon.get(row.get("band"), "🔹")
            out.append(
                _list_line(
                    icon,
                    f"{row['subject']}: {row['avg']:.1f}% ({row['band']}; {row['attempts']} attempts)",
                )
            )
    else:
        out.append(_list_line("⚪", "No subject-level performance captured in the selected window."))
    out.append(
        _info_line(
            "🧩",
            "Perseverance (hints per attempt)",
            _fmt(learning.get("perseverance_index")),
        )
    )

    out.append(_section("PROCESSING SPEED"))
    if spd_attempts > 0:
        out.append(
            _list_line(
                "📊",
                f"Overall mean {_fmt(spd.get('mean'), ' mins')} • median {_fmt(spd.get('median'), ' mins')} • "
                f"p90 {_fmt(spd.get('p90'), ' mins')} (n={spd_attempts})",
            )
        )
        for row in spd_meta:
            out.append(
                _list_line(
                    "➤",
                    f"{row['subject']}: {_fmt(row['avg_time'], ' mins')} avg • {_fmt(row['total_time'], ' mins')} total • {row['attempts']} attempts",
                )
            )
    else:
        out.append(_list_line("⚪", "No processing speed data recorded for this period."))

    out.append(_section("AI LITERACY & LEARNING GAIN"))
    if ai and ai.get("available"):
        out.append(_info_line("🧠", "Pre-test", f"{_fmt(ai.get('pre_score'))} / {_fmt(ai.get('max_score'))}"))
        out.append(_info_line("🚀", "Post-test", f"{_fmt(ai.get('post_score'))} / {_fmt(ai.get('max_score'))}"))
        out.append(_info_line("📈", "Learning gain", _fmt(ai.get("learning_gain"), "%")))
        out.append(
            _info_line(
                "🎓",
                "Level shift",
                f"{ai.get('level_before','—')} → {ai.get('level_after','—')}",
            )
        )
        concepts = ai.get("concepts_mastered") or []
        applications = ai.get("applications") or []
        if concepts:
            out.append(_list_line("🟪", "Key concepts: " + ", ".join(concepts)))
        if applications:
            out.append(_list_line("🟦", "Applications: " + "; ".join(applications)))
    else:
        out.append(_list_line("⚪", "No AI literacy assessments recorded in the selected window."))

    out.append(_section("COMMUNICATION & SUPPORT"))
    out.append(
        _list_line(
            "🤖",
            "Derived from hints and retries; native AI interaction fields are not available in this dataset.",
        )
    )
    out.append(
        _info_line(
            "💡",
            "Hints per attempt",
            _fmt(learning.get("perseverance_index")),
        )
    )

    out.append(_section("TECHNOLOGY & ACCESSIBILITY"))
    out.append(_list_line("💻", "Device telemetry is incomplete in the supplied dataset."))

    recommendations = d.get("recommendations") or [
        "Encourage regular short practice sessions (5–7 mins) on priority subjects.",
        "Review missed questions together to reinforce strategies.",
        "Maintain shorter, high-quality sessions if average length is below 10 mins.",
    ]
    out.append(_section("GOALS & RECOMMENDATIONS"))
    out.append(_bullet(recommendations, icon="✅"))

    out.append(_section("OPEN QUESTIONS"))
    out.append(_list_line("ℹ️", "The dataset does not capture unanswered or out-of-scope questions."))

    return "".join(out)
