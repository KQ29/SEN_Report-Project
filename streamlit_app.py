# streamlit_app.py
import math
import operator
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from html import escape
from typing import List

from constants import PRIMARY_JSON_PATH, SECONDARY_JSON_PATH
from loaders import load_json, extract_user_id_and_audience, merge_data
from joins import build_indexes
from metrics_aggregate import aggregate_student
from metrics_period import (
    period_stats,
    available_date_range_for_user,
    compute_trend,
    compute_focus_score,
    compute_dropoff_risk,
    accuracy_and_mastery,
    response_time_stats,
    engagement_consistency,
    independence_support,
    communication_social,
    emotional_regulation_summary,
    activity_attempt_profile,
    ai_literacy_stats,          # NEW
)
from charts_plotly import (
    pie_for_score,
    pie_for_completed,
    alltime_kpi_bars,
    bar_mastery_subjects,
    bar_response_time_subjects,
)
from ui_profile import display_user_metadata
from ui_subjects import render_subject_growth
from ui_personalisation import render_personalisation_usage
from report_builder import build_report
from qwen_integration import generate_ai_report

st.set_page_config(page_title="Student Report Generator", layout="wide")

SEN_REPORT_CSS = """
<style>
[data-testid="stMain"] .sen-report {
    font-size: 0.95rem;
    background: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 0.75rem;
    padding: 1.1rem 1.35rem;
    margin-top: 0.75rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
}
[data-testid="stMain"] .sen-report__header {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem 1rem;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 1rem;
}
[data-testid="stMain"] .sen-report__title {
    font-weight: 600;
    font-size: 1.1rem;
}
[data-testid="stMain"] .sen-report__meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    font-size: 0.88rem;
    opacity: 0.85;
}
[data-testid="stMain"] .sen-report__meta span {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
[data-testid="stMain"] .sen-report__grid {
    display: grid;
    gap: 0.9rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
[data-testid="stMain"] .sen-card {
    background: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 0.65rem;
    padding: 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    min-height: 100%;
}
[data-testid="stMain"] .sen-card h4 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
}
[data-testid="stMain"] .sen-metric {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
}
[data-testid="stMain"] .sen-metric__label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}
[data-testid="stMain"] .sen-metric__value {
    font-weight: 500;
}
[data-testid="stMain"] .sen-metric__value--emph {
    font-weight: 600;
    font-size: 1.02rem;
}
[data-testid="stMain"] .sen-tag {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
[data-testid="stMain"] .sen-tag--low {
    background: rgba(106, 168, 79, 0.2);
    color: rgb(66, 115, 39);
}
[data-testid="stMain"] .sen-tag--medium {
    background: rgba(255, 193, 7, 0.22);
    color: rgb(166, 121, 0);
}
[data-testid="stMain"] .sen-tag--high {
    background: rgba(220, 53, 69, 0.22);
    color: rgb(156, 26, 37);
}
[data-testid="stMain"] .sen-report__list {
    margin: 0;
    padding-left: 1.15rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
[data-testid="stMain"] .sen-report__footer {
    margin-top: 1rem;
    display: grid;
    gap: 0.85rem;
}
[data-testid="stMain"] .sen-report__smallprint {
    font-size: 0.78rem;
    opacity: 0.65;
}
@media (max-width: 640px) {
    [data-testid="stMain"] .sen-report {
        padding: 1rem;
    }
    [data-testid="stMain"] .sen-metric {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
"""


def _inject_sen_report_css() -> None:
    if st.session_state.get("_sen_report_css_loaded"):
        return
    st.session_state["_sen_report_css_loaded"] = True
    st.markdown(SEN_REPORT_CSS, unsafe_allow_html=True)


def render_sen_report(report: dict) -> None:
    """Render a styled SEN report snapshot using the prepared report dict."""
    _inject_sen_report_css()

    def fmt_metric(value, unit="", decimals=1, plus_sign=False):
        if value in (None, "", "—", "missed"):
            return "missed"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return "missed"
            decimals_to_use = decimals
            if isinstance(value, int) or decimals == 0:
                decimals_to_use = 0
            if decimals_to_use == 0:
                formatted = f"{value:.0f}"
                formatted = formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
            else:
                formatted = f"{value:.{decimals_to_use}f}"
            sign = "+" if plus_sign and value > 0 else ""
            return f"{sign}{formatted}{unit}"
        return str(value)

    def metric(label, value, *, emphasize=False, raw=False):
        label_html = escape(label)
        if raw:
            value_html = value
        else:
            value_str = value if value not in (None, "", "—") else "missed"
            value_html = escape(value_str)
        value_class = "sen-metric__value sen-metric__value--emph" if emphasize else "sen-metric__value"
        return f"<div class='sen-metric'><span class='sen-metric__label'>{label_html}</span><span class='{value_class}'>{value_html}</span></div>"

    student = report.get("student", {})
    period = report.get("period", {})
    usage = report.get("usage", {})
    focus = report.get("focus", {})
    learning = report.get("learning", {})
    ai_support = report.get("ai_support", {})
    routine = report.get("routine", {})
    independence = report.get("independence", {})
    communication = report.get("communication", {})
    emotional = report.get("emotional_regulation", {})
    activity_profile = report.get("activity_performance", {})
    goals = report.get("goals") or []
    recommendations = report.get("recommendations") or []
    questions = report.get("questions") or {}
    drivers = {
        "Accuracy & Mastery": report.get("accuracy_mastery"),
        "Processing Speed": report.get("processing_speed"),
        "AI Literacy": report.get("ai_literacy"),
    }

    dropoff_raw = routine.get("dropoff_risk")
    dropoff_value = str(dropoff_raw).lower() if dropoff_raw else "missed"
    if dropoff_value in {"low", "medium", "high"}:
        dropoff_class = f"sen-tag sen-tag--{dropoff_value}"
        dropoff_html = f"<span class='{dropoff_class}'>{escape(dropoff_value.capitalize())}</span>"
    else:
        dropoff_html = "missed"

    lessons_done = usage.get("lessons_done")
    lessons_total = usage.get("lessons_total")

    def safe_value(val):
        if val in (None, "", "—"):
            return "missed"
        return val

    lessons_summary = f"{safe_value(lessons_done)} / {safe_value(lessons_total)}"

    period_range = f"{escape(period.get('start') or 'missed')} → {escape(period.get('end') or 'missed')}"
    generated_on = escape(period.get("generated_on") or "missed")
    prepared_for = report.get("prepared_for") or "missed"

    usage_card = "".join([
        metric("Active days", fmt_metric(usage.get("active_days"), decimals=0)),
        metric("Sessions", fmt_metric(usage.get("sessions"), decimals=0)),
        metric("Avg session", fmt_metric(usage.get("avg_session_mins"), unit=" mins")),
        metric("Time on task", fmt_metric(usage.get("total_time_mins"), unit=" mins")),
        metric("Completion", fmt_metric(usage.get("completion_pct"), unit="%", decimals=0)),
        metric("Trend vs prev", fmt_metric(usage.get("trend_vs_prev_pct"), unit="%", plus_sign=True)),
        metric("Lessons", lessons_summary),
    ])

    focus_card = "".join([
        metric("Focus score", fmt_metric(focus.get("focus_score"), decimals=0), emphasize=True),
        metric("Change vs prev", fmt_metric(focus.get("focus_score_delta"), decimals=1, plus_sign=True)),
        metric("Class median", fmt_metric(focus.get("class_median"), decimals=0)),
        metric("Avg sustained block", fmt_metric(focus.get("avg_sustained_block_mins"), unit=" mins")),
    ])

    learning_card = "".join([
        metric("Skills highlighted", str(len(learning.get("skills", [])))),
        metric("Perseverance index", fmt_metric(learning.get("perseverance_index"), decimals=1)),
        metric("Hints / activity", fmt_metric(ai_support.get("hints_per_activity"), decimals=2)),
    ])

    routine_card = "".join([
        metric("Drop-off risk", dropoff_html, raw=True, emphasize=True),
        metric("All-time active days", fmt_metric(routine.get("active_days_total"), decimals=0)),
        metric("Course completion", fmt_metric(routine.get("completion_pct_all"), unit="%", decimals=0)),
        metric("Avg session (all-time)", fmt_metric(routine.get("avg_session_all"), unit=" mins")),
        metric("Devices noted", fmt_metric(len(report.get("devices", {})), decimals=0)),
    ])

    independence_card = "".join([
        metric("Hint rate", fmt_metric(independence.get("hint_rate"), unit="%", decimals=1)),
        metric("Retry rate", fmt_metric(independence.get("retry_rate"), unit="%", decimals=1)),
        metric("Support features", fmt_metric(independence.get("support_feature_uses"), decimals=0)),
        metric("Accessibility uses", fmt_metric(independence.get("accessibility_uses"), decimals=0)),
        metric("Independence index", fmt_metric(independence.get("independence"), decimals=1), emphasize=True),
    ])

    communication_card = "".join([
        metric("Messages", fmt_metric(communication.get("messages"), decimals=0)),
        metric("Interactions", fmt_metric(communication.get("interactions"), decimals=0)),
        metric("Student initiated", fmt_metric(communication.get("student_initiated"), decimals=0)),
        metric("Avg length", fmt_metric(communication.get("avg_length"), decimals=1)),
        metric("Avg turns", fmt_metric(communication.get("avg_turns"), decimals=1)),
        metric("Last interaction", communication.get("last_interaction_type") or "missed"),
    ])

    adjustments_text = ", ".join(emotional.get("top_adjustments") or [])
    adjustments_text = adjustments_text if adjustments_text else "missed"
    timeline_entries = emotional.get("timeline") or []
    timeline_text = ", ".join(f"{item['date']} ({item['zone']})" for item in timeline_entries) if timeline_entries else "missed"
    emotional_card = "".join([
        metric("Entries", fmt_metric(emotional.get("records"), decimals=0)),
        metric("Current zone", emotional.get("latest_zone") or "missed"),
        metric("Latest mood", emotional.get("latest_mood") or "missed"),
        metric("Green time", fmt_metric(emotional.get("green_pct"), unit="%", decimals=1)),
        metric("Stability index", fmt_metric(emotional.get("stability_index"), unit="%", decimals=1), emphasize=True),
        metric("Avatar changes", fmt_metric(emotional.get("avatar_changes"), decimals=0)),
        metric("Fav avatar", emotional.get("favorite_avatar") or "missed"),
        metric("Background changes", fmt_metric(emotional.get("background_changes"), decimals=0)),
        metric("Fav background", emotional.get("favorite_background") or "missed"),
        metric("Sensory adjustments", adjustments_text),
        metric("Recent timeline", timeline_text),
    ])

    attempt_details = activity_profile.get("attempt_details") or []
    recent_attempt_text = "missed"
    if attempt_details:
        last_attempt = attempt_details[-1]
        status = "✅" if last_attempt.get("is_right") else "❌"
        recent_attempt_text = (
            f"{status} {last_attempt.get('activity_type', 'activity')} in {last_attempt.get('attempts') or 'missed'} attempt(s)"
        )
    activity_card = "".join([
        metric("Logged activities", fmt_metric(activity_profile.get("activities_recorded"), decimals=0)),
        metric("MCQ attempted", fmt_metric(activity_profile.get("mcq_attempted"), decimals=0)),
        metric("MCQ correct", fmt_metric(activity_profile.get("mcq_correct_pct"), unit="%", decimals=1)),
        metric("Avg attempts", fmt_metric(activity_profile.get("mcq_avg_attempts"), decimals=2)),
        metric("1st try success", fmt_metric(activity_profile.get("mcq_first_try_success_pct"), unit="%", decimals=1)),
        metric("Latest outcome", recent_attempt_text),
    ])

    driver_html = ""
    driver_rows = []
    if drivers["Accuracy & Mastery"]:
        acc = drivers["Accuracy & Mastery"]
        overall = fmt_metric(acc.get("overall"), unit="%", decimals=0)
        driver_rows.append(metric("Accuracy (overall)", overall, emphasize=True))
    if drivers["Processing Speed"]:
        spd = drivers["Processing Speed"]
        driver_rows.append(metric("Median response", fmt_metric(spd.get("median"), unit=" mins")))
        driver_rows.append(metric("P90 response", fmt_metric(spd.get("p90"), unit=" mins")))
    if drivers["AI Literacy"]:
        ai = drivers["AI Literacy"]
        if ai.get("available"):
            driver_rows.append(metric("Learning gain", fmt_metric(ai.get("learning_gain"), unit="%", decimals=0, plus_sign=True)))
            lv = f"{ai.get('level_before', 'missed')} → {ai.get('level_after', 'missed')}"
            driver_rows.append(metric("Level change", lv))
        else:
            driver_rows.append(metric("Learning gain", "Assessment not available"))
    if driver_rows:
        driver_html = f"<div class='sen-card'><h4>Key Drivers</h4>{''.join(driver_rows)}</div>"

    def render_list(title, items):
        if not items:
            return ""
        entries = "".join(f"<li>{escape(str(item))}</li>" for item in items if item not in (None, ""))
        if not entries:
            return ""
        return f"<div class='sen-card'><h4>{escape(title)}</h4><ul class='sen-report__list'>{entries}</ul></div>"

    def render_questions(title, mapping):
        if not mapping:
            return ""
        rows = []
        for prompt, answer in mapping.items():
            if answer in (None, "") and prompt in (None, ""):
                continue
            rows.append(
                f"<div class='sen-metric'><span class='sen-metric__label'>{escape(str(prompt or 'Question'))}</span>"
                f"<span class='sen-metric__value'>{escape(str(answer or 'missed'))}</span></div>"
            )
        if not rows:
            return ""
        return f"<div class='sen-card'><h4>{escape(title)}</h4>{''.join(rows)}</div>"

    grid_cards = [
        f"<div class='sen-card'><h4>Student</h4>"
        f"{metric('Name', student.get('name') or 'missed', emphasize=True)}"
        f"{metric('ID', str(student.get('id') or 'missed'))}"
        f"{metric('Class', student.get('class') or 'missed')}"
        f"{metric('Year', student.get('year') or 'missed')}"
        f"</div>",
        f"<div class='sen-card'><h4>Usage</h4>{usage_card}</div>",
        f"<div class='sen-card'><h4>Focus</h4>{focus_card}</div>",
        f"<div class='sen-card'><h4>Learning & Support</h4>{learning_card}{metric('Prepared for', prepared_for or 'missed')}</div>",
        f"<div class='sen-card'><h4>Routine</h4>{routine_card}</div>",
        f"<div class='sen-card'><h4>Emotional Regulation</h4>{emotional_card}</div>",
        f"<div class='sen-card'><h4>Independence & Support</h4>{independence_card}</div>",
        f"<div class='sen-card'><h4>Communication & Social</h4>{communication_card}</div>",
        f"<div class='sen-card'><h4>Activity Performance</h4>{activity_card}</div>",
    ]
    if driver_html:
        grid_cards.append(driver_html)

    footer_sections = [
        render_list("Goals", goals),
        render_list("Recommendations", recommendations),
        render_questions("Open Questions", questions),
    ]
    footer_sections = [section for section in footer_sections if section]

    footer_html = ""
    if footer_sections:
        footer_html = f"<div class='sen-report__footer'>{''.join(footer_sections)}</div>"

    audience_html = escape(prepared_for or "missed")

    report_html = f"""
    <div class="sen-report">
        <div class="sen-report__header">
            <div class="sen-report__title">SEN Report Snapshot</div>
            <div class="sen-report__meta">
                <span>Range: {period_range}</span>
                <span>Generated: {generated_on}</span>
                <span>Audience: {audience_html}</span>
            </div>
        </div>
        <div class="sen-report__grid">
            {''.join(grid_cards)}
        </div>
        {footer_html}
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)


def bar_period_kpis(labels, values, units):
    colors = ["#2E86AB", "#6AA84F"]
    texts = [f"{v:.1f}{u}" for v, u in zip(values, units)]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels, y=values,
                marker_color=colors,
                text=texts, textposition="outside", cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f}%{customdata}<extra></extra>",
                customdata=units,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=10, t=40, b=40),
        title="Period KPIs",
        xaxis=dict(tickangle=-10),
        yaxis=dict(rangemode="tozero"),
        showlegend=False,
    )
    return fig


def derive_goals_and_recommendations(curr, acc, spd, indep, emo, attempts, focus_score, dropoff):
    """Construct personalised goals and recommendations based on current period metrics."""
    goals: List[str] = []
    recs: List[str] = []

    def f(val, default=0.0) -> float:
        try:
            return float(val)
        except Exception:
            return default

    # --- Accuracy insights ---
    overall = f((acc or {}).get("overall"))
    subs = (acc or {}).get("subjects") or {}
    weakest_subj = min(subs, key=subs.get) if subs else None
    if overall and overall < 60:
        if weakest_subj:
            target = min(65, int(overall) + 5)
            recs.append(f"Prioritise {weakest_subj} with two short practice blocks this week.")
            goals.append(f"Raise {weakest_subj} accuracy to {target}%.")
        else:
            recs.append("Focus on core misconceptions; review missed answers right after each session.")
            goals.append("Lift overall accuracy by at least 5%.")
    elif overall:
        recs.append("Maintain accuracy by mixing new items with quick review questions.")
        goals.append("Hold accuracy within ±5% of the current level.")

    # --- Engagement & focus ---
    avg_session = f(curr.get("avg_session_mins"))
    dropoff = (dropoff or "").lower()
    if dropoff == "high" or avg_session < 5:
        recs.append("Schedule three 5–7 minute sessions on priority topics this week.")
        goals.append("Complete three short sessions before the next check-in.")
    elif dropoff == "medium" or avg_session < 10:
        recs.append("Use consistent ≤10 minute sessions to build routine momentum.")
        goals.append("Log activity on at least five days this week.")

    if focus_score and focus_score < 55:
        recs.append("Reduce distractions and start each session with one warm-up item.")
        goals.append("Raise focus score by 10 points.")

    # --- Independence cues ---
    hint_rate = f((indep or {}).get("hint_rate"))
    retry_rate = f((indep or {}).get("retry_rate"))
    if hint_rate > 30:
        recs.append("Try each item for 30 seconds unaided before opening a hint.")
        goals.append("Lower hint usage below 20%.")
    if retry_rate > 40:
        recs.append("Pause between retries; review a worked example before re-attempting.")
        goals.append("Keep average attempts per MCQ at or below 1.5.")

    # --- Activity attempts ---
    ft_success = f((attempts or {}).get("mcq_first_try_success_pct"))
    mcq_attempted = f((attempts or {}).get("mcq_attempted"))
    if ft_success and ft_success < 50 and mcq_attempted >= 5:
        recs.append("Add quick retrieval practice focused on first attempts for MCQs.")
        goals.append("Reach 60% first-try success.")

    # --- Processing speed ---
    median_rt = f((spd or {}).get("median"))
    if median_rt and median_rt > 2.0:
        recs.append("Use three-item timed drills (2–3 mins) to build fluency.")
        goals.append("Cut median response time by 20%.")

    # --- Emotional regulation ---
    green_pct = f((emo or {}).get("green_pct"))
    stability_idx = f((emo or {}).get("stability_index"))
    if (green_pct and green_pct < 40) or (stability_idx and stability_idx < 60):
        adjustments = ", ".join((emo or {}).get("top_adjustments") or []) or "preferred adjustments"
        recs.append(f"Open sessions with a regulation check-in; keep {adjustments} enabled.")
        goals.append("Achieve Green zone coverage of at least 50%.")

    # Fallback to ensure actionable output
    if not recs:
        recs.append("Keep regular, short sessions and review missed items together.")
    if not goals:
        goals.append("Complete two short sessions before the next review.")

    # Deduplicate while preserving order and cap list length
    seen = set()
    recs = [text for text in recs if not (text in seen or seen.add(text))][:5]
    seen.clear()
    goals = [text for text in goals if not (text in seen or seen.add(text))][:5]
    return goals, recs


def compute_report_payload(data, user_id, audience, start_date, end_date, data_label):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    agg = aggregate_student(data, user_id)
    curr = period_stats(data, user_id, start_dt, end_dt)
    prev_span_days = max(1, (end_dt.date() - start_dt.date()).days + 1)
    prev_end = start_dt - timedelta(seconds=1)
    prev_start = prev_end - timedelta(days=prev_span_days - 1)
    prev = period_stats(data, user_id, prev_start, prev_end)
    trend_vs_prev = compute_trend(curr["total_time_mins"], prev["total_time_mins"])

    focus_score_now = compute_focus_score(curr["completion_pct"], curr["avg_session_mins"])
    focus_score_prev = compute_focus_score(prev["completion_pct"], prev["avg_session_mins"])
    focus_delta = focus_score_now - focus_score_prev

    active_days_total = agg.get("active_days_total", 0)
    completion_pct_all = agg.get("lesson_completion_rate", 0)
    avg_time_all = agg.get("avg_session_length", 0)
    dropoff_risk = compute_dropoff_risk(active_days_total, completion_pct_all, avg_time_all)

    acc_block = accuracy_and_mastery(data, user_id, start_dt, end_dt)
    spd_block = response_time_stats(data, user_id, start_dt, end_dt)
    eng_block = engagement_consistency(data, user_id, start_dt, end_dt)
    indep_block = independence_support(data, user_id, start_dt, end_dt)
    comm_block = communication_social(data, user_id, start_dt, end_dt)
    emo_block = emotional_regulation_summary(data, user_id, start_dt, end_dt)
    attempts_block = activity_attempt_profile(data, user_id, start_dt, end_dt)
    ai_block = ai_literacy_stats(data, user_id, start_dt, end_dt)
    independence_for_report = {k: v for k, v in indep_block.items() if k not in {"support_rate", "help_requests"}}

    goals, recs = derive_goals_and_recommendations(
        curr=curr,
        acc=acc_block,
        spd=spd_block,
        indep=indep_block,
        emo=emo_block,
        attempts=attempts_block,
        focus_score=focus_score_now,
        dropoff=dropoff_risk,
    )

    report_data = {
        "student": {"name": agg["name"], "id": user_id, "class": agg.get("class_level", "missed"), "year": agg.get("class_level", "missed")},
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat(), "generated_on": date.today().isoformat()},
        "prepared_for": "Teacher" if audience == "teacher" else "Parent/Carer",
        "devices": {},
        "usage": {
            "active_days": curr.get("active_days", "missed"),
            "sessions": curr["sessions"],
            "avg_session_mins": curr["avg_session_mins"],
            "lessons_done": curr["lessons_done"],
            "lessons_total": curr["lessons_total"],
            "completion_pct": curr["completion_pct"],
            "total_time_mins": curr["total_time_mins"],
            "trend_vs_prev_pct": trend_vs_prev,
        },
        "focus": {
            "focus_score": focus_score_now,
            "focus_score_delta": focus_delta,
            "class_median": 62,
            "avg_sustained_block_mins": curr["avg_session_mins"],
        },
        "learning": {"skills": [], "perseverance_index": agg.get("avg_hints_used", "missed")},
        "language": {},
        "ai_support": {"hints_per_activity": agg.get("avg_hints_used", "missed")},
        "routine": {
            "dropoff_risk": dropoff_risk,
            "active_days_total": active_days_total,
            "completion_pct_all": completion_pct_all,
            "avg_session_all": avg_time_all,
        },
        "goals": goals,
        "recommendations": recs,
        "questions": {},
        # report drivers
        "accuracy_mastery": acc_block,
        "processing_speed": spd_block,
        "ai_literacy": ai_block,
        "independence": independence_for_report,
        "communication": comm_block,
        "emotional_regulation": emo_block,
        "activity_performance": attempts_block,
    }

    return {
        "user_id": user_id,
        "audience": audience,
        "start_date": start_date,
        "end_date": end_date,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "data_label": data_label,
        "agg": agg,
        "curr": curr,
        "prev": prev,
        "trend_vs_prev": trend_vs_prev,
        "acc_block": acc_block,
        "spd_block": spd_block,
        "eng_block": eng_block,
        "indep_block": indep_block,
        "comm_block": comm_block,
        "emo_block": emo_block,
        "attempts_block": attempts_block,
        "ai_block": ai_block,
        "independence_for_report": independence_for_report,
        "goals": goals,
        "recs": recs,
        "report_data": report_data,
        "focus_score_now": focus_score_now,
        "focus_delta": focus_delta,
        "dropoff_risk": dropoff_risk,
        "active_days_total": active_days_total,
        "completion_pct_all": completion_pct_all,
        "avg_time_all": avg_time_all,
    }


def collect_student_period_metrics(data, start_dt, end_dt):
    rows = []
    users = data.get("user", []) or []
    for user in users:
        uid = user.get("user_id")
        if uid is None:
            continue
        try:
            curr = period_stats(data, uid, start_dt, end_dt)
        except Exception:
            continue
        focus = compute_focus_score(curr.get("completion_pct", 0), curr.get("avg_session_mins", 0))
        rows.append({
            "user_id": uid,
            "name": user.get("name") or f"User {uid}",
            "focus_score": round(focus, 1) if focus is not None else None,
            "completion_pct": round(curr.get("completion_pct", 0), 1),
            "avg_session_mins": round(curr.get("avg_session_mins", 0), 1),
            "total_time_mins": round(curr.get("total_time_mins", 0), 1),
            "lessons_done": curr.get("lessons_done", 0),
            "lessons_total": curr.get("lessons_total", 0),
            "sessions": curr.get("sessions", 0),
            "active_days": curr.get("active_days", 0),
        })
    return rows


_FILTER_PATTERN = re.compile(r"\s*([a-zA-Z0-9_ ]+)\s*(<=|>=|==|=|!=|<|>)\s*(.+?)\s*")

FIELD_SYNONYMS = {
    "name": ["student", "student_name", "learner"],
    "focus_score": ["focus", "focus score", "attention score"],
    "completion_pct": ["completion", "completion percent", "progress %"],
    "avg_session_mins": ["avg session", "average session", "session length"],
    "total_time_mins": ["time on task", "total time"],
    "lessons_done": ["lessons", "lessons completed", "topics done"],
    "lessons_total": ["lessons total", "topics total"],
    "sessions": ["session_count", "sessions total"],
    "active_days": ["active days", "days active"],
}

def _normalize_field_name(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", field.strip().lower())


def resolve_field_name(field: str, df: pd.DataFrame) -> str:
    normalized = _normalize_field_name(field)
    if normalized in df.columns:
        return normalized
    for canonical, synonyms in FIELD_SYNONYMS.items():
        all_names = [canonical] + synonyms
        for name in all_names:
            if _normalize_field_name(name) == normalized and canonical in df.columns:
                return canonical
    return None


def apply_filter_expression(df: pd.DataFrame, expr: str):
    expr = expr.strip()
    if not expr:
        return df, None

    ops = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def _apply_single(condition: str):
        condition = condition.strip()
        if not condition:
            return pd.Series(True, index=df.index)
        match = _FILTER_PATTERN.match(condition)
        if not match:
            raise ValueError(f"Invalid clause '{condition}'. Example: focus_score <= 55")
        field_raw, op, value_raw = match.groups()
        field = resolve_field_name(field_raw, df)
        if field is None:
            raise ValueError(f"Unknown field '{field_raw}'.")
        series = df[field]
        comparator = ops[op]
        if pd.api.types.is_numeric_dtype(series):
            try:
                value = float(value_raw)
            except ValueError:
                raise ValueError(f"'{field_raw}' expects a numeric value.")
            left = pd.to_numeric(series, errors="coerce")
            return comparator(left, value)
        value = str(value_raw).strip().strip('"\'')
        left = series.astype(str).str.lower()
        return comparator(left, value.lower())

    try:
        and_parts = [p.strip() for p in re.split(r"&{2}", expr) if p.strip()]
        if and_parts:
            mask = pd.Series(True, index=df.index)
            for part in and_parts:
                or_parts = [c.strip() for c in re.split(r"\|\|", part) if c.strip()]
                if not or_parts:
                    continue
                or_mask = pd.Series(False, index=df.index)
                for clause in or_parts:
                    or_mask |= _apply_single(clause)
                mask &= or_mask
        else:
            mask = _apply_single(expr)
    except ValueError as e:
        human_fields = ", ".join(sorted(df.columns))
        return df, f"{e} Available fields: {human_fields}"

    filtered = df[mask.fillna(False)]
    return filtered, None


def _clean_name_query(text: str) -> str:
    text = re.sub(r"for\s+(teacher|parent|carer)\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"user[_\s]?id\s*\d+", "", text, flags=re.IGNORECASE)
    return text.strip()


def match_user_by_name(query: str, users: List[dict]):
    cleaned = _clean_name_query(query or "")
    tokens = [tok for tok in re.split(r"\s+", cleaned.lower()) if tok]
    if not tokens:
        return None, None
    matches = []
    for user in users:
        uid = user.get("user_id")
        name = (user.get("name") or "").strip()
        if not name or uid is None:
            continue
        name_lower = name.lower()
        if all(tok in name_lower for tok in tokens):
            matches.append(user)
    if not matches:
        return None, f"No student matches '{cleaned}'."
    uids = {m.get("user_id") for m in matches if m.get("user_id") is not None}
    if len(uids) > 1:
        names = ", ".join(sorted({m.get("name") or str(m.get("user_id")) for m in matches}))
        return None, f"Multiple students match '{cleaned}': {names}. Please specify the full name."
    uid = next(iter(uids), None)
    return uid, None


def main():
    st.title("Student Report Generator (merged datasets)")

    # -------- Data sources --------
    uploaded = st.sidebar.file_uploader("Upload one or more JSON files", type=["json"], accept_multiple_files=True)
    if uploaded:
        datasets = [load_json(u) for u in uploaded]
        data = merge_data(*datasets)
        data_label = " + ".join([u.name for u in uploaded])
    else:
        datasets = []
        labels = []
        try:
            ds1 = load_json(PRIMARY_JSON_PATH)
        except Exception:
            ds1 = {}
        if ds1:
            datasets.append(ds1)
            labels.append(PRIMARY_JSON_PATH or "missed")
        if SECONDARY_JSON_PATH:
            try:
                ds2 = load_json(SECONDARY_JSON_PATH)
            except Exception:
                ds2 = {}
            if ds2:
                datasets.append(ds2)
                labels.append(SECONDARY_JSON_PATH)
        if not datasets:
            st.error("No data loaded. Upload JSON files or set valid defaults.")
            return
        data = merge_data(*datasets)
        data_label = " + ".join(labels) if labels else "missed"
    st.sidebar.success(f"Loaded: {data_label}")

    # -------- Query / user selection --------
    users_df = pd.DataFrame(data.get("user", []))
    if not users_df.empty:
        cols = [c for c in ["user_id", "name", "class_level", "grade_level", "email"] if c in users_df.columns]
        if cols:
            st.markdown("### Students roster")
            st.dataframe(users_df[cols], use_container_width=True, hide_index=True)
    query = st.text_input(
        "Query",
        value="",
        placeholder="e.g. Mia Sanchez for teacher",
    )
    user_id, audience = extract_user_id_and_audience(query)
    name_error = None
    if user_id is None and not users_df.empty:
        user_id, name_error = match_user_by_name(query, users_df.to_dict("records"))
    if user_id is None:
        if name_error:
            st.warning(name_error)
        else:
            st.warning("Type a student's full name (e.g. 'Ava Patel') or specify `user_id 1`.")
        return

    idx = build_indexes(data)
    rec_start, rec_end = available_date_range_for_user(data, user_id, idx)

    today = date.today()
    default_end = rec_end.date() if rec_end else today
    default_start = rec_start.date() if rec_start else (default_end - timedelta(days=6))

    st.sidebar.caption("Pick a date range that overlaps your user's events.")
    start_date = st.sidebar.date_input("Report start date", value=default_start)
    end_date = st.sidebar.date_input("Report end date", value=default_end)

    if rec_start and rec_end:
        st.info(f"Recommended range for user {user_id}: **{rec_start.date()} → {rec_end.date()}**")

    # -------- Run --------
    run_clicked = st.button("Run")
    state_key = f"report_payload_{user_id}"
    if run_clicked:
        try:
            payload = compute_report_payload(data, user_id, audience, start_date, end_date, data_label)
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.session_state.pop(state_key, None)
            return
        st.session_state[state_key] = payload
        st.session_state.pop(f"ai_report_{user_id}", None)
        st.session_state.pop(f"ai_report_error_{user_id}", None)

    payload = st.session_state.get(state_key)
    if not payload:
        st.info("Set your filters and click Run to generate the report.")
        return
    if (
        payload["start_date"] != start_date
        or payload["end_date"] != end_date
        or payload.get("data_label") != data_label
        or payload.get("audience") != audience
    ):
        st.warning("Inputs changed — click Run to refresh the report.")
        return

    agg = payload["agg"]
    curr = payload["curr"]
    trend_vs_prev = payload["trend_vs_prev"]
    acc_block = payload["acc_block"]
    spd_block = payload["spd_block"]
    eng_block = payload["eng_block"]
    indep_block = payload["indep_block"]
    comm_block = payload["comm_block"]
    emo_block = payload["emo_block"]
    attempts_block = payload["attempts_block"]
    ai_block = payload["ai_block"]
    independence_for_report = payload["independence_for_report"]
    goals = payload["goals"]
    recs = payload["recs"]
    report_data = payload["report_data"]
    start_dt = payload["start_dt"]
    end_dt = payload["end_dt"]

    # ----- Profile -----
    display_user_metadata(agg)

    # ----- Period KPI pies -----
    st.markdown("### Period KPI charts")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(pie_for_score(curr.get("avg_score", 0)), use_container_width=True)
    with r1c2:
        st.plotly_chart(
            pie_for_completed(curr["lessons_done"], curr["lessons_total"], "Lessons completed (topics ≥80%)"),
            use_container_width=True
        )

    # ----- Period KPI combined bars (no Altair) -----
    labels = ["Avg session length (period)", "Time-on-task (period)"]
    values = [curr["avg_session_mins"], curr["total_time_mins"]]
    units  = [" mins", " mins"]
    st.plotly_chart(bar_period_kpis(labels, values, units), use_container_width=True)

    # ----- All-time KPIs -----
    st.markdown("### All-time KPI charts")
    labels = [
        "All-time points",
        "All-time avg session",
        f"Avg chapter progress ({agg['chapters_seen']} seen) (%)",
        "Hints usage (all-time) (%)",
    ]
    values = [
        float(agg["total_points"]),
        float(agg["avg_session_length"]),
        float(agg["avg_chapter_progress_val"]),
        float(agg["avg_hints_used"] * 100.0),
    ]
    units = ["", " mins", " %", " %"]
    st.plotly_chart(alltime_kpi_bars(labels, values, units), use_container_width=True)

    # ---------- Advanced Learning KPIs ----------
    st.markdown("### Advanced Learning KPIs")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Accuracy & Mastery",
        "Processing Speed",
        "Engagement & Consistency",
        "Independence",
        "Communication",
        "Emotional Regulation",
        "Activity Performance",
        "AI Literacy",
    ])

    with tab1:
        st.caption(f"Overall accuracy: **{acc_block['overall']:.1f}%**")
        st.plotly_chart(bar_mastery_subjects(acc_block["subjects"]), use_container_width=True)

    with tab2:
        st.caption(
            f"Attempts: **{spd_block['attempts']}** • Mean: **{spd_block['mean']}** • "
            f"Median: **{spd_block['median']}** • P90: **{spd_block['p90']}**"
        )
        st.plotly_chart(bar_response_time_subjects(spd_block["per_subject"], unit_label=" mins"), use_container_width=True)

    with tab3:
        st.write("Active days:", eng_block["active_days"])
        st.write("Total sessions:", eng_block["sessions_total"])
        st.write("Longest streak:", eng_block["streak"])
        weeks = eng_block.get("weeks", [])
        st.caption(f"Weeks with activity: {sum(1 for col in eng_block.get('heat', []) if sum(col) > 0)} / {len(weeks)}")

    with tab4:
        st.write("Hint rate:", f"{indep_block['hint_rate']}%")
        st.write("Retry rate:", f"{indep_block['retry_rate']}%")
        st.write("Support rate:", f"{indep_block['support_rate']}%")
        st.write("Help requests:", int(indep_block["help_requests"]))
        st.write("Support features used:", int(indep_block["support_feature_uses"]))
        st.write("Accessibility toggles:", int(indep_block["accessibility_uses"]))
        st.write("Independence index:", f"{indep_block['independence']}")

    with tab5:
        st.write("Messages (period):", comm_block["messages"])
        st.write("Personalisation changes (proxy):", comm_block["personalisation_changes"])
        st.write("Avatar/Text interactions:", comm_block["interactions"])
        st.write("Student-initiated:", comm_block["student_initiated"])
        st.write("Avg length (chars):", comm_block["avg_length"])
        st.write("Avg turns:", comm_block["avg_turns"])
        st.write("Last interaction type:", comm_block.get("last_interaction_type") or "missed")

    with tab6:
        if emo_block["records"]:
            st.write("Entries captured:", int(emo_block["records"]))
            st.write("Latest zone:", emo_block.get("latest_zone") or "missed")
            st.write("Latest mood:", emo_block.get("latest_mood") or "missed")
            st.write("Green time:", f"{emo_block['green_pct']}%")
            st.write("Stability index:", f"{emo_block['stability_index']}%")
            adjustments = emo_block.get("top_adjustments") or []
            st.write("Top sensory adjustments:", ", ".join(adjustments) if adjustments else "missed")
            timeline = emo_block.get("timeline") or []
            if timeline:
                timeline_text = ", ".join(f"{item['date']} ({item['zone']})" for item in timeline)
                st.write("Recent timeline:", timeline_text)
        else:
            st.info("No emotional regulation entries in this period.")

    with tab7:
        st.write("Activities recorded:", attempts_block["activities_recorded"])
        st.write("MCQ attempted:", attempts_block["mcq_attempted"])
        st.write("MCQ correct:", f"{attempts_block['mcq_correct_pct']}%")
        st.write("Avg attempts (MCQ):", attempts_block["mcq_avg_attempts"])
        st.write("First-try success:", f"{attempts_block['mcq_first_try_success_pct']}%")
        detail_df = pd.DataFrame(attempts_block["attempt_details"])
        if not detail_df.empty:
            st.dataframe(detail_df)
        else:
            st.info("No activity performance captured for this period.")

    with tab8:
        if ai_block.get("available"):
            st.write(f"Pre-test: {ai_block.get('pre_score', 'missed')} / {int(ai_block.get('max_score') or 100)}")
            st.write(f"Post-test: {ai_block.get('post_score', 'missed')} / {int(ai_block.get('max_score') or 100)}")
            st.write(f"Learning Gain: {ai_block.get('learning_gain', 'missed')}%")
            st.write(f"Level (before → after): {ai_block.get('level_before', 'missed')} → {ai_block.get('level_after', 'missed')}")
            concepts = ai_block.get("concepts_mastered") or []
            apps = ai_block.get("applications") or []
            if concepts:
                st.write("Key Concepts Mastered:", ", ".join(concepts))
            if apps:
                st.write("Skill Applications:", "; ".join(apps))
        else:
            st.info("AI Literacy assessment not found. Add 'ai_literacy_assessment' to your JSON to enable this section.")

    # ---------- Personalisation usage ----------
    st.markdown("### Personalisation usage")
    render_personalisation_usage(data, user_id, start_dt, end_dt)

    # ---------- Subject growth ----------
    render_subject_growth(agg)

    # ---------- SEN Report ----------
    st.subheader("🧾 SEN Report (auto-generated)")
    if not curr["had_ts"]:
        st.warning("No reliable timestamps found in your selected range.")

    render_sen_report(report_data)

    report_text = build_report(report_data)
    st.text_area("Report (copy-ready)", value=report_text, height=600)

    st.subheader("🤖 AI Narrative Draft")
    ai_state_key = f"ai_report_{user_id}"
    ai_error_key = f"ai_report_error_{user_id}"
    generated_ai = st.session_state.get(ai_state_key)
    ai_error = st.session_state.get(ai_error_key)
    if st.button("Generate AI narrative", key=f"ai_btn_{user_id}"):
        st.session_state.pop(ai_error_key, None)
        with st.spinner("Calling Qwen2.5-0.5B-Instruct..."):
            try:
                generated_ai = generate_ai_report(report_data, max_new_tokens=420)
                st.session_state[ai_state_key] = generated_ai
            except Exception as exc:
                st.session_state[ai_error_key] = str(exc)
                ai_error = str(exc)
    if ai_error:
        st.error(f"AI report generation failed: {ai_error}")
    if generated_ai:
        st.text_area("AI-generated summary", value=generated_ai, height=400)
    else:
        st.caption("Click \"Generate AI narrative\" to have Qwen draft a human-readable summary from these metrics.")

    st.markdown("### 🔎 Multi-student metric search")
    metrics_cache_key = f"student_metrics_{payload['data_label']}_{payload['start_date']}_{payload['end_date']}"
    metrics_rows = st.session_state.get(metrics_cache_key)
    if metrics_rows is None:
        metrics_rows = collect_student_period_metrics(data, start_dt, end_dt)
        st.session_state[metrics_cache_key] = metrics_rows

    df_metrics = pd.DataFrame(metrics_rows)
    if df_metrics.empty:
        st.info("No comparable student metrics found for this date range.")
    else:
        synonym_lines = []
        for canonical, names in FIELD_SYNONYMS.items():
            if canonical in df_metrics.columns:
                human_list = ", ".join([canonical] + names)
                synonym_lines.append(f"- **{canonical}** → {human_list}")
        if synonym_lines:
            st.caption(
                "You can filter by any of these fields/synonyms:\n" + "\n".join(synonym_lines)
            )
        filter_expr = st.text_input(
            "Filter expression",
            key=f"search_expr_{user_id}",
            placeholder="focus_score <= 50",
            help="Use field comparisons such as `focus_score <= 50`, `completion_pct > 80`, `name = Ava Patel`."
        )
        filtered_df = df_metrics
        warning_msg = None
        if filter_expr:
            filtered_df, warning_msg = apply_filter_expression(df_metrics, filter_expr)
        if warning_msg:
            st.warning(warning_msg)
        if filtered_df.empty:
            st.info("No students matched that filter.")
        else:
            st.dataframe(
                filtered_df.sort_values(by="focus_score", ascending=True, na_position="last"),
                use_container_width=True,
            )

if __name__ == "__main__":
    main()
