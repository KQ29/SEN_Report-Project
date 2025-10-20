# streamlit_app.py
import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from html import escape

from constants import PRIMARY_JSON_PATH, SECONDARY_JSON_PATH
from loaders import load_json, extract_user_id_and_audience, merge_data
from joins import build_indexes
from metrics_aggregate import aggregate_student
from metrics_period import (
    period_stats,
    available_date_range_for_user,
    compute_trend,
    compute_focus_score,
    accuracy_and_mastery,
    response_time_stats,
    engagement_consistency,
    independence_support,
    communication_social,
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
        if value in (None, "", "—"):
            return "—"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return "—"
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
            value_html = escape(value if value not in (None, "") else "—")
        value_class = "sen-metric__value sen-metric__value--emph" if emphasize else "sen-metric__value"
        return f"<div class='sen-metric'><span class='sen-metric__label'>{label_html}</span><span class='{value_class}'>{value_html}</span></div>"

    student = report.get("student", {})
    period = report.get("period", {})
    usage = report.get("usage", {})
    focus = report.get("focus", {})
    learning = report.get("learning", {})
    ai_support = report.get("ai_support", {})
    routine = report.get("routine", {})
    goals = report.get("goals") or []
    recommendations = report.get("recommendations") or []
    questions = report.get("questions") or {}
    drivers = {
        "Accuracy & Mastery": report.get("accuracy_mastery"),
        "Processing Speed": report.get("processing_speed"),
        "AI Literacy": report.get("ai_literacy"),
    }

    dropoff = (routine.get("dropoff_risk") or "—").lower()
    dropoff_class = f"sen-tag sen-tag--{dropoff}" if dropoff in {"low", "medium", "high"} else "sen-tag"
    dropoff_copy = dropoff.capitalize() if dropoff not in {"", "—"} else "—"
    dropoff_html = f"<span class='{dropoff_class}'>{escape(dropoff_copy)}</span>" if dropoff_copy != "—" else escape(dropoff_copy)

    lessons_done = usage.get("lessons_done")
    lessons_total = usage.get("lessons_total")

    def safe_value(val):
        if val in (None, ""):
            return "—"
        return val

    lessons_summary = f"{safe_value(lessons_done)} / {safe_value(lessons_total)}"

    period_range = f"{escape(period.get('start', '—'))} → {escape(period.get('end', '—'))}"
    generated_on = escape(period.get("generated_on", "—"))
    prepared_for = report.get("prepared_for", "—")

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
        metric("Devices noted", fmt_metric(len(report.get("devices", {})), decimals=0)),
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
            lv = f"{ai.get('level_before', '—')} → {ai.get('level_after', '—')}"
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
                f"<span class='sen-metric__value'>{escape(str(answer or '—'))}</span></div>"
            )
        if not rows:
            return ""
        return f"<div class='sen-card'><h4>{escape(title)}</h4>{''.join(rows)}</div>"

    grid_cards = [
        f"<div class='sen-card'><h4>Student</h4>"
        f"{metric('Name', student.get('name', '—'), emphasize=True)}"
        f"{metric('ID', str(student.get('id', '—')))}"
        f"{metric('Class', student.get('class', '—'))}"
        f"{metric('Year', student.get('year', '—'))}"
        f"</div>",
        f"<div class='sen-card'><h4>Usage</h4>{usage_card}</div>",
        f"<div class='sen-card'><h4>Focus</h4>{focus_card}</div>",
        f"<div class='sen-card'><h4>Learning & Support</h4>{learning_card}{metric('Prepared for', prepared_for or '—')}</div>",
        f"<div class='sen-card'><h4>Routine</h4>{routine_card}</div>",
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

    audience_html = escape(prepared_for or "—")

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


def main():
    st.title("Student Report Generator (merged datasets)")

    # -------- Data sources --------
    uploaded = st.sidebar.file_uploader("Upload one or more JSON files", type=["json"], accept_multiple_files=True)
    if uploaded:
        datasets = [load_json(u) for u in uploaded]
        data = merge_data(*datasets)
        data_label = " + ".join([u.name for u in uploaded])
    else:
        try:
            ds1 = load_json(PRIMARY_JSON_PATH)
        except Exception:
            ds1 = {}
        try:
            ds2 = load_json(SECONDARY_JSON_PATH)
        except Exception:
            ds2 = {}
        if not ds1 and not ds2:
            st.error("No data loaded. Upload JSON files or set valid defaults.")
            return
        data = merge_data(ds1, ds2)
        data_label = f"{PRIMARY_JSON_PATH} + {SECONDARY_JSON_PATH}"
    st.sidebar.success(f"Loaded: {data_label}")

    # -------- Query / user selection --------
    query = st.text_input("Query", value="user_id 1 for teacher")
    user_id, audience = extract_user_id_and_audience(query)
    if user_id is None:
        st.warning("Enter a query like: `user_id 1` or `user_id 2 for teacher`")
        users_df = pd.DataFrame(data.get("user", []))
        if not users_df.empty:
            cols = [c for c in ["user_id", "name", "email", "class_level"] if c in users_df.columns]
            if cols:
                st.write("Users available:")
                st.table(users_df[cols])
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
    if st.button("Run"):
        try:
            agg = aggregate_student(data, user_id)
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt   = datetime.combine(end_date, datetime.max.time())

        curr = period_stats(data, user_id, start_dt, end_dt)
        prev_span_days = max(1, (end_dt.date() - start_dt.date()).days + 1)
        prev_end = start_dt - timedelta(seconds=1)
        prev_start = prev_end - timedelta(days=prev_span_days - 1)
        prev = period_stats(data, user_id, prev_start, prev_end)
        trend_vs_prev = compute_trend(curr["total_time_mins"], prev["total_time_mins"])

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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Accuracy & Mastery", "Processing Speed", "Engagement & Consistency",
            "Independence", "Communication", "AI Literacy"  # NEW tab
        ])

        with tab1:
            acc_block = accuracy_and_mastery(data, user_id, start_dt, end_dt)
            st.caption(f"Overall accuracy: **{acc_block['overall']:.1f}%**")
            st.plotly_chart(bar_mastery_subjects(acc_block["subjects"]), use_container_width=True)

        with tab2:
            spd_block = response_time_stats(data, user_id, start_dt, end_dt)
            st.caption(
                f"Attempts: **{spd_block['attempts']}** • Mean: **{spd_block['mean']}** • "
                f"Median: **{spd_block['median']}** • P90: **{spd_block['p90']}**"
            )
            st.plotly_chart(bar_response_time_subjects(spd_block["per_subject"], unit_label=" mins"), use_container_width=True)

        with tab3:
            eng = engagement_consistency(data, user_id, start_dt, end_dt)
            st.write("Active days:", eng["active_days"])
            st.write("Total sessions:", eng["sessions_total"])
            st.write("Longest streak:", eng["streak"])
            weeks = eng.get("weeks", [])
            st.caption(f"Weeks with activity: {sum(1 for col in eng.get('heat', []) if sum(col) > 0)} / {len(weeks)}")

        with tab4:
            indep = independence_support(data, user_id, start_dt, end_dt)
            st.write("Hint rate:", f"{indep['hint_rate']}%")
            st.write("Retry rate:", f"{indep['retry_rate']}%")
            st.write("Independence index:", f"{indep['independence']}")

        with tab5:
            comm = communication_social(data, user_id, start_dt, end_dt)
            st.write("Messages (period):", comm["messages"])
            st.write("Personalisation changes (proxy):", comm["personalisation_changes"])

        with tab6:
            ai_block = ai_literacy_stats(data, user_id, start_dt, end_dt)
            if ai_block.get("available"):
                st.write(f"Pre-test: {ai_block.get('pre_score','—')} / {int(ai_block.get('max_score') or 100)}")
                st.write(f"Post-test: {ai_block.get('post_score','—')} / {int(ai_block.get('max_score') or 100)}")
                st.write(f"Learning Gain: {ai_block.get('learning_gain','—')}%")
                st.write(f"Level (before → after): {ai_block.get('level_before','—')} → {ai_block.get('level_after','—')}")
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
        render_subject_growth(aggregate_student(data, user_id))

        # ---------- SEN Report ----------
        st.subheader("🧾 SEN Report (auto-generated)")
        if not curr["had_ts"]:
            st.warning("No reliable timestamps found in your selected range.")

        acc_block = accuracy_and_mastery(data, user_id, start_dt, end_dt)
        spd_block = response_time_stats(data, user_id, start_dt, end_dt)
        ai_block  = ai_literacy_stats(data, user_id, start_dt, end_dt)

        focus_score_now = compute_focus_score(curr["completion_pct"], curr["avg_session_mins"])
        focus_score_prev = compute_focus_score(prev["completion_pct"], prev["avg_session_mins"])
        focus_delta = focus_score_now - focus_score_prev
        dropoff_risk = (
            "high" if (curr.get("active_days", 0) <= 2 or curr["completion_pct"] < 30)
            else ("medium" if curr["completion_pct"] < 60 else "low")
        )

        report_data = {
            "student": {"name": agg["name"], "id": user_id, "class": agg.get("class_level", "—"), "year": agg.get("class_level", "—")},
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat(), "generated_on": date.today().isoformat()},
            "prepared_for": "Teacher" if audience == "teacher" else "Parent/Carer",
            "devices": {},
            "usage": {
                "active_days": curr.get("active_days", "—"),
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
            "learning": {"skills": [], "perseverance_index": agg.get("avg_hints_used", "—")},
            "language": {},
            "ai_support": {"hints_per_activity": agg.get("avg_hints_used", "—")},
            "routine": {"dropoff_risk": dropoff_risk},
            "goals": [],
            "recommendations": [],
            "questions": {},
            # report drivers
            "accuracy_mastery": acc_block,
            "processing_speed": spd_block,
            "ai_literacy": ai_block,   # NEW
        }

        render_sen_report(report_data)

        report_text = build_report(report_data)
        st.text_area("Report (copy-ready)", value=report_text, height=600)
if __name__ == "__main__":
    main()
