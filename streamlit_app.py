# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

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
from ui_event_log import render_event_log_table
from report_builder import build_report

st.set_page_config(page_title="Student Report Generator", layout="wide")


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

        report_text = build_report(report_data)
        st.text_area("Report (copy-ready)", value=report_text, height=600)

        # ---------- Events table ----------
        st.markdown("### Event log")
        render_event_log_table(data, user_id)


if __name__ == "__main__":
    main()
