# ui_personalisation.py
import streamlit as st
import pandas as pd
from typing import Dict, Optional

from constants import AVATAR_KEYS, FONT_KEYS, BACKGROUND_KEYS
from utils_time import pick_first_ts, get_time_spent
from joins import build_indexes, topic_session_user_id, chapter_session_user_id, perf_user_id

import plotly.graph_objects as go


def _extract_attr(record: Dict, candidates):
    # Exact match first
    for k in candidates:
        if k in record and record[k] not in (None, ""):
            return str(record[k])
    # Fallback: any key that contains the token
    for key in record.keys():
        lk = key.lower()
        for c in candidates:
            if c.lower() in lk and record.get(key) not in (None, ""):
                return str(record[key])
    return None


def collect_personalisation_usage(
    data: Dict,
    user_id: int,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
) -> Dict[str, Dict[str, float]]:
    """Sum time_spent (minutes) by avatar/font/background within an optional period."""
    idx = build_indexes(data)
    usage = {"avatar": {}, "font": {}, "background": {}}

    def _within(r: Dict, keys):
        if not start_dt or not end_dt:
            return True
        ts = pick_first_ts(r, keys)
        return (ts is not None) and (start_dt <= ts <= end_dt)

    def _add(kind: str, label: Optional[str], minutes: float):
        if not label or minutes is None:
            return
        try:
            m = float(minutes)
        except Exception:
            return
        if m <= 0:
            return
        usage[kind][label] = usage[kind].get(label, 0.0) + m

    # topic_session
    for r in data.get("topic_session", []):
        if topic_session_user_id(r, idx) != user_id:
            continue
        if not _within(r, ["started_at", "completed_at", "created_at", "timestamp", "date"]):
            continue
        _add("avatar", _extract_attr(r, AVATAR_KEYS), get_time_spent(r))
        _add("font", _extract_attr(r, FONT_KEYS), get_time_spent(r))
        _add("background", _extract_attr(r, BACKGROUND_KEYS), get_time_spent(r))

    # chapter_session
    for r in data.get("chapter_session", []):
        if chapter_session_user_id(r, idx) != user_id:
            continue
        if not _within(r, ["started_at", "completed_at", "created_at", "timestamp", "date"]):
            continue
        _add("avatar", _extract_attr(r, AVATAR_KEYS), get_time_spent(r))
        _add("font", _extract_attr(r, FONT_KEYS), get_time_spent(r))
        _add("background", _extract_attr(r, BACKGROUND_KEYS), get_time_spent(r))

    # lesson_session
    for r in data.get("lesson_session", []):
        if r.get("user_id") != user_id:
            continue
        if not _within(r, ["created_at", "started_at", "completed_at", "timestamp", "date"]):
            continue
        _add("avatar", _extract_attr(r, AVATAR_KEYS), get_time_spent(r))
        _add("font", _extract_attr(r, FONT_KEYS), get_time_spent(r))
        _add("background", _extract_attr(r, BACKGROUND_KEYS), get_time_spent(r))

    # daily_activity_log
    for r in data.get("daily_activity_log", []):
        if r.get("user_id") != user_id:
            continue
        if not _within(r, ["login_timestamp", "created_at", "timestamp", "date"]):
            continue
        _add("avatar", _extract_attr(r, AVATAR_KEYS), get_time_spent(r))
        _add("font", _extract_attr(r, FONT_KEYS), get_time_spent(r))
        _add("background", _extract_attr(r, BACKGROUND_KEYS), get_time_spent(r))

    # activity_performance (optional time_spent)
    for r in data.get("activity_performance", []):
        if perf_user_id(r, idx) != user_id:
            continue
        if not _within(r, ["submitted_at"]):
            continue
        _add("avatar", _extract_attr(r, AVATAR_KEYS), get_time_spent(r))
        _add("font", _extract_attr(r, FONT_KEYS), get_time_spent(r))
        _add("background", _extract_attr(r, BACKGROUND_KEYS), get_time_spent(r))

    return usage


def _usage_df(usage: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    # Ensure the dataframe has stable columns even if empty
    rows = []
    for kind, d in usage.items():
        for name, mins in d.items():
            rows.append({"type": kind, "name": name, "minutes": round(float(mins), 1)})
    df = pd.DataFrame(rows, columns=["type", "name", "minutes"])
    if not df.empty:
        df = df.sort_values(["type", "minutes"], ascending=[True, False])
    return df


def _usage_bar(df: pd.DataFrame, kind: str, title_suffix: str) -> go.Figure:
    """Horizontal bar chart with Plotly for a given kind: avatar/font/background."""
    if df is None or df.empty or "type" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=180,
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"No {kind} usage — {title_suffix}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    sub = df[df["type"] == kind].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=180,
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"No {kind} usage — {title_suffix}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    sub = sub.nlargest(8, "minutes")
    names = list(sub["name"].astype(str))
    minutes = list(pd.to_numeric(sub["minutes"], errors="coerce").fillna(0.0))

    fig = go.Figure(
        data=[
            go.Bar(
                x=minutes,
                y=names,
                orientation="h",
                text=[f"{m:.1f} mins" for m in minutes],
                textposition="outside",
                cliponaxis=False,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{kind.title()} usage — {title_suffix}",
        xaxis=dict(title="Minutes", rangemode="tozero", showgrid=True),
        yaxis=dict(title="", automargin=True),
        showlegend=False,
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    return fig


def render_personalisation_usage(data, user_id, start_dt, end_dt):
    st.subheader("🎭 Personalisation usage (Avatar • Font • Background)")

    tabs = st.tabs(["This period", "All time"])

    # Period
    with tabs[0]:
        usage_p = collect_personalisation_usage(data, user_id, start_dt, end_dt)
        df_p = _usage_df(usage_p)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(_usage_bar(df_p, "avatar", "period"), use_container_width=True)
        with c2:
            st.plotly_chart(_usage_bar(df_p, "font", "period"), use_container_width=True)
        with c3:
            st.plotly_chart(_usage_bar(df_p, "background", "period"), use_container_width=True)

        if not df_p.empty:
            st.download_button(
                "Download period personalisation (.csv)",
                df_p.to_csv(index=False).encode("utf-8"),
                file_name=f"user_{user_id}_personalisation_period.csv",
                mime="text/csv",
            )
        else:
            st.info("No personalisation usage found in the selected period.")

    # All-time
    with tabs[1]:
        usage_a = collect_personalisation_usage(data, user_id, None, None)
        df_a = _usage_df(usage_a)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(_usage_bar(df_a, "avatar", "all-time"), use_container_width=True)
        with c2:
            st.plotly_chart(_usage_bar(df_a, "font", "all-time"), use_container_width=True)
        with c3:
            st.plotly_chart(_usage_bar(df_a, "background", "all-time"), use_container_width=True)

        if not df_a.empty:
            st.download_button(
                "Download all-time personalisation (.csv)",
                df_a.to_csv(index=False).encode("utf-8"),
                file_name=f"user_{user_id}_personalisation_alltime.csv",
                mime="text/csv",
            )
        else:
            st.info("No personalisation usage found in the dataset.")
