# ui_profile.py
import streamlit as st


def _metric_chip(label: str, value: str, *, icon: str = "") -> str:
    icon_part = f"<span class='chip-icon'>{icon}</span>" if icon else ""
    return (
        f"<div class='metric-chip'>"
        f"{icon_part}"
        f"<div class='chip-body'>"
        f"<span class='chip-label'>{label}</span>"
        f"<span class='chip-value'>{value}</span>"
        f"</div></div>"
    )


def _detail_row(label: str, value: str) -> str:
    return (
        "<div class='detail-row'>"
        f"<span class='detail-label'>{label}</span>"
        f"<span class='detail-value'>{value}</span>"
        "</div>"
    )


PROFILE_STYLE = """
<style>
    .profile-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 45%, #0f172a 100%);
        border-radius: 18px;
        padding: 28px 32px;
        display: flex;
        gap: 28px;
        margin-bottom: 16px;
        box-shadow: 0 24px 40px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .profile-card img {
        border-radius: 18px;
        box-shadow: 0 18px 36px rgba(59, 130, 246, 0.4);
        width: 140px;
        height: 140px;
        object-fit: cover;
    }
    .profile-body {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 16px;
        color: #f8fafc;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .profile-heading {
        font-size: 1.6rem;
        font-weight: 700;
        display: flex;
        align-items: baseline;
        gap: 14px;
    }
    .profile-heading span {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(226, 232, 240, 0.85);
    }
    .profile-meta {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 12px 18px;
        margin-top: 8px;
    }
    .detail-row {
        display: flex;
        flex-direction: column;
        background: rgba(15, 23, 42, 0.68);
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: inset 0 1px 1px rgba(226, 232, 240, 0.15);
        min-height: 92px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .detail-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(148, 163, 184, 0.82);
        margin-bottom: 4px;
    }
    .detail-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        word-break: break-word;
        white-space: normal;
    }
    .metric-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 12px;
    }
    .metric-chip {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(17, 24, 39, 0.7));
        border-radius: 18px;
        padding: 16px 20px;
        border: 1px solid rgba(59, 130, 246, 0.25);
        box-shadow: 0 20px 35px rgba(15, 23, 42, 0.22);
        min-width: 210px;
    }
    .chip-icon {
        font-size: 1.6rem;
        filter: drop-shadow(0 6px 12px rgba(94, 234, 212, 0.35));
    }
    .chip-body {
        display: flex;
        flex-direction: column;
        line-height: 1.1;
    }
    .chip-label {
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(191, 219, 254, 0.9);
    }
    .chip-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        text-shadow: 0 8px 18px rgba(37, 99, 235, 0.35);
    }
</style>
"""


def display_user_metadata(agg):
    """Render a polished profile card with key learner metadata."""
    st.markdown(PROFILE_STYLE, unsafe_allow_html=True)

    with st.expander("User Profile & Metadata", expanded=True):
        avatar_url = agg.get("avatar")
        if not avatar_url:
            avatar_url = "https://ui-avatars.com/api/?background=2563EB&color=fff&name=" + agg["name"].replace(" ", "+")

        profile_header = (
            "<div class='profile-card'>"
            f"<img src='{avatar_url}' alt='avatar' />"
            "<div class='profile-body'>"
            f"<div class='profile-heading'>{agg['name']}<span>{agg['class_level']} • {agg['school_name']}</span></div>"
            "<div class='profile-meta'>"
            f"{_detail_row('Age', agg['age_display'])}"
            f"{_detail_row('Gender', agg['gender'])}"
            f"{_detail_row('Reading Level', agg['reading_level'])}"
            f"{_detail_row('Email', agg['email'])}"
            f"{_detail_row('Parental Email', agg['parental_email'])}"
            f"{_detail_row('Account Created', agg['created_at'])}"
            f"{_detail_row('Last Updated', agg['updated_at'])}"
            "</div>"
            "</div>"
            "</div>"
        )
        st.markdown(profile_header, unsafe_allow_html=True)

        total_points = agg.get("total_points", 0.0)
        avg_session = agg.get("avg_session_length", 0.0)
        lessons_completed = agg.get("lessons_completed", "—")
        completion_rate = agg.get("lesson_completion_rate", "—")

        if isinstance(total_points, (int, float)):
            total_points_display = f"{total_points:,.0f}"
        else:
            total_points_display = str(total_points)

        if isinstance(avg_session, (int, float)):
            avg_session_display = f"{avg_session:.1f} mins"
        elif avg_session in ("—", None, ""):
            avg_session_display = "—"
        else:
            avg_session_display = f"{avg_session} mins"

        if lessons_completed in ("—", None, ""):
            lessons_completed_display = "—"
        else:
            lessons_completed_display = str(lessons_completed)

        if isinstance(completion_rate, (int, float)):
            completion_rate_display = f"{completion_rate:.1f}%"
        elif completion_rate in ("—", None, ""):
            completion_rate_display = "—"
        else:
            completion_rate_display = str(completion_rate)

        metrics = [
            _metric_chip("All-time points", total_points_display, icon="🏆"),
            _metric_chip("Avg session length", avg_session_display, icon="⏱️"),
            _metric_chip("Lessons completed", lessons_completed_display, icon="📘"),
            _metric_chip("Lesson completion rate", completion_rate_display, icon="🎯"),
        ]
        st.markdown("<div class='metric-grid'>" + "".join(metrics) + "</div>", unsafe_allow_html=True)
