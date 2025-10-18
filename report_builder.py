# report_builder.py
from datetime import datetime
from typing import Dict, Any, List

def _fmt(v, suffix=""):
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        # keep one decimal for nicer readability
        return f"{v:.1f}{suffix}"
    return f"{v}{suffix}"

def _line(txt: str) -> str:
    return txt.rstrip() + "\n"

def _section(title: str) -> str:
    return f"\n{title}\n" + "-" * max(6, len(title)) + "\n"

def _bullet(items: List[str]) -> str:
    return "".join([f"- {x}\n" for x in items])

def build_report(d: Dict[str, Any]) -> str:
    # Core references
    student = d.get("student", {})
    period = d.get("period", {})
    usage = d.get("usage", {})
    focus = d.get("focus", {})
    learning = d.get("learning", {})
    routine = d.get("routine", {})
    acc = d.get("accuracy_mastery", {})
    spd = d.get("processing_speed", {})
    ai = d.get("ai_literacy", {})  # NEW

    out = []

    # Header
    out.append(_line("Student Learning Report (SEN)"))
    out.append(_line(""))
    out.append(_line(f"Student: {student.get('name','—')}"))
    out.append(_line(f"Student ID: {student.get('id','—')}"))
    out.append(_line(f"Class / Year: {student.get('class','—')} / {student.get('year','—')}"))
    out.append(_line(f"Reporting Period: {period.get('start','—')} – {period.get('end','—')}"))
    out.append(_line(f"Prepared for: {d.get('prepared_for','—')}"))
    out.append(_line(f"Generated on: {period.get('generated_on') or datetime.today().date().isoformat()}"))
    out.append(_line("Data Sources: activity_performance, chapter_session, topic_session, lesson_session, daily_activity_log, topics, enrollment"))
    out.append(_line(""))

    # 1) Executive Summary
    out.append(_section("1) Executive Summary"))
    out.append(_line("Summary: Overall progress is steady with moderate gains; continue current routine and supports."))
    out.append(_line(f"Focus score: {focus.get('focus_score', '—')} (↑ {focus.get('focus_score_delta','—')} from last period)"))
    out.append(_line(f"Completion rate: {usage.get('lessons_done',0)}/{usage.get('lessons_total',0)} ({_fmt(usage.get('completion_pct'), '%')})"))
    out.append(_line(f"Time-on-task: {_fmt(usage.get('total_time_mins'),' mins')} total ({_fmt(usage.get('trend_vs_prev_pct'), '%')} vs last period)"))

    # 2) SEN Profile & Accommodations
    out.append(_section("2) SEN Profile & Accommodations"))
    out.append(_line("Summary: Derived metrics only (no pre-set accommodations in source)."))
    out.append(_line("Primary Needs: —"))
    out.append(_line("Accommodations: —"))
    out.append(_line("Effectiveness: TTS ON → 0% vs OFF → 0% (0pp)"))
    out.append(_line("Stability: Font size changed 0× this period"))

    # 3) Engagement & Usage
    out.append(_section("3) Engagement & Usage"))
    out.append(_line("Summary: Moderate engagement; room for higher completion."))
    out.append(_line(f"Active Days: {usage.get('active_days','—')}"))
    out.append(_line(f"Sessions: {usage.get('sessions','—')} (avg. {_fmt(usage.get('avg_session_mins'),' mins')})"))
    out.append(_line(f"Completion: {usage.get('lessons_done',0)} of {usage.get('lessons_total',0)} ({_fmt(usage.get('completion_pct'),'%')})"))
    out.append(_line(f"Trend: {_fmt(usage.get('trend_vs_prev_pct'),'%')} vs last period"))

    # 4) Focus & Concentration
    out.append(_section("4) Focus & Concentration"))
    out.append(_line("Summary: Improved attention relative to class median."))
    out.append(_line(f"Focus score: {focus.get('focus_score','—')} (class median: {focus.get('class_median','—')})"))
    out.append(_line(f"Avg. attention block: {_fmt(focus.get('avg_sustained_block_mins'),' mins')}"))

    # 5) Learning Progress & Mastery
    out.append(_section("5) Learning Progress & Mastery"))
    out.append(_line("Summary: Subject-level growth based on activity performance."))
    # Print short per-subject breakdown if present
    subs_meta = acc.get("subjects_meta") or []
    if subs_meta:
        for row in subs_meta:
            out.append(_line(f"- {row['subject']}: {row['avg']:.1f}% ({row['band']}, {row['attempts']} attempts)"))
    else:
        out.append(_line("No subject-level performance available in the selected period."))
    out.append(_line(f"Perseverance index: {learning.get('perseverance_index','—')} (fraction of attempts using hints)"))

    # 6) Processing Speed
    out.append(_section("6) Processing Speed"))
    out.append(_line("Summary: Average response/processing time per subject based on total time spent on attempts."))
    if spd and spd.get("attempts",0) > 0:
        out.append(_line(f"Overall mean: {_fmt(spd.get('mean'),' mins')} • median: {_fmt(spd.get('median'),' mins')} • p90: {_fmt(spd.get('p90'),' mins')} (n={spd.get('attempts')})"))
        per_sub = spd.get("per_subject_meta") or []
        for r in per_sub:
            out.append(_line(f"- {r['subject']}: {_fmt(r['avg_time'],' mins')} (total {_fmt(r['total_time'],' mins')}, {r['attempts']} attempts)"))
    else:
        out.append(_line("No processing speed data in the selected period."))

    # 7) AI Literacy & Learning Gain (NEW)
    out.append(_section("7) AI Literacy & Learning Gain"))
    if ai and ai.get("available"):
        pre_s = ai.get("pre_score")
        post_s = ai.get("post_score")
        mx = ai.get("max_score") or 100.0
        lg = ai.get("learning_gain")
        out.append(_line(f"Pre-test: {_fmt(pre_s)} / {mx:.0f}  |  Post-test: {_fmt(post_s)} / {mx:.0f}"))
        out.append(_line(f"Learning Gain: {_fmt(lg, '%')}"))
        out.append(_line(f"Level (before → after): {ai.get('level_before','—')} → {ai.get('level_after','—')}"))
        concepts = ai.get("concepts_mastered") or []
        apps = ai.get("applications") or []
        if concepts:
            out.append(_line("Key Concepts Mastered: " + ", ".join(concepts)))
        if apps:
            out.append(_line("Skill Applications: " + "; ".join(apps)))
    else:
        out.append(_line("Not available in this dataset (add local 'ai_literacy_assessment' for pre/post tracking)."))

    # 8) Reading, Language & Expression
    out.append(_section("8) Reading, Language & Expression"))
    out.append(_line("Summary: Not available in this dataset."))
    out.append(_line("Readability: —"))
    out.append(_line("TTR: —"))

    # 9) AI Interaction Quality & Support Usage
    out.append(_section("9) AI Interaction Quality & Support Usage"))
    out.append(_line("Summary: Derived hints usage (no built-in AI support fields in source)."))
    out.append(_line(f"Hints used per attempt: {learning.get('perseverance_index','—')}"))

    # 10) Motivation & Routine
    out.append(_section("10) Motivation & Routine"))
    out.append(_line(f"Summary: Potential drop-off risk."))
    out.append(_line(f"Drop-off risk: {routine.get('dropoff_risk','—')}"))

    # 11) Technology & Accessibility Diagnostics
    out.append(_section("11) Technology & Accessibility Diagnostics"))
    out.append(_line("Summary: Device info is partial in this dataset."))

    # 12) Goals & Recommendations
    recs = d.get("recommendations") or [
        "Encourage regular short practice sessions (5–7 mins) on weaker subjects",
        "Review missed questions in recent attempts",
        "Use shorter sessions if average session length is below 10 mins",
    ]
    out.append(_section("12) Goals & Recommendations"))
    out.append(_bullet(recs))

    # 13) Unanswered & Out-of-Scope Questions
    out.append(_section("13) Unanswered & Out-of-Scope Questions"))
    out.append(_line("Summary: Not tracked in this dataset."))
    out.append(_line("Total questions: —"))
    out.append(_line("Unanswered: — | Out-of-scope: —"))

    return "".join(out)
