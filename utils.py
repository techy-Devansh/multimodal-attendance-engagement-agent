"""
SkillWallet Utilities Module
Provides helper functions for time formatting, CSV exports, synthetic demo data generation,
and professional PDF report compilation using ReportLab.
"""

import io
import os
import math
import uuid
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

import database


def sec_to_mmss(seconds: float) -> str:
    """Formats floating-point seconds into MM:SS string."""
    if seconds is None or math.isnan(seconds) or seconds < 0:
        return "00:00"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def clean_temp_file(file_path: Optional[str]) -> None:
    """Safely removes a temporary file without throwing on file lock errors."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass


def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame into UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def generate_demo_dataset(
    session_id: Optional[str] = None,
    session_name: str = "Demo Lecture — Data Structures & Algorithms",
    duration_sec: float = 180.0,
    presence_threshold_sec: float = 5.0,
    db_path: str = database.DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Generates a realistic multimodal classroom dataset:
    - 12 Anonymous students (Present, Late, and Fleeting cases)
    - 36 Synchronized intervals with high engagement, drop periods, and Q&A peaks
    - Rule-based AI teaching recommendations
    Persists data directly into SQLite for instant exploration.
    """
    if not session_id:
        session_id = f"DEMO-{uuid.uuid4().hex[:6].upper()}"

    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Attendance Records (12 tracked students)
    # 10 consistent attendees, 1 late entry, 1 fleeting passerby
    attendance_data = [
        {"track_id": "STU-01", "first_seen_sec": 1.2, "last_seen_sec": 178.5, "total_visible_sec": 165.4, "sustained_present": True, "status": "Present", "avg_confidence": 0.96},
        {"track_id": "STU-02", "first_seen_sec": 0.8, "last_seen_sec": 179.1, "total_visible_sec": 170.2, "sustained_present": True, "status": "Present", "avg_confidence": 0.94},
        {"track_id": "STU-03", "first_seen_sec": 2.5, "last_seen_sec": 176.0, "total_visible_sec": 158.0, "sustained_present": True, "status": "Present", "avg_confidence": 0.91},
        {"track_id": "STU-04", "first_seen_sec": 1.0, "last_seen_sec": 180.0, "total_visible_sec": 172.5, "sustained_present": True, "status": "Present", "avg_confidence": 0.95},
        {"track_id": "STU-05", "first_seen_sec": 3.0, "last_seen_sec": 177.2, "total_visible_sec": 160.8, "sustained_present": True, "status": "Present", "avg_confidence": 0.92},
        {"track_id": "STU-06", "first_seen_sec": 1.5, "last_seen_sec": 179.0, "total_visible_sec": 168.4, "sustained_present": True, "status": "Present", "avg_confidence": 0.93},
        {"track_id": "STU-07", "first_seen_sec": 4.0, "last_seen_sec": 175.5, "total_visible_sec": 154.2, "sustained_present": True, "status": "Present", "avg_confidence": 0.90},
        {"track_id": "STU-08", "first_seen_sec": 2.0, "last_seen_sec": 178.0, "total_visible_sec": 162.7, "sustained_present": True, "status": "Present", "avg_confidence": 0.92},
        {"track_id": "STU-09", "first_seen_sec": 5.5, "last_seen_sec": 176.8, "total_visible_sec": 150.0, "sustained_present": True, "status": "Present", "avg_confidence": 0.89},
        {"track_id": "STU-10", "first_seen_sec": 2.2, "last_seen_sec": 179.5, "total_visible_sec": 166.1, "sustained_present": True, "status": "Present", "avg_confidence": 0.94},
        # Late arrival
        {"track_id": "STU-11", "first_seen_sec": 68.0, "last_seen_sec": 178.0, "total_visible_sec": 105.0, "sustained_present": True, "status": "Present", "avg_confidence": 0.88},
        # Fleeting passerby / doorway occlusion
        {"track_id": "STU-12", "first_seen_sec": 12.0, "last_seen_sec": 15.5, "total_visible_sec": 3.5, "sustained_present": False, "status": "Partial / Fleeting", "avg_confidence": 0.76}
    ]

    # 2. Synchronized Interval Series (36 intervals @ 5s each = 180s)
    interval_sec = 5.0
    num_intervals = int(duration_sec // interval_sec)
    intervals_data = []

    for i in range(num_intervals):
        t_start = i * interval_sec
        t_end = (i + 1) * interval_sec

        # Phase 1: 0 - 45s (Warm-up & concept setup) -> Fused ~72
        if t_start < 45.0:
            v_score = 74.0 + 4.0 * np.sin(i * 0.4)
            a_score = 70.0 + 5.0 * np.cos(i * 0.5)
            speech = 60.0
            silence = 40.0
            orient = 82.0
            motion = 12.0
            faces = 10
            event = "NORMAL"

        # Phase 2: 45 - 75s (Interactive Q&A peak) -> Fused ~84
        elif t_start < 75.0:
            v_score = 86.0 + 3.0 * np.sin(i * 0.6)
            a_score = 85.0 + 4.0 * np.cos(i * 0.7)
            speech = 68.0
            silence = 32.0
            orient = 92.0
            motion = 16.0
            faces = 11
            event = "INTERACTION_PEAK"

        # Phase 3: 75 - 115s (Dense theoretical slide / attention drop) -> Fused ~36
        elif t_start < 115.0:
            v_score = 38.0 + 3.0 * np.sin(i * 0.3)
            a_score = 34.0 + 4.0 * np.cos(i * 0.4)
            speech = 12.0
            silence = 88.0
            orient = 38.0
            motion = 2.5
            faces = 11
            event = "ENGAGEMENT_DROP"

        # Phase 4: 115 - 150s (Teacher recap & live problem solving) -> Fused ~74
        elif t_start < 150.0:
            v_score = 75.0 + 3.0 * np.sin(i * 0.5)
            a_score = 72.0 + 4.0 * np.cos(i * 0.6)
            speech = 58.0
            silence = 42.0
            orient = 84.0
            motion = 11.0
            faces = 11
            event = "NORMAL"

        # Phase 5: 150 - 180s (Summary & Q&A wrap-up) -> Fused ~78
        else:
            v_score = 80.0 + 3.0 * np.sin(i * 0.4)
            a_score = 76.0 + 3.0 * np.cos(i * 0.5)
            speech = 64.0
            silence = 36.0
            orient = 88.0
            motion = 14.0
            faces = 11
            event = "NORMAL"

        fused = round(0.60 * v_score + 0.40 * a_score, 1)

        intervals_data.append({
            "interval_idx": i,
            "start_sec": t_start,
            "end_sec": t_end,
            "video_score": round(v_score, 1),
            "audio_score": round(a_score, 1),
            "fused_score": fused,
            "face_count": faces,
            "head_orientation_score": round(orient, 1),
            "motion_index": round(motion, 1),
            "speech_ratio": round(speech, 1),
            "silence_ratio": round(silence, 1),
            "event_tag": event
        })

    # 3. AI Recommendations
    recommendations_data = [
        {
            "rule_id": "R-RECAP-01",
            "title": "Concept Check & Recap Needed",
            "category": "Attention Recovery",
            "severity": "HIGH",
            "time_window": "01:15 – 01:55",
            "trigger_metric": "Fused engagement dropped to 34.2/100 (avg 36.1/100) for 40s during theoretical slide progression",
            "rationale": "Sustained downward head orientation (38%) and high silence ratio (88%) indicated comprehension bottleneck and loss of student focus.",
            "recommendation": "Pause lecture progression immediately. Conduct a 90-second rapid recap of the core recurrence relation before moving to proofs.",
            "suggested_action": "Teacher script: 'Let's take 60 seconds to review the base case. Who can summarize why the tree depth is log N?'"
        },
        {
            "rule_id": "R-POSITIVE-05",
            "title": "Instructional High-Point Reinforcement",
            "category": "Positive Reinforcement",
            "severity": "INFO",
            "time_window": "00:45 – 01:15",
            "trigger_metric": "Peak multimodal engagement reached 86.8/100 with active speech (68%) and high forward gaze (92%)",
            "rationale": "Co-occurrence of conversational turn-taking and direct board alignment marked peak participatory learning.",
            "recommendation": "Preserve the interactive questioning prompt used in this interval as an instructional pattern for future sessions.",
            "suggested_action": "Pedagogical insight: Replicate student-led code tracing at the beginning of complex algorithmic modules."
        },
        {
            "rule_id": "R-ATTENDANCE-06",
            "title": "Sustained Presence Attendance Discrepancy",
            "category": "Attendance & Integrity",
            "severity": "INFO",
            "time_window": "Entire Session",
            "trigger_metric": "1 of 12 tracked attendees (STU-12) logged visible duration of 3.5s (below 5.0s sustained threshold)",
            "rationale": "Track STU-12 was detected briefly near the room perimeter, indicative of a fleeting passerby or camera blind spot.",
            "recommendation": "Cross-reference manual roll call for student at perimeter seat 12 to ensure accurate status.",
            "suggested_action": "Action: Flag STU-12 as Partial / Fleeting in final department ledger."
        }
    ]

    session_header = {
        "session_id": session_id,
        "session_name": session_name,
        "filename": "classroom_sample_lecture_dsa.mp4",
        "created_at": created_at,
        "video_duration_sec": duration_sec,
        "fps": 30.0,
        "total_frames": int(duration_sec * 30),
        "has_audio": 1,
        "audio_duration_sec": duration_sec,
        "presence_threshold_sec": presence_threshold_sec,
        "video_weight": 0.60,
        "audio_weight": 0.40,
        "interval_sec": interval_sec,
        "is_demo": 1,
        "processing_time_sec": 1.45
    }

    # Persist in SQLite
    database.init_db(db_path)
    database.save_session(session_header, db_path)
    database.save_attendance_batch(session_id, attendance_data, db_path)
    database.save_intervals_batch(session_id, intervals_data, db_path)
    database.save_recommendations_batch(session_id, recommendations_data, db_path)

    return {
        "session": session_header,
        "attendance": attendance_data,
        "intervals": intervals_data,
        "recommendations": recommendations_data
    }


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that adds page numbers and running footer to ReportLab PDFs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Footer
        footer_text = f"SkillWallet — Multimodal Attendance & Engagement Agent | Page {self._pageNumber} of {page_count}"
        self.drawString(36, 25, footer_text)
        disclaimer = "CONFIDENTIAL & PRIVACY-PRESERVING: Non-biometric anonymous educational analytics."
        self.drawRightString(612 - 36, 25, disclaimer)

        # Thin rule above footer
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 35, 612 - 36, 35)

        self.restoreState()


def generate_pdf_report(
    session_meta: Dict[str, Any],
    attendance_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    recommendations: List[Dict[str, Any]]
) -> bytes:
    """
    Builds a professional, comprehensive PDF report summarizing:
    - Session Metadata & Edge Processing Integrity
    - Privacy-Preserving Attendance Register with Sustained Presence Verification
    - Multimodal Engagement Metrics & Trend Analysis
    - Rule-based Pedagogical Teaching Recommendations
    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a")
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0284c7")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    cell_header = ParagraphStyle(
        "TableCellHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#d97706")
    )

    story = []

    # 1. Header Banner
    is_demo = bool(session_meta.get("is_demo", 0))
    demo_badge_str = " [DEMO DATASET]" if is_demo else ""

    story.append(Paragraph(f"SkillWallet — Multimodal Capstone Report{demo_badge_str}", title_style))
    story.append(Paragraph("Automated Privacy-Preserving Attendance & Multimodal Engagement Analytics", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # 2. Session Metadata Grid
    meta_data = [
        [
            Paragraph("<b>Session ID:</b>", cell_style),
            Paragraph(str(session_meta.get("session_id", "N/A")), cell_style),
            Paragraph("<b>Date / Time:</b>", cell_style),
            Paragraph(str(session_meta.get("created_at", "N/A")), cell_style)
        ],
        [
            Paragraph("<b>Session Name:</b>", cell_style),
            Paragraph(str(session_meta.get("session_name", "Classroom Session")), cell_style),
            Paragraph("<b>Video Duration:</b>", cell_style),
            Paragraph(f"{sec_to_mmss(session_meta.get('video_duration_sec', 0))} ({session_meta.get('video_duration_sec', 0)}s)", cell_style)
        ],
        [
            Paragraph("<b>Presence Threshold:</b>", cell_style),
            Paragraph(f"{session_meta.get('presence_threshold_sec', 5.0)}s sustained", cell_style),
            Paragraph("<b>Fusion Weights:</b>", cell_style),
            Paragraph(f"Video {int(session_meta.get('video_weight', 0.6)*100)}% / Audio {int(session_meta.get('audio_weight', 0.4)*100)}%", cell_style)
        ],
        [
            Paragraph("<b>FPS & Frames:</b>", cell_style),
            Paragraph(f"{session_meta.get('fps', 0)} FPS ({session_meta.get('total_frames', 0)} frames)", cell_style),
            Paragraph("<b>Processing Time:</b>", cell_style),
            Paragraph(f"{session_meta.get('processing_time_sec', 0)}s (Local Edge CPU)", cell_style)
        ]
    ]

    t_meta = Table(meta_data, colWidths=[110, 160, 110, 160])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Privacy & Integrity Notice
    privacy_notice = (
        "<b>Privacy & Ethical AI Assurance:</b> This system operates on strictly anonymous sequential tracking IDs "
        "(e.g., STU-01). No facial recognition, biometric identity enrollment, or name identification is performed. "
        "Multimodal engagement indices reflect observed behavioral proxies (gaze orientation, head stability, vocal activity) "
        "and do NOT constitute validated measures of cognitive attention or student comprehension."
    )
    story.append(Paragraph(privacy_notice, body_style))
    story.append(Spacer(1, 12))

    # 3. Attendance Section
    story.append(Paragraph("1. Attendance Verification & Sustained Presence Analysis", section_heading))

    if not attendance_df.empty:
        total_tracked = len(attendance_df)
        present_count = len(attendance_df[attendance_df["sustained_present"] == 1])
        fleeting_count = len(attendance_df[attendance_df["sustained_present"] == 0])
        att_rate = round((present_count / float(total_tracked)) * 100.0, 1) if total_tracked > 0 else 0.0

        summary_p = (
            f"<b>Total Detected Tracks:</b> {total_tracked} | "
            f"<b>Sustained Present (≥ {session_meta.get('presence_threshold_sec', 5.0)}s):</b> {present_count} ({att_rate}%) | "
            f"<b>Partial / Fleeting:</b> {fleeting_count}"
        )
        story.append(Paragraph(summary_p, body_style))
        story.append(Spacer(1, 6))

        # Attendance Table
        att_headers = ["Anonymous ID", "First Seen", "Last Seen", "Visible Duration", "Sustained (≥ Thr)", "Status"]
        table_rows = [[Paragraph(h, cell_header) for h in att_headers]]

        for _, row in attendance_df.iterrows():
            sustained_str = "YES" if row.get("sustained_present") in (1, True) else "NO (< Thr)"
            status_str = str(row.get("status", "Present"))
            table_rows.append([
                Paragraph(str(row.get("track_id")), cell_style),
                Paragraph(sec_to_mmss(row.get("first_seen_sec")), cell_style),
                Paragraph(sec_to_mmss(row.get("last_seen_sec")), cell_style),
                Paragraph(f"{row.get('total_visible_sec', 0):.1f}s", cell_style),
                Paragraph(sustained_str, cell_style),
                Paragraph(status_str, cell_style),
            ])

        t_att = Table(table_rows, colWidths=[90, 80, 80, 100, 100, 90])
        t_att.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        story.append(t_att)
    else:
        story.append(Paragraph("<i>No attendance records detected for this session.</i>", body_style))

    story.append(Spacer(1, 14))

    # 4. Multimodal Engagement Analytics Summary
    story.append(Paragraph("2. Multimodal Engagement & Interaction Summary", section_heading))

    if not intervals_df.empty:
        avg_fused = round(intervals_df["fused_score"].mean(), 1)
        min_fused = round(intervals_df["fused_score"].min(), 1)
        max_fused = round(intervals_df["fused_score"].max(), 1)
        drops_count = len(intervals_df[intervals_df["event_tag"] == "ENGAGEMENT_DROP"])
        peaks_count = len(intervals_df[intervals_df["event_tag"] == "INTERACTION_PEAK"])
        avg_speech = round(intervals_df["speech_ratio"].mean(), 1) if "speech_ratio" in intervals_df else 0.0

        eng_summary = [
            [
                Paragraph("<b>Mean Fused Score:</b>", cell_style),
                Paragraph(f"{avg_fused} / 100", cell_style),
                Paragraph("<b>Score Range:</b>", cell_style),
                Paragraph(f"{min_fused} – {max_fused} / 100", cell_style)
            ],
            [
                Paragraph("<b>Engagement Drops Flagged:</b>", cell_style),
                Paragraph(f"{drops_count} intervals (< 40)", cell_style),
                Paragraph("<b>Interaction Peaks Flagged:</b>", cell_style),
                Paragraph(f"{peaks_count} intervals (≥ 78)", cell_style)
            ],
            [
                Paragraph("<b>Average Speech Presence:</b>", cell_style),
                Paragraph(f"{avg_speech}% active discourse", cell_style),
                Paragraph("<b>Sampling Frequency:</b>", cell_style),
                Paragraph(f"{session_meta.get('interval_sec', 5.0)}s interval window", cell_style)
            ]
        ]
        t_eng = Table(eng_summary, colWidths=[130, 140, 130, 140])
        t_eng.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_eng)
    else:
        story.append(Paragraph("<i>No interval analytics available.</i>", body_style))

    story.append(Spacer(1, 14))

    # 5. Rule-Based AI Teaching Recommendations
    story.append(Paragraph("3. Transparent Pedagogical AI Recommendations", section_heading))
    intro_rec = (
        "Generated deterministically by the rule-based reasoning engine. Every recommendation is paired with "
        "the empirical metric trigger, psychological/pedagogical rationale, and concrete teacher scripting."
    )
    story.append(Paragraph(intro_rec, body_style))
    story.append(Spacer(1, 6))

    if recommendations:
        for r in recommendations:
            sev = r.get("severity", "INFO")
            sev_color = "#dc2626" if sev == "HIGH" else "#d97706" if sev == "MEDIUM" else "#0284c7"

            rec_box = [
                [
                    Paragraph(f"<b>[{sev}] {r.get('rule_id')}: {r.get('title')}</b>", ParagraphStyle(
                        "RecTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor(sev_color)
                    )),
                    Paragraph(f"<b>Window:</b> {r.get('time_window', 'N/A')}", ParagraphStyle(
                        "RecWin", parent=styles["Normal"], fontName="Helvetica", fontSize=8, alignment=2, textColor=colors.HexColor("#64748b")
                    ))
                ],
                [
                    Paragraph(f"<b>Trigger Metric:</b> {r.get('trigger_metric')}", cell_style),
                    Paragraph("", cell_style)
                ],
                [
                    Paragraph(f"<b>Pedagogical Rationale:</b> {r.get('rationale')}", cell_style),
                    Paragraph("", cell_style)
                ],
                [
                    Paragraph(f"<b>Recommended Action:</b> {r.get('recommendation')}", cell_style),
                    Paragraph("", cell_style)
                ],
                [
                    Paragraph(f"<b>Suggested Teacher Script:</b> <i>\"{r.get('suggested_action')}\"</i>", ParagraphStyle(
                        "Script", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8, textColor=colors.HexColor("#1e293b")
                    )),
                    Paragraph("", cell_style)
                ]
            ]
            t_box = Table(rec_box, colWidths=[400, 140])
            t_box.setStyle(TableStyle([
                ('SPAN', (0, 1), (1, 1)),
                ('SPAN', (0, 2), (1, 2)),
                ('SPAN', (0, 3), (1, 3)),
                ('SPAN', (0, 4), (1, 4)),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor(sev_color)),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(KeepTogether([t_box, Spacer(1, 8)]))
    else:
        story.append(Paragraph("<i>No pedagogical anomalies detected. Engagement metrics remained within balanced baseline.</i>", body_style))

    # Build PDF with custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
