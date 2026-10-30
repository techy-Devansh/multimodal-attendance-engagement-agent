"""
SkillWallet Multimodal Fusion Module
Combines video engagement indicators and audio speech/silence metrics
into an interval-based multimodal score (0-100) with configurable weights.
Detects engagement drops, interaction peaks, and provides Plotly visualizations.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fuse_multimodal_intervals(
    video_intervals: List[Dict[str, Any]],
    audio_intervals: List[Dict[str, Any]],
    video_weight: float = 0.60,
    audio_weight: float = 0.40,
    has_audio: bool = True
) -> List[Dict[str, Any]]:
    """
    Fuses video interval metrics and audio interval metrics on matching time intervals.
    Automatically handles video-only mode when has_audio is False.
    """
    # Normalize weights
    if not has_audio or audio_weight <= 0.0:
        w_v = 1.0
        w_a = 0.0
    else:
        tot = video_weight + audio_weight
        w_v = video_weight / tot if tot > 0 else 0.60
        w_a = audio_weight / tot if tot > 0 else 0.40

    # Build audio dictionary lookup by interval_idx
    audio_lookup = {a["interval_idx"]: a for a in audio_intervals}

    fused_results: List[Dict[str, Any]] = []

    for v in video_intervals:
        idx = v["interval_idx"]
        v_score = float(v.get("video_score", 50.0))

        if idx in audio_lookup and has_audio:
            a_rec = audio_lookup[idx]
            a_score = float(a_rec.get("audio_score", 50.0))
            speech_ratio = float(a_rec.get("speech_ratio", 0.0))
            silence_ratio = float(a_rec.get("silence_ratio", 100.0))
        else:
            # Neutral audio fallback
            a_score = 50.0
            speech_ratio = 50.0 if not has_audio else 0.0
            silence_ratio = 50.0 if not has_audio else 100.0

        # Calculate fused score
        fused = round(w_v * v_score + w_a * a_score, 1)
        fused = max(0.0, min(100.0, fused))

        # Classify interval event tag
        face_cnt = v.get("face_count", 0)
        orient_score = v.get("head_orientation_score", 50.0)
        motion_idx = v.get("motion_index", 5.0)

        if fused < 40.0:
            event_tag = "ENGAGEMENT_DROP"
        elif fused >= 78.0 and speech_ratio >= 35.0:
            event_tag = "INTERACTION_PEAK"
        elif speech_ratio > 82.0 and motion_idx < 10.0:
            event_tag = "MONOLOGUE"
        elif motion_idx < 3.0 and speech_ratio < 10.0:
            event_tag = "LOW_ACTIVITY"
        else:
            event_tag = "NORMAL"

        fused_results.append({
            "interval_idx": idx,
            "start_sec": v["start_sec"],
            "end_sec": v["end_sec"],
            "video_score": v_score,
            "audio_score": a_score,
            "fused_score": fused,
            "face_count": face_cnt,
            "head_orientation_score": orient_score,
            "motion_index": motion_idx,
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
            "event_tag": event_tag
        })

    return fused_results


def recalculate_fusion_df(
    intervals_df: pd.DataFrame,
    video_weight: float,
    audio_weight: float,
    has_audio: bool = True
) -> pd.DataFrame:
    """
    Fast recalculation of fused scores and event tags directly on a DataFrame
    when user adjusts the UI weight sliders.
    """
    df = intervals_df.copy()
    if not has_audio or audio_weight <= 0.0:
        w_v = 1.0
        w_a = 0.0
    else:
        tot = video_weight + audio_weight
        w_v = video_weight / tot if tot > 0 else 0.60
        w_a = audio_weight / tot if tot > 0 else 0.40

    df["fused_score"] = (w_v * df["video_score"] + w_a * df["audio_score"]).round(1).clip(0.0, 100.0)

    # Re-evaluate event tags
    def assign_tag(row):
        score = row["fused_score"]
        speech = row.get("speech_ratio", 0.0)
        motion = row.get("motion_index", 5.0)
        if score < 40.0:
            return "ENGAGEMENT_DROP"
        elif score >= 78.0 and speech >= 35.0:
            return "INTERACTION_PEAK"
        elif speech > 82.0 and motion < 10.0:
            return "MONOLOGUE"
        elif motion < 3.0 and speech < 10.0:
            return "LOW_ACTIVITY"
        else:
            return "NORMAL"

    df["event_tag"] = df.apply(assign_tag, axis=1)
    return df


def create_engagement_timeline_chart(intervals_df: pd.DataFrame) -> go.Figure:
    """
    Generates a dark-themed Plotly timeline chart displaying:
    - Fused Engagement Score curve
    - Rolling 3-interval trendline
    - Video and Audio sub-scores
    - Optimal Engagement Target Band (55 - 85)
    - Highlighted Drop and Peak markers with descriptive tooltips
    """
    if intervals_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title="No Interval Data Available",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22"
        )
        return fig

    # Compute rolling trend
    df = intervals_df.sort_values("start_sec").copy()
    df["rolling_trend"] = df["fused_score"].rolling(window=3, min_periods=1, center=True).mean().round(1)

    fig = go.Figure()

    # Optimal engagement shaded band
    fig.add_hrect(
        y0=55, y1=85,
        fillcolor="rgba(16, 185, 129, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Target Attentive Band (55–85)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#10b981")
    )

    # Low engagement critical zone
    fig.add_hrect(
        y0=0, y1=40,
        fillcolor="rgba(239, 68, 68, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Attention Intervention Zone (<40)",
        annotation_position="bottom left",
        annotation_font=dict(size=10, color="#ef4444")
    )

    # Video sub-score line (dashed cyan)
    fig.add_trace(go.Scatter(
        x=df["start_sec"],
        y=df["video_score"],
        mode="lines",
        name="Video Sub-Score",
        line=dict(color="rgba(56, 189, 248, 0.5)", width=1.8, dash="dot"),
        hovertemplate="Time: %{x}s<br>Video Score: %{y:.1f}<extra></extra>"
    ))

    # Audio sub-score line (dashed purple)
    fig.add_trace(go.Scatter(
        x=df["start_sec"],
        y=df["audio_score"],
        mode="lines",
        name="Audio Sub-Score",
        line=dict(color="rgba(192, 132, 252, 0.5)", width=1.8, dash="dash"),
        hovertemplate="Time: %{x}s<br>Audio Score: %{y:.1f}<extra></extra>"
    ))

    # Rolling smoothed trend line
    fig.add_trace(go.Scatter(
        x=df["start_sec"],
        y=df["rolling_trend"],
        mode="lines",
        name="Smoothed Trend",
        line=dict(color="#f59e0b", width=2.5),
        hovertemplate="Time: %{x}s<br>Trend: %{y:.1f}<extra></extra>"
    ))

    # Fused Multimodal Score main curve
    fig.add_trace(go.Scatter(
        x=df["start_sec"],
        y=df["fused_score"],
        mode="lines+markers",
        name="Fused Engagement Score",
        line=dict(color="#00f2fe", width=3.2),
        marker=dict(size=6, color="#00f2fe"),
        hovertemplate=(
            "<b>Interval %{text}</b> (%{x}s)<br>"
            "Fused Score: <b>%{y:.1f} / 100</b><br>"
            "<extra></extra>"
        ),
        text=df["interval_idx"]
    ))

    # Marker annotations: Drops
    drops = df[df["event_tag"] == "ENGAGEMENT_DROP"]
    if not drops.empty:
        fig.add_trace(go.Scatter(
            x=drops["start_sec"],
            y=drops["fused_score"],
            mode="markers",
            name="Engagement Drop",
            marker=dict(symbol="triangle-down", size=14, color="#ef4444", line=dict(width=1.5, color="#ffffff")),
            hovertemplate="<b>Engagement Drop Detected!</b><br>Score: %{y:.1f}<br>Time: %{x}s<extra></extra>"
        ))

    # Marker annotations: Peaks
    peaks = df[df["event_tag"] == "INTERACTION_PEAK"]
    if not peaks.empty:
        fig.add_trace(go.Scatter(
            x=peaks["start_sec"],
            y=peaks["fused_score"],
            mode="markers",
            name="Interaction Peak",
            marker=dict(symbol="star", size=14, color="#10b981", line=dict(width=1.5, color="#ffffff")),
            hovertemplate="<b>Interaction Peak Detected!</b><br>Score: %{y:.1f}<br>Time: %{x}s<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0e1117",
        title=dict(
            text="Multimodal Classroom Engagement Timeline (Interval Analysis)",
            font=dict(size=16, color="#f1f5f9", family="sans-serif")
        ),
        xaxis=dict(
            title="Session Timeline (Seconds)",
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            showline=True,
            linecolor="#30363d"
        ),
        yaxis=dict(
            title="Engagement Index (0 – 100)",
            range=[0, 105],
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            showline=True,
            linecolor="#30363d"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#94a3b8")
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="x unified"
    )

    return fig


def create_metrics_breakdown_chart(intervals_df: pd.DataFrame) -> go.Figure:
    """
    Subplot chart showing granular components:
    1. Head Orientation Score & Motion Index
    2. Speech Ratio vs Silence Ratio
    """
    if intervals_df.empty:
        return go.Figure()

    df = intervals_df.sort_values("start_sec")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Visual Proxies: Head Orientation & Motion Index", "Auditory Proxies: Speech vs Silence Ratio (%)")
    )

    # Subplot 1: Head Orientation
    fig.add_trace(
        go.Scatter(
            x=df["start_sec"],
            y=df["head_orientation_score"],
            mode="lines",
            name="Gaze/Head Orientation",
            line=dict(color="#38bdf8", width=2.5)
        ),
        row=1, col=1
    )

    # Subplot 1: Motion Index
    fig.add_trace(
        go.Scatter(
            x=df["start_sec"],
            y=df["motion_index"],
            mode="lines",
            name="Motion Index",
            line=dict(color="#fbbf24", width=2.0, dash="dash")
        ),
        row=1, col=1
    )

    # Subplot 2: Speech Ratio
    fig.add_trace(
        go.Bar(
            x=df["start_sec"],
            y=df["speech_ratio"],
            name="Speech Ratio (%)",
            marker=dict(color="#34d399")
        ),
        row=2, col=1
    )

    # Subplot 2: Silence Ratio
    fig.add_trace(
        go.Bar(
            x=df["start_sec"],
            y=df["silence_ratio"],
            name="Silence Ratio (%)",
            marker=dict(color="#64748b")
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0e1117",
        height=480,
        barmode="stack",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1)
    )

    fig.update_xaxes(gridcolor="#21262d", title_text="Timeline (Seconds)", row=2, col=1)
    fig.update_xaxes(gridcolor="#21262d", row=1, col=1)
    fig.update_yaxes(gridcolor="#21262d", range=[0, 105], title_text="Score", row=1, col=1)
    fig.update_yaxes(gridcolor="#21262d", range=[0, 105], title_text="Percentage (%)", row=2, col=1)

    return fig
