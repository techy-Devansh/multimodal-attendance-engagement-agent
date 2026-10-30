"""
SkillWallet AI Teaching Agent Module
Rule-based, transparent, and interpretable pedagogical agent.
Analyzes multimodal engagement intervals and attendance patterns to formulate
actionable teaching recommendations with clear reasoning and specific classroom scripts.
Operates 100% locally with zero paid APIs or external LLM dependencies.
"""

from typing import List, Dict, Any
import pandas as pd


class PedagogicalAgent:
    """
    Transparent, rule-based AI reasoning engine for classroom engagement optimization.
    Every recommendation is grounded in empirical multimodal metrics.
    """

    def __init__(self):
        self.rules = [
            self._rule_sustained_engagement_drop,
            self._rule_prolonged_monologue,
            self._rule_prolonged_silence,
            self._rule_restlessness_and_fatigue,
            self._rule_interaction_peak_reinforcement,
            self._rule_attendance_fleeting_alert
        ]

    def analyze_session(
        self,
        intervals_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        presence_threshold_sec: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Runs all pedagogical rules across the session data and returns
        structured recommendation cards.
        """
        recommendations: List[Dict[str, Any]] = []

        if intervals_df.empty:
            return [{
                "rule_id": "R-INFO-00",
                "title": "Awaiting Session Data",
                "category": "System",
                "severity": "INFO",
                "time_window": "00:00",
                "trigger_metric": "Zero intervals recorded",
                "rationale": "No interval metrics are currently available to evaluate.",
                "recommendation": "Upload classroom video or launch a demo session to receive real-time pedagogical recommendations.",
                "suggested_action": "Load video file."
            }]

        for rule_fn in self.rules:
            res = rule_fn(intervals_df, attendance_df, presence_threshold_sec)
            if res:
                if isinstance(res, list):
                    recommendations.extend(res)
                else:
                    recommendations.append(res)

        # Sort recommendations by severity: HIGH first, then MEDIUM, then INFO
        severity_order = {"HIGH": 0, "MEDIUM": 1, "INFO": 2}
        recommendations.sort(key=lambda x: severity_order.get(x.get("severity", "INFO"), 3))

        return recommendations

    def _rule_sustained_engagement_drop(
        self,
        df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        threshold_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Rule R-RECAP-01: Detects sustained drops in fused engagement (< 42) lasting >= 10s.
        """
        recs = []
        df_sorted = df.sort_values("start_sec").reset_index(drop=True)

        drop_streak = []
        for idx, row in df_sorted.iterrows():
            if row["fused_score"] < 42.0:
                drop_streak.append(row)
            else:
                if len(drop_streak) >= 2:
                    start_t = drop_streak[0]["start_sec"]
                    end_t = drop_streak[-1]["end_sec"]
                    min_score = min(r["fused_score"] for r in drop_streak)
                    avg_score = round(sum(r["fused_score"] for r in drop_streak) / len(drop_streak), 1)

                    recs.append({
                        "rule_id": "R-RECAP-01",
                        "title": "Concept Check & Recap Needed",
                        "category": "Attention Recovery",
                        "severity": "HIGH",
                        "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                        "trigger_metric": f"Fused engagement dropped to {min_score}/100 (avg {avg_score}/100) for {int(end_t - start_t)}s",
                        "rationale": "Sustained downward head orientation combined with low vocal participation suggests cognitive overload or confusion with recently presented material.",
                        "recommendation": "Pause lecture progression immediately. Conduct a 90-second rapid recap of the core premise before moving to subsequent topics.",
                        "suggested_action": "Prompt class: 'Let's pause for 60 seconds. On a scale of 1-5, how comfortable is everyone with this step? Who can summarize the core constraint?'"
                    })
                drop_streak = []

        # Check trailing drop
        if len(drop_streak) >= 2:
            start_t = drop_streak[0]["start_sec"]
            end_t = drop_streak[-1]["end_sec"]
            min_score = min(r["fused_score"] for r in drop_streak)
            avg_score = round(sum(r["fused_score"] for r in drop_streak) / len(drop_streak), 1)
            recs.append({
                "rule_id": "R-RECAP-01",
                "title": "Concept Check & Recap Needed",
                "category": "Attention Recovery",
                "severity": "HIGH",
                "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                "trigger_metric": f"Fused engagement dropped to {min_score}/100 (avg {avg_score}/100) for {int(end_t - start_t)}s",
                "rationale": "Sustained downward head orientation combined with low vocal participation suggests cognitive overload or confusion with recently presented material.",
                "recommendation": "Pause lecture progression immediately. Conduct a 90-second rapid recap of the core premise before moving to subsequent topics.",
                "suggested_action": "Prompt class: 'Let's pause for 60 seconds. Can someone explain the transition we just made between these two formulas?'"
            })

        return recs

    def _rule_prolonged_monologue(
        self,
        df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        threshold_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Rule R-QA-02: Detects prolonged one-way speaking (> 80% speech) with static audience.
        """
        recs = []
        df_sorted = df.sort_values("start_sec").reset_index(drop=True)

        mono_streak = []
        for idx, row in df_sorted.iterrows():
            if row.get("speech_ratio", 0.0) >= 80.0 and row.get("motion_index", 5.0) < 12.0:
                mono_streak.append(row)
            else:
                if len(mono_streak) >= 3:
                    start_t = mono_streak[0]["start_sec"]
                    end_t = mono_streak[-1]["end_sec"]
                    recs.append({
                        "rule_id": "R-QA-02",
                        "title": "Transition from Monologue to Interactive Q&A",
                        "category": "Pedagogy & Pacing",
                        "severity": "MEDIUM",
                        "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                        "trigger_metric": f"Continuous high speech ratio (>80%) with static student posture for {int(end_t - start_t)}s",
                        "rationale": "Prolonged uninterrupted teacher monologue induces passive habituation. Introducing student dialogue reactivates working memory.",
                        "recommendation": "Implement a 2-minute Think-Pair-Share exercise or launch a live audience poll to shift from passive absorption to active synthesis.",
                        "suggested_action": "Teacher script: 'Turn to your neighbor for 90 seconds. Compare your approach to question 2 before we reveal the solution.'"
                    })
                mono_streak = []

        if len(mono_streak) >= 3:
            start_t = mono_streak[0]["start_sec"]
            end_t = mono_streak[-1]["end_sec"]
            recs.append({
                "rule_id": "R-QA-02",
                "title": "Transition from Monologue to Interactive Q&A",
                "category": "Pedagogy & Pacing",
                "severity": "MEDIUM",
                "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                "trigger_metric": f"Continuous high speech ratio (>80%) with static student posture for {int(end_t - start_t)}s",
                "rationale": "Prolonged uninterrupted teacher monologue induces passive habituation. Introducing student dialogue reactivates working memory.",
                "recommendation": "Implement a 2-minute Think-Pair-Share exercise or launch a live audience poll to shift from passive absorption to active synthesis.",
                "suggested_action": "Teacher script: 'Turn to your neighbor for 90 seconds. Compare your approach to question 2 before we reveal the solution.'"
            })

        return recs

    def _rule_prolonged_silence(
        self,
        df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        threshold_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Rule R-SILENCE-03: Detects prolonged dead silence (> 85% silence) lasting >= 15s.
        """
        recs = []
        df_sorted = df.sort_values("start_sec").reset_index(drop=True)

        silence_streak = []
        for idx, row in df_sorted.iterrows():
            if row.get("silence_ratio", 0.0) >= 85.0 and row.get("fused_score", 50.0) < 45.0:
                silence_streak.append(row)
            else:
                if len(silence_streak) >= 2:
                    start_t = silence_streak[0]["start_sec"]
                    end_t = silence_streak[-1]["end_sec"]
                    recs.append({
                        "rule_id": "R-SILENCE-03",
                        "title": "Unplanned Silence / Dead Time Intervention",
                        "category": "Classroom Dynamics",
                        "severity": "HIGH",
                        "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                        "trigger_metric": f"Auditory silence exceeded 85% for {int(end_t - start_t)}s with low engagement index",
                        "rationale": "Extended unplanned pauses during presentation create awkward dead air and cause attention drift.",
                        "recommendation": "If assigning silent individual work, explicitly frame the deliverable and countdown timer. If waiting for answers, scaffold with a multiple-choice hint.",
                        "suggested_action": "Teacher script: 'If anyone is stuck on line 4, remember that we first need to verify the matrix dimensions.'"
                    })
                silence_streak = []

        if len(silence_streak) >= 2:
            start_t = silence_streak[0]["start_sec"]
            end_t = silence_streak[-1]["end_sec"]
            recs.append({
                "rule_id": "R-SILENCE-03",
                "title": "Unplanned Silence / Dead Time Intervention",
                "category": "Classroom Dynamics",
                "severity": "HIGH",
                "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                "trigger_metric": f"Auditory silence exceeded 85% for {int(end_t - start_t)}s with low engagement index",
                "rationale": "Extended unplanned pauses during presentation create awkward dead air and cause attention drift.",
                "recommendation": "If assigning silent individual work, explicitly frame the deliverable and countdown timer. If waiting for answers, scaffold with a multiple-choice hint.",
                "suggested_action": "Teacher script: 'If anyone is stuck on line 4, remember that we first need to verify the matrix dimensions.'"
            })

        return recs

    def _rule_restlessness_and_fatigue(
        self,
        df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        threshold_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Rule R-RESTLESS-04: Detects high erratic motion index (> 25) with declining forward gaze.
        """
        recs = []
        df_sorted = df.sort_values("start_sec").reset_index(drop=True)

        high_motion_rows = df_sorted[
            (df_sorted["motion_index"] >= 25.0) &
            (df_sorted["head_orientation_score"] < 60.0)
        ]

        if len(high_motion_rows) >= 2:
            first_row = high_motion_rows.iloc[0]
            last_row = high_motion_rows.iloc[-1]
            start_t = first_row["start_sec"]
            end_t = last_row["end_sec"]

            recs.append({
                "rule_id": "R-RESTLESS-04",
                "title": "Restlessness & Pacing Calibration",
                "category": "Physical Engagement",
                "severity": "MEDIUM",
                "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                "trigger_metric": f"Elevated motion index (>25) combined with averted head pose across multiple intervals",
                "rationale": "Physical restlessness coupled with averted eye contact signals cognitive saturation, room discomfort, or fatigue.",
                "recommendation": "Incorporate a micro-break, change presentation modality (switch to live whiteboard / demo), or adjust speaking cadence.",
                "suggested_action": "Intervention: 'Let's take a 30-second breather. Stand up or stretch, and then we will apply this concept directly to a case study.'"
            })

        return recs

    def _rule_interaction_peak_reinforcement(
        self,
        df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        threshold_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Rule R-POSITIVE-05: Detects high interaction peaks (fused score >= 78) to reinforce effective teaching moments.
        """
        recs = []
        peaks = df[df["event_tag"] == "INTERACTION_PEAK"]

        if not peaks.empty:
            best_peak = peaks.sort_values("fused_score", ascending=False).iloc[0]
            start_t = best_peak["start_sec"]
            end_t = best_peak["end_sec"]
            max_score = best_peak["fused_score"]

            recs.append({
                "rule_id": "R-POSITIVE-05",
                "title": "Instructional High-Point Reinforcement",
                "category": "Positive Reinforcement",
                "severity": "INFO",
                "time_window": f"{int(start_t // 60):02d}:{int(start_t % 60):02d} – {int(end_t // 60):02d}:{int(end_t % 60):02d}",
                "trigger_metric": f"Peak multimodal engagement reached {max_score}/100 with active speech ({best_peak.get('speech_ratio', 0)}%) and forward gaze ({best_peak.get('head_orientation_score', 0)}/100)",
                "rationale": "Co-occurrence of dialogue bursts and forward facial orientation marks maximum participatory engagement.",
                "recommendation": "Catalog the teaching method used during this window (e.g. cold-call discussion, live coding demonstration, provocative query) as a reusable instructional template.",
                "suggested_action": "Pedagogical insight: Replicate this conversational entry point at the start of complex theoretical units."
            })

        return recs

    def _rule_attendance_fleeting_alert(
        self,
        df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        threshold_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Rule R-ATTENDANCE-06: Identifies students who failed to meet sustained presence threshold.
        """
        recs = []
        if attendance_df.empty:
            return recs

        fleeting = attendance_df[attendance_df["sustained_present"] == 0]
        total = len(attendance_df)

        if len(fleeting) > 0 and total > 0:
            fleeting_pct = round((len(fleeting) / float(total)) * 100.0, 1)
            ids_str = ", ".join(fleeting["track_id"].tolist()[:5])
            if len(fleeting) > 5:
                ids_str += f" (+{len(fleeting) - 5} more)"

            recs.append({
                "rule_id": "R-ATTENDANCE-06",
                "title": "Sustained Presence Attendance Discrepancy",
                "category": "Attendance & Integrity",
                "severity": "MEDIUM" if fleeting_pct > 25.0 else "INFO",
                "time_window": "Entire Session",
                "trigger_metric": f"{len(fleeting)} of {total} tracked attendees ({fleeting_pct}%) logged visible duration < {threshold_sec}s threshold",
                "rationale": "Short visible durations indicate either fleeting passerby detections, camera blind spot occlusion at room boundaries, or late entrance.",
                "recommendation": f"Verify physical attendance log against flagged anonymous IDs: [{ids_str}]. Consider adjusting camera field of view or presence threshold.",
                "suggested_action": "Action: Cross-reference manual roll call for the flagged positions."
            })

        return recs
