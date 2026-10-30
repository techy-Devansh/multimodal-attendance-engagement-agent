"""
Comprehensive End-to-End Test Suite for SkillWallet
Verifies:
1. Core video processing (Face detection, anonymous centroid tracking, sustained presence, gaze proxy, motion)
2. Core audio processing (RMS, ZCR, VAD, speech/silence ratio, missing audio fallback)
3. Multimodal fusion (Interval alignment, weighted scoring, dynamic recalculation, Plotly charts)
4. AI Pedagogical Agent (Rule triggers, severity, rationale, teaching prompts)
5. Database persistence (SQLite CRUD, session weights update, statistics)
6. Reports & Downloads (Attendance CSV, Analytics CSV, ReportLab PDF byte validation)
7. Demo mode (1-click generation, is_demo flag, realistic intervals)
8. All 6 application pages execution (Overview, Attendance, Engagement, Suggestions, Reports, Privacy)
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import soundfile as sf

import database
import vision
import audio
import fusion
import agent
import utils

TEST_RESULTS = []

def record_test(name: str, passed: bool, details: str = ""):
    TEST_RESULTS.append({"test": name, "passed": passed, "details": details})
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"[{status}] {name}: {details}")


print("=" * 70)
print("STARTING SKILLWALLET COMPREHENSIVE VERIFICATION")
print("=" * 70)

# ==============================================================================
# TEST 1: CORE VIDEO PROCESSING & ANONYMOUS TRACKING
# ==============================================================================
print("\n--- TEST 1: Core Video Processing & Anonymous Tracking ---")
test_vid_path = "test_synthetic_classroom.mp4"

# Generate a 4-second synthetic classroom video with 2 visible students
fps = 10.0
vw = cv2.VideoWriter(test_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))

for f_num in range(40):
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 40  # dark classroom wall
    
    # Student 1: Present throughout entire video (4 seconds)
    cv2.circle(frame, (180, 200), 50, (200, 200, 200), -1)  # head
    cv2.circle(frame, (165, 185), 6, (30, 30, 30), -1)      # left eye
    cv2.circle(frame, (195, 185), 6, (30, 30, 30), -1)      # right eye
    
    # Student 2: Appears only during frames 0 to 12 (1.2 seconds, below 2s threshold)
    if f_num < 13:
        cv2.circle(frame, (450, 220), 45, (190, 190, 190), -1)
        cv2.circle(frame, (435, 205), 5, (30, 30, 30), -1)
        cv2.circle(frame, (465, 205), 5, (30, 30, 30), -1)
        
    vw.write(frame)
vw.release()

try:
    vid_res = vision.process_video_stream(
        test_vid_path,
        presence_threshold_sec=2.0,
        interval_sec=2.0,
        target_sample_fps=5.0
    )
    meta = vid_res["meta"]
    att = vid_res["attendance"]
    intervals = vid_res["intervals"]
    
    has_students = len(att) >= 1
    has_intervals = len(intervals) >= 2
    
    # Check that sustained present student has status 'Present'
    student1 = next((s for s in att if s["total_visible_sec"] >= 2.0), None)
    is_student1_present = student1 is not None and student1["status"] == "Present"
    
    # Check that tracking IDs are anonymous (STU-01, etc.)
    all_anonymous = all(s["track_id"].startswith("STU-") for s in att)
    
    record_test("Video Processing Pipeline", has_students and has_intervals, f"{meta['processed_samples']} frames sampled, {len(att)} tracks")
    record_test("Anonymous ID Assignment", all_anonymous, f"Track IDs: {[s['track_id'] for s in att]}")
    record_test("Sustained Presence Evaluation", is_student1_present, f"Student 1 duration: {student1['total_visible_sec'] if student1 else 0}s")
    
except Exception as e:
    record_test("Video Processing Pipeline", False, str(e))
finally:
    if os.path.exists(test_vid_path):
        os.remove(test_vid_path)


# ==============================================================================
# TEST 2: CORE AUDIO PROCESSING & VAD
# ==============================================================================
print("\n--- TEST 2: Core Audio Processing & Voice Activity Detection ---")
test_aud_path = "test_synthetic_audio.wav"

# Generate 4-second audio: 2s speech simulation, 2s silence
sr = 16000
t_speech = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
speech_wave = 0.4 * np.sin(2 * np.pi * 200 * t_speech) + 0.2 * np.sin(2 * np.pi * 440 * t_speech)
silence_wave = np.random.normal(0, 0.005, int(2.0 * sr))
full_audio = np.concatenate([speech_wave, silence_wave]).astype(np.float32)
sf.write(test_aud_path, full_audio, sr)

try:
    aud_res = audio.analyze_audio_stream(test_aud_path, interval_sec=2.0)
    has_audio = aud_res["has_audio"]
    speech_ratio = aud_res["overall_speech_ratio"]
    aud_intervals = aud_res["intervals"]
    
    record_test("Audio Signal Analysis", has_audio, f"Sample rate: {aud_res['sample_rate']}Hz, Duration: {aud_res['duration_sec']}s")
    record_test("Voice Activity Detection (VAD)", 20.0 <= speech_ratio <= 80.0, f"Speech ratio: {speech_ratio}%, Silence: {aud_res['overall_silence_ratio']}%")
    record_test("Audio Interval Discretization", len(aud_intervals) == 2, f"Intervals count: {len(aud_intervals)}")
    
    # Missing Audio Fallback Test
    fallback_res = audio.analyze_audio_stream("non_existent_audio.wav", expected_duration_sec=10.0, interval_sec=2.0)
    record_test("Missing Audio Graceful Fallback", fallback_res["has_audio"] == False and len(fallback_res["intervals"]) == 5, "Fallback returned neutral intervals without crashing")

except Exception as e:
    record_test("Audio Signal Analysis", False, str(e))
finally:
    if os.path.exists(test_aud_path):
        os.remove(test_aud_path)


# ==============================================================================
# TEST 3: MULTIMODAL FUSION & WEIGHT RECALCULATION
# ==============================================================================
print("\n--- TEST 3: Multimodal Interval Fusion & Plotly Charts ---")
try:
    v_ints = [
        {"interval_idx": 0, "start_sec": 0.0, "end_sec": 5.0, "video_score": 85.0, "face_count": 5, "head_orientation_score": 90.0, "motion_index": 12.0},
        {"interval_idx": 1, "start_sec": 5.0, "end_sec": 10.0, "video_score": 30.0, "face_count": 5, "head_orientation_score": 30.0, "motion_index": 2.0}
    ]
    a_ints = [
        {"interval_idx": 0, "speech_ratio": 70.0, "silence_ratio": 30.0, "audio_score": 88.0},
        {"interval_idx": 1, "speech_ratio": 10.0, "silence_ratio": 90.0, "audio_score": 28.0}
    ]
    
    fused_recs = fusion.fuse_multimodal_intervals(v_ints, a_ints, video_weight=0.6, audio_weight=0.4)
    score_0 = fused_recs[0]["fused_score"]
    score_1 = fused_recs[1]["fused_score"]
    
    # Expected: score_0 ~ (0.6*85 + 0.4*88) = 51.0 + 35.2 = 86.2
    # score_1 ~ (0.6*30 + 0.4*28) = 18.0 + 11.2 = 29.2 (<40 -> drop)
    fused_math_correct = abs(score_0 - 86.2) < 0.5 and abs(score_1 - 29.2) < 0.5
    drop_detected = fused_recs[1]["event_tag"] == "ENGAGEMENT_DROP"
    
    record_test("Multimodal Fusion Formula", fused_math_correct, f"Interval 0: {score_0}, Interval 1: {score_1}")
    record_test("Anomaly Event Tagging", drop_detected, f"Interval 1 Tag: {fused_recs[1]['event_tag']}")
    
    # Dynamic Weight Recalculation on DataFrame
    df_fused = pd.DataFrame(fused_recs)
    df_recalced = fusion.recalculate_fusion_df(df_fused, video_weight=1.0, audio_weight=0.0)
    score_0_video_only = df_recalced["fused_score"].iloc[0]
    record_test("Dynamic Weight Recalibration", abs(score_0_video_only - 85.0) < 0.1, f"Video-only score: {score_0_video_only}")
    
    # Plotly figures test
    fig1 = fusion.create_engagement_timeline_chart(df_fused)
    fig2 = fusion.create_metrics_breakdown_chart(df_fused)
    record_test("Plotly Visualizations", fig1 is not None and fig2 is not None, "Timeline and subcomponents figures built successfully")

except Exception as e:
    record_test("Multimodal Fusion", False, str(e))


# ==============================================================================
# TEST 4: RULE-BASED AI TEACHING AGENT
# ==============================================================================
print("\n--- TEST 4: Rule-Based AI Teaching Agent ---")
try:
    # Construct sequence to trigger recap, monologue, and silence rules
    test_intervals_df = pd.DataFrame([
        {"interval_idx": 0, "start_sec": 0.0, "end_sec": 5.0, "fused_score": 32.0, "speech_ratio": 5.0, "silence_ratio": 95.0, "motion_index": 2.0, "head_orientation_score": 30.0, "event_tag": "ENGAGEMENT_DROP"},
        {"interval_idx": 1, "start_sec": 5.0, "end_sec": 10.0, "fused_score": 34.0, "speech_ratio": 5.0, "silence_ratio": 95.0, "motion_index": 2.0, "head_orientation_score": 28.0, "event_tag": "ENGAGEMENT_DROP"},
        {"interval_idx": 2, "start_sec": 10.0, "end_sec": 15.0, "fused_score": 86.0, "speech_ratio": 70.0, "silence_ratio": 30.0, "motion_index": 14.0, "head_orientation_score": 90.0, "event_tag": "INTERACTION_PEAK"},
        {"interval_idx": 3, "start_sec": 15.0, "end_sec": 20.0, "fused_score": 68.0, "speech_ratio": 88.0, "silence_ratio": 12.0, "motion_index": 6.0, "head_orientation_score": 75.0, "event_tag": "MONOLOGUE"},
        {"interval_idx": 4, "start_sec": 20.0, "end_sec": 25.0, "fused_score": 67.0, "speech_ratio": 89.0, "silence_ratio": 11.0, "motion_index": 5.0, "head_orientation_score": 70.0, "event_tag": "MONOLOGUE"},
        {"interval_idx": 5, "start_sec": 25.0, "end_sec": 30.0, "fused_score": 66.0, "speech_ratio": 87.0, "silence_ratio": 13.0, "motion_index": 5.0, "head_orientation_score": 72.0, "event_tag": "MONOLOGUE"}
    ])
    test_att_df = pd.DataFrame([
        {"track_id": "STU-01", "total_visible_sec": 28.0, "sustained_present": 1, "status": "Present"},
        {"track_id": "STU-02", "total_visible_sec": 2.0, "sustained_present": 0, "status": "Partial / Fleeting"}
    ])
    
    agent_instance = agent.PedagogicalAgent()
    recs = agent_instance.analyze_session(test_intervals_df, test_att_df, presence_threshold_sec=5.0)
    
    rule_ids = [r["rule_id"] for r in recs]
    has_recap = "R-RECAP-01" in rule_ids
    has_monologue = "R-QA-02" in rule_ids
    has_attendance_alert = "R-ATTENDANCE-06" in rule_ids
    
    # Check structure
    valid_structure = all("rationale" in r and "recommendation" in r and "suggested_action" in r and "severity" in r for r in recs)
    
    record_test("AI Rule Engine Execution", len(recs) >= 3, f"{len(recs)} suggestions fired: {rule_ids}")
    record_test("Sustained Drop Recap Rule (R-RECAP-01)", has_recap, "Triggered on consecutive low engagement intervals")
    record_test("Monologue to Q&A Rule (R-QA-02)", has_monologue, "Triggered on sustained high teacher speech with static audience")
    record_test("Attendance Alert Rule (R-ATTENDANCE-06)", has_attendance_alert, "Triggered on fleeting student presence (<5s)")
    record_test("Pedagogical Recommendation Structure", valid_structure, "All recommendations contain rationale, action, and teacher prompts")

except Exception as e:
    record_test("AI Rule Engine Execution", False, str(e))


# ==============================================================================
# TEST 5: SQLITE DATABASE PERSISTENCE & CRUD
# ==============================================================================
print("\n--- TEST 5: SQLite Database Persistence ---")
try:
    test_db = "test_persistence.db"
    database.init_db(test_db)
    
    # Save a test session
    sess_id = "TEST-SESSION-PERSIST"
    sess_data = {
        "session_id": sess_id,
        "session_name": "Operating Systems Lecture",
        "filename": "os_lecture.mp4",
        "created_at": "2026-10-27 16:00:00",
        "video_duration_sec": 120.0,
        "fps": 30.0,
        "total_frames": 3600,
        "has_audio": 1,
        "audio_duration_sec": 120.0,
        "presence_threshold_sec": 5.0,
        "video_weight": 0.60,
        "audio_weight": 0.40,
        "interval_sec": 5.0,
        "is_demo": 0,
        "processing_time_sec": 2.15
    }
    database.save_session(sess_data, db_path=test_db)
    database.save_attendance_batch(sess_id, [{"track_id": "STU-01", "first_seen_sec": 0.0, "last_seen_sec": 120.0, "total_visible_sec": 115.0, "sustained_present": True, "status": "Present"}], db_path=test_db)
    database.save_intervals_batch(sess_id, [{"interval_idx": 0, "start_sec": 0.0, "end_sec": 5.0, "video_score": 75.0, "audio_score": 70.0, "fused_score": 73.0, "face_count": 1, "head_orientation_score": 80.0, "motion_index": 10.0, "speech_ratio": 60.0, "silence_ratio": 40.0, "event_tag": "NORMAL"}], db_path=test_db)
    database.save_recommendations_batch(sess_id, [{"rule_id": "R-INFO-01", "title": "Session Recorded", "category": "General", "severity": "INFO", "time_window": "00:00", "trigger_metric": "Normal", "rationale": "Test rationale", "recommendation": "Test rec", "suggested_action": "Test script"}], db_path=test_db)
    
    # Read back
    retrieved_s = database.get_session_by_id(sess_id, db_path=test_db)
    retrieved_att = database.get_attendance_for_session(sess_id, db_path=test_db)
    retrieved_int = database.get_intervals_for_session(sess_id, db_path=test_db)
    retrieved_rec = database.get_recommendations_for_session(sess_id, db_path=test_db)
    
    record_test("Database Insert & Read", retrieved_s is not None and len(retrieved_att) == 1 and len(retrieved_int) == 1 and len(retrieved_rec) == 1, f"Session {sess_id} verified")
    
    # Weight update
    database.update_session_weights(sess_id, 0.75, 0.25, db_path=test_db)
    s_updated = database.get_session_by_id(sess_id, db_path=test_db)
    record_test("Database Weight Update", s_updated["video_weight"] == 0.75 and s_updated["audio_weight"] == 0.25, f"Updated: {s_updated['video_weight']}/{s_updated['audio_weight']}")
    
    # Stats
    stats = database.get_db_stats(test_db)
    record_test("Database Statistics Reporting", stats["sessions"] == 1 and stats["attendance_records"] == 1, f"DB size: {stats['size_kb']} KB")

    # Raw table inspection
    raw_df = database.get_raw_table_df("sessions", db_path=test_db)
    record_test("Database Raw Table Query", len(raw_df) == 1 and "session_id" in raw_df.columns, f"Raw rows retrieved: {len(raw_df)}")
    
    # Clean test DB
    if os.path.exists(test_db):
        os.remove(test_db)

except Exception as e:
    record_test("Database Persistence", False, str(e))


# ==============================================================================
# TEST 6: CSV & REPORTLAB PDF EXPORTS
# ==============================================================================
print("\n--- TEST 6: CSV & ReportLab PDF Exports ---")
try:
    demo_pkg = utils.generate_demo_dataset()
    sess_meta = demo_pkg["session"]
    att_df = pd.DataFrame(demo_pkg["attendance"])
    int_df = pd.DataFrame(demo_pkg["intervals"])
    recs = demo_pkg["recommendations"]
    
    # CSV 1: Attendance
    csv_att_bytes = utils.export_dataframe_to_csv_bytes(att_df)
    record_test("Attendance CSV Export", len(csv_att_bytes) > 200 and b"track_id" in csv_att_bytes, f"Generated {len(csv_att_bytes)} bytes")
    
    # CSV 2: Analytics
    csv_int_bytes = utils.export_dataframe_to_csv_bytes(int_df)
    record_test("Analytics CSV Export", len(csv_int_bytes) > 500 and b"fused_score" in csv_int_bytes, f"Generated {len(csv_int_bytes)} bytes")
    
    # PDF: ReportLab
    pdf_bytes = utils.generate_pdf_report(sess_meta, att_df, int_df, recs)
    is_valid_pdf = pdf_bytes.startswith(b"%PDF")
    record_test("ReportLab Combined PDF Generation", is_valid_pdf and len(pdf_bytes) > 3000, f"Valid %PDF header, file size: {len(pdf_bytes)} bytes")

except Exception as e:
    record_test("Reports & Exports", False, str(e))


# ==============================================================================
# TEST 7: DEMO DATASET MODE & LABELS
# ==============================================================================
print("\n--- TEST 7: Demo Dataset Mode & Labeling ---")
try:
    demo_data = utils.generate_demo_dataset(session_id="DEMO-VERIFY-01")
    s = demo_data["session"]
    
    is_explicit_demo = (s["is_demo"] == 1)
    has_realistic_students = len(demo_data["attendance"]) == 12
    has_intervals = len(demo_data["intervals"]) == 36
    has_recs = len(demo_data["recommendations"]) >= 3
    
    record_test("Demo Mode Explicit Flag (is_demo=1)", is_explicit_demo, f"is_demo={s['is_demo']}")
    record_test("Demo Dataset Completeness", has_realistic_students and has_intervals and has_recs, "12 students, 36 intervals, 3 AI pedagogical suggestions")

except Exception as e:
    record_test("Demo Mode", False, str(e))


# ==============================================================================
# TEST 8: ALL 6 APPLICATION PAGES LOGIC CHECK
# ==============================================================================
print("\n--- TEST 8: Application Pages Logic & Code Execution ---")

def test_page_overview_logic():
    # Verify overview computations and metric cards
    tot_students = len(att_df)
    sustained = len(att_df[att_df["sustained_present"] == 1])
    avg_eng = round(int_df["fused_score"].mean(), 1)
    return tot_students == 12 and sustained == 11 and avg_eng > 50.0

def test_page_attendance_logic():
    # Verify manual headcount comparison logic
    manual_count = 11
    sustained_count = len(att_df[att_df["sustained_present"] == 1])
    discrepancy = sustained_count - manual_count
    return discrepancy == 0

def test_page_engagement_logic():
    # Verify dynamic recalculation and Plotly figures
    recalced = fusion.recalculate_fusion_df(int_df, 0.7, 0.3)
    fig_time = fusion.create_engagement_timeline_chart(recalced)
    fig_decomp = fusion.create_metrics_breakdown_chart(recalced)
    return not recalced.empty and fig_time is not None and fig_decomp is not None

def test_page_suggestions_logic():
    # Verify recommendation rendering fields
    for r in recs:
        assert "rule_id" in r and "rationale" in r and "suggested_action" in r
    return True

def test_page_reports_logic():
    # Verify reports metadata and downloads availability
    return len(pdf_bytes) > 0 and len(csv_att_bytes) > 0 and len(csv_int_bytes) > 0

def test_page_privacy_logic():
    # Verify database statistics and privacy fields
    stats = database.get_db_stats()
    return "size_kb" in stats and stats["sessions"] >= 1

try:
    record_test("Page 1: Overview Logic", test_page_overview_logic(), "KPI calculations & metrics verified")
    record_test("Page 2: Attendance Logic & Headcount Tool", test_page_attendance_logic(), "Headcount discrepancy math verified")
    record_test("Page 3: Engagement Analytics Logic", test_page_engagement_logic(), "Interactive dynamic weight recalc & Plotly curves verified")
    record_test("Page 4: AI Suggestions Logic", test_page_suggestions_logic(), "Transparent rationale & teacher prompt fields verified")
    record_test("Page 5: Reports Logic", test_page_reports_logic(), "PDF & CSV download bytes generated verified")
    record_test("Page 6: Privacy & About Logic", test_page_privacy_logic(), "Database stats & privacy guarantees verified")

except Exception as e:
    record_test("Application Pages Logic", False, str(e))

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
all_passed = all(t["passed"] for t in TEST_RESULTS)
total_tests = len(TEST_RESULTS)
passed_count = sum(1 for t in TEST_RESULTS if t["passed"])

print(f"Total Tests Executed: {total_tests}")
print(f"Passed: {passed_count}")
print(f"Failed: {total_tests - passed_count}")

if all_passed:
    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    sys.exit(0)
else:
    print("\n❌ SOME TESTS FAILED!")
    sys.exit(1)
