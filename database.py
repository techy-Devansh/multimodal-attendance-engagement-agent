"""
SkillWallet Database Management Module
Handles SQLite persistence for sessions, attendance, engagement intervals, and AI recommendations.
Uses safe connection lifecycle management with guaranteed connection cleanup for Windows compatibility.
"""

import sqlite3
import os
import contextlib
import pandas as pd
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = "skillwallet.db"


@contextlib.contextmanager
def managed_connection(db_path: str = DEFAULT_DB_PATH):
    """Context manager ensuring connections are always closed immediately after use."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize database tables with schema definition."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                video_duration_sec REAL DEFAULT 0.0,
                fps REAL DEFAULT 0.0,
                total_frames INTEGER DEFAULT 0,
                has_audio INTEGER DEFAULT 0,
                audio_duration_sec REAL DEFAULT 0.0,
                presence_threshold_sec REAL DEFAULT 5.0,
                video_weight REAL DEFAULT 0.60,
                audio_weight REAL DEFAULT 0.40,
                interval_sec REAL DEFAULT 5.0,
                is_demo INTEGER DEFAULT 0,
                processing_time_sec REAL DEFAULT 0.0
            )
        """)

        # Attendance records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                first_seen_sec REAL NOT NULL,
                last_seen_sec REAL NOT NULL,
                total_visible_sec REAL NOT NULL,
                sustained_present INTEGER NOT NULL,
                status TEXT NOT NULL,
                avg_confidence REAL DEFAULT 1.0,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)

        # Multimodal engagement intervals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engagement_intervals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                interval_idx INTEGER NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                video_score REAL NOT NULL,
                audio_score REAL NOT NULL,
                fused_score REAL NOT NULL,
                face_count INTEGER NOT NULL,
                head_orientation_score REAL NOT NULL,
                motion_index REAL NOT NULL,
                speech_ratio REAL NOT NULL,
                silence_ratio REAL NOT NULL,
                event_tag TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)

        # AI Recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                time_window TEXT NOT NULL,
                trigger_metric TEXT NOT NULL,
                rationale TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)

        conn.commit()


def save_session(session_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace a session header row."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, session_name, filename, created_at, video_duration_sec,
                fps, total_frames, has_audio, audio_duration_sec, presence_threshold_sec,
                video_weight, audio_weight, interval_sec, is_demo, processing_time_sec
            ) VALUES (
                :session_id, :session_name, :filename, :created_at, :video_duration_sec,
                :fps, :total_frames, :has_audio, :audio_duration_sec, :presence_threshold_sec,
                :video_weight, :audio_weight, :interval_sec, :is_demo, :processing_time_sec
            )
        """, session_data)
        conn.commit()


def save_attendance_batch(session_id: str, records: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> None:
    """Save all student attendance entries for a session."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE session_id = ?", (session_id,))
        for rec in records:
            cursor.execute("""
                INSERT INTO attendance (
                    session_id, track_id, first_seen_sec, last_seen_sec,
                    total_visible_sec, sustained_present, status, avg_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                rec["track_id"],
                rec["first_seen_sec"],
                rec["last_seen_sec"],
                rec["total_visible_sec"],
                1 if rec["sustained_present"] else 0,
                rec["status"],
                rec.get("avg_confidence", 1.0)
            ))
        conn.commit()


def save_intervals_batch(session_id: str, records: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> None:
    """Save all engagement intervals for a session."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM engagement_intervals WHERE session_id = ?", (session_id,))
        for r in records:
            cursor.execute("""
                INSERT INTO engagement_intervals (
                    session_id, interval_idx, start_sec, end_sec, video_score,
                    audio_score, fused_score, face_count, head_orientation_score,
                    motion_index, speech_ratio, silence_ratio, event_tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                r["interval_idx"],
                r["start_sec"],
                r["end_sec"],
                r["video_score"],
                r["audio_score"],
                r["fused_score"],
                r["face_count"],
                r["head_orientation_score"],
                r["motion_index"],
                r["speech_ratio"],
                r["silence_ratio"],
                r.get("event_tag", "NORMAL")
            ))
        conn.commit()


def save_recommendations_batch(session_id: str, records: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> None:
    """Save all rule-based AI recommendations for a session."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ai_recommendations WHERE session_id = ?", (session_id,))
        for r in records:
            cursor.execute("""
                INSERT INTO ai_recommendations (
                    session_id, rule_id, title, category, severity,
                    time_window, trigger_metric, rationale, recommendation, suggested_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                r["rule_id"],
                r["title"],
                r["category"],
                r["severity"],
                r["time_window"],
                r["trigger_metric"],
                r["rationale"],
                r["recommendation"],
                r.get("suggested_action", "")
            ))
        conn.commit()


def get_all_sessions(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all sessions ordered by creation date descending."""
    init_db(db_path)
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_session_by_id(session_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve session details by ID."""
    init_db(db_path)
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_attendance_for_session(session_id: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Get attendance records as a pandas DataFrame."""
    init_db(db_path)
    with managed_connection(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT track_id, first_seen_sec, last_seen_sec, total_visible_sec, sustained_present, status, avg_confidence "
            "FROM attendance WHERE session_id = ? ORDER BY track_id ASC",
            conn,
            params=(session_id,)
        )
        return df


def get_intervals_for_session(session_id: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Get engagement intervals as a pandas DataFrame."""
    init_db(db_path)
    with managed_connection(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT interval_idx, start_sec, end_sec, video_score, audio_score, fused_score, "
            "face_count, head_orientation_score, motion_index, speech_ratio, silence_ratio, event_tag "
            "FROM engagement_intervals WHERE session_id = ? ORDER BY interval_idx ASC",
            conn,
            params=(session_id,)
        )
        return df


def get_recommendations_for_session(session_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Get AI recommendations as a list of dictionaries."""
    init_db(db_path)
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rule_id, title, category, severity, time_window, trigger_metric, rationale, recommendation, suggested_action "
            "FROM ai_recommendations WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return [dict(r) for r in cursor.fetchall()]


def update_session_weights(session_id: str, video_weight: float, audio_weight: float, db_path: str = DEFAULT_DB_PATH) -> None:
    """Update fusion weights in the session record."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET video_weight = ?, audio_weight = ? WHERE session_id = ?",
            (video_weight, audio_weight, session_id)
        )
        conn.commit()


def delete_session(session_id: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """Delete a session and all cascading records."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM engagement_intervals WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM ai_recommendations WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def delete_all_data(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clear all records from all tables."""
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance")
        cursor.execute("DELETE FROM engagement_intervals")
        cursor.execute("DELETE FROM ai_recommendations")
        cursor.execute("DELETE FROM sessions")
        conn.commit()


def get_db_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Return summary statistics of the database."""
    init_db(db_path)
    with managed_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM attendance")
        attendance_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM engagement_intervals")
        interval_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ai_recommendations")
        rec_count = cursor.fetchone()[0]

    db_size_kb = 0.0
    if os.path.exists(db_path):
        db_size_kb = round(os.path.getsize(db_path) / 1024.0, 2)

    return {
        "sessions": session_count,
        "attendance_records": attendance_count,
        "intervals": interval_count,
        "recommendations": rec_count,
        "size_kb": db_size_kb,
        "db_path": os.path.abspath(db_path)
    }


def get_raw_table_df(table_name: str, limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve raw records from a specified database table."""
    allowed = {"sessions", "attendance", "engagement_intervals", "ai_recommendations"}
    if table_name not in allowed:
        return pd.DataFrame()
    init_db(db_path)
    with managed_connection(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {int(limit)}", conn)
        return df

