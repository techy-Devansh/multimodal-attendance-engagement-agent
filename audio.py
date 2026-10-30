"""
SkillWallet Audio Processing Module
Analyzes classroom audio streams for speech activity, silence intervals,
and interaction dynamics using signal processing (RMS Energy, Zero-Crossing Rate, and VAD).
Handles missing or corrupted audio streams gracefully.
"""

import os
import math
import numpy as np
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple


def compute_frame_rms(audio_segment: np.ndarray) -> float:
    """Computes Root Mean Square (RMS) energy for an audio frame."""
    if len(audio_segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_segment))))


def compute_frame_zcr(audio_segment: np.ndarray) -> float:
    """Computes Zero-Crossing Rate (ZCR) for an audio frame."""
    if len(audio_segment) < 2:
        return 0.0
    signs = np.sign(audio_segment)
    # Replace zeros with 1 to avoid zero multiplication artifacts
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs))) / (2.0 * (len(audio_segment) - 1))
    return float(crossings)


def load_audio_file(file_path: str, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """
    Safely loads audio using soundfile (supporting WAV, MP3, FLAC, OGG, etc.).
    Returns (audio_signal, sample_rate, error_message).
    """
    if not os.path.exists(file_path):
        return None, target_sr, f"File does not exist: {file_path}"

    try:
        data, sr = sf.read(file_path)
    except Exception as e:
        return None, target_sr, f"Audio format or container not decodable directly by SoundFile: {str(e)}"

    if data is None or len(data) == 0:
        return None, target_sr, "Audio file is empty."

    # Convert stereo/multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Basic normalization to [-1.0, 1.0] if integer or extreme scale
    max_val = np.max(np.abs(data))
    if max_val > 0.0:
        data = data / max_val

    # Simple resampling if required using linear interpolation
    if sr != target_sr and sr > 0:
        new_len = int(round(len(data) * float(target_sr) / sr))
        data = np.interp(
            np.linspace(0, len(data), new_len, endpoint=False),
            np.arange(len(data)),
            data
        )
        sr = target_sr

    return data.astype(np.float32), sr, None


def analyze_audio_stream(
    audio_path: Optional[str] = None,
    audio_data: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    expected_duration_sec: float = 0.0,
    interval_sec: float = 5.0
) -> Dict[str, Any]:
    """
    Analyzes audio data or audio file for Voice Activity Detection (VAD),
    speech presence, silence duration, and interaction dynamics.
    If audio is unavailable or missing, returns a structured fallback response.
    """
    # Check if audio source is provided
    if audio_data is None:
        if not audio_path or not os.path.exists(audio_path):
            return {
                "has_audio": False,
                "error_msg": "No audio file provided or audio stream unavailable.",
                "duration_sec": expected_duration_sec,
                "intervals": generate_neutral_intervals(expected_duration_sec, interval_sec)
            }

        audio_data, sr, err = load_audio_file(audio_path, target_sr=sample_rate)
        if err or audio_data is None:
            return {
                "has_audio": False,
                "error_msg": f"Audio processing fallback: {err}",
                "duration_sec": expected_duration_sec,
                "intervals": generate_neutral_intervals(expected_duration_sec, interval_sec)
            }
        sample_rate = sr

    duration_sec = len(audio_data) / float(sample_rate) if sample_rate > 0 else 0.0
    if duration_sec <= 0.0:
        return {
            "has_audio": False,
            "error_msg": "Audio duration is 0 seconds.",
            "duration_sec": expected_duration_sec,
            "intervals": generate_neutral_intervals(expected_duration_sec, interval_sec)
        }

    # Voice Activity Detection (VAD) via short-time frames
    frame_len = int(sample_rate * 0.030)  # 30 ms analysis frame
    hop_len = int(sample_rate * 0.015)    # 15 ms hop (50% overlap)
    num_frames = max(1, (len(audio_data) - frame_len) // hop_len + 1)

    frame_rms_list = []
    frame_zcr_list = []

    for f_i in range(num_frames):
        start_idx = f_i * hop_len
        end_idx = start_idx + frame_len
        frame = audio_data[start_idx:end_idx]
        frame_rms_list.append(compute_frame_rms(frame))
        frame_zcr_list.append(compute_frame_zcr(frame))

    rms_arr = np.array(frame_rms_list)
    zcr_arr = np.array(frame_zcr_list)

    # Dynamic noise floor estimation (15th percentile of RMS)
    noise_floor = float(np.percentile(rms_arr, 15)) if len(rms_arr) > 0 else 0.01
    speech_threshold = max(0.02, noise_floor * 2.2)

    # Classify each frame: 1 = speech, 0 = silence/background noise
    # Human speech typically has vocal pitch/formants: moderate ZCR (0.015 to 0.45)
    is_speech = (rms_arr > speech_threshold) & (zcr_arr >= 0.015) & (zcr_arr <= 0.45)

    # Segment into interval buckets
    effective_duration = max(duration_sec, expected_duration_sec)
    num_intervals = max(1, int(math.ceil(effective_duration / interval_sec)))
    interval_records: List[Dict[str, Any]] = []

    frame_duration_sec = hop_len / float(sample_rate)

    for i in range(num_intervals):
        i_start = i * interval_sec
        i_end = min(effective_duration, (i + 1) * interval_sec)
        i_dur = max(0.001, i_end - i_start)

        start_frame = int(i_start / frame_duration_sec)
        end_frame = int(i_end / frame_duration_sec)

        interval_speech_frames = is_speech[start_frame:end_frame] if start_frame < len(is_speech) else np.array([])

        if len(interval_speech_frames) > 0:
            speech_ratio = float(np.mean(interval_speech_frames)) * 100.0
        else:
            speech_ratio = 0.0

        silence_ratio = max(0.0, 100.0 - speech_ratio)

        # Audio engagement scoring logic (0 to 100)
        # Optimal pedagogical discourse: 30% to 75% active speech (questions, answers, discussion) -> 80-95
        # Prolonged monologue: > 85% speech -> 60-70
        # Dead silence: > 80% silence -> 25-35
        if speech_ratio < 15.0:
            audio_score = 25.0 + (speech_ratio / 15.0) * 20.0
        elif speech_ratio <= 75.0:
            audio_score = 65.0 + ((speech_ratio - 15.0) / 60.0) * 30.0
        else:
            # High monologue or continuous speaking without pauses
            audio_score = max(50.0, 95.0 - ((speech_ratio - 75.0) / 25.0) * 30.0)

        interval_records.append({
            "interval_idx": i,
            "start_sec": round(i_start, 2),
            "end_sec": round(i_end, 2),
            "speech_ratio": round(speech_ratio, 1),
            "silence_ratio": round(silence_ratio, 1),
            "audio_score": round(audio_score, 1)
        })

    overall_speech_pct = float(np.mean(is_speech)) * 100.0 if len(is_speech) > 0 else 0.0

    return {
        "has_audio": True,
        "error_msg": None,
        "duration_sec": round(duration_sec, 2),
        "sample_rate": sample_rate,
        "overall_speech_ratio": round(overall_speech_pct, 1),
        "overall_silence_ratio": round(100.0 - overall_speech_pct, 1),
        "noise_floor": round(noise_floor, 4),
        "intervals": interval_records
    }


def generate_neutral_intervals(duration_sec: float, interval_sec: float = 5.0) -> List[Dict[str, Any]]:
    """Generates neutral fallback intervals when audio stream is absent."""
    num_intervals = max(1, int(math.ceil(duration_sec / interval_sec))) if duration_sec > 0 else 1
    return [
        {
            "interval_idx": i,
            "start_sec": round(i * interval_sec, 2),
            "end_sec": round(min(duration_sec, (i + 1) * interval_sec), 2),
            "speech_ratio": 50.0,
            "silence_ratio": 50.0,
            "audio_score": 50.0
        }
        for i in range(num_intervals)
    ]


def generate_synthetic_classroom_audio(
    duration_sec: float = 120.0,
    sr: int = 16000,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, int, str]:
    """
    Generates realistic synthetic classroom acoustic waveform for demonstration:
    - Ambient room acoustics (subtle pink/white ventilation noise)
    - Teacher lecture bursts (harmonic formant frequencies ~150Hz - 800Hz)
    - Interactive Q&A conversation bursts with pauses
    - A 20-second lull/silence period to trigger recap recommendations
    """
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # 1. Background classroom ambient hum
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, total_samples)

    # 2. Synthesize speech simulation waveform with varied envelopes
    speech = np.zeros(total_samples)

    # Time segments:
    # 0 to 40s: Steady lecture (high speech, short breath pauses)
    # 40 to 60s: Interactive Q&A (alternating speaker bursts)
    # 60 to 85s: Low engagement lull / independent silent task (silence)
    # 85 to 120s: Discussion & wrap up (healthy interactive rhythm)

    for i in range(total_samples):
        sec = t[i]
        in_speech_window = False

        if 2.0 <= sec <= 38.0:
            # Teacher speaking with brief pauses every 4 seconds
            if (sec % 4.0) < 3.2:
                in_speech_window = True
        elif 42.0 <= sec <= 60.0:
            # Q&A turns
            if (sec % 3.0) < 2.0:
                in_speech_window = True
        elif 60.0 < sec < 82.0:
            # Silent work / drop period (no speech)
            in_speech_window = False
        elif 82.0 <= sec <= 118.0:
            # Active recap & wrap up
            if (sec % 3.5) < 2.5:
                in_speech_window = True

        if in_speech_window:
            # Harmonic voice simulation (fundamental + formants)
            f0 = 180.0 + 20.0 * np.sin(2 * np.pi * 0.5 * sec)
            formant1 = 0.35 * np.sin(2 * np.pi * f0 * sec)
            formant2 = 0.20 * np.sin(2 * np.pi * (f0 * 2.2) * sec)
            formant3 = 0.12 * np.sin(2 * np.pi * (f0 * 3.8) * sec)
            speech[i] = formant1 + formant2 + formant3

    # Combine signal
    combined = (speech + noise).astype(np.float32)
    # Normalize to -0.9 to 0.9 peak
    max_amp = np.max(np.abs(combined))
    if max_amp > 0:
        combined = combined * (0.85 / max_amp)

    saved_path = output_path or "demo_classroom_audio.wav"
    sf.write(saved_path, combined, sr)

    return combined, sr, saved_path
