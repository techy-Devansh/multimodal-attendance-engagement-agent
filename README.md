# SkillWallet — Multimodal Attendance & Engagement Agent
**College Capstone Project | Department of Computer Science & Engineering**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg)](https://opencv.org/)
[![Privacy-Preserving](https://img.shields.io/badge/Privacy-Zero%20Biometrics-success.svg)](#privacy--ethical-ai-framework)

---

## 📌 Executive Summary & Abstract
Traditional classroom roll calls consume valuable instructional time, while passive lecture monitoring fails to capture student engagement or cognitive drop-offs. **SkillWallet** is an edge-computed, multimodal AI agent designed to automate classroom attendance verification and quantify student engagement dynamics without violating student privacy.

Operating purely on local hardware with zero paid APIs or cloud dependencies, SkillWallet fuses:
1. **Computer Vision (OpenCV)**: Multi-face detection, anonymous centroid tracking (`STU-01`, `STU-02`), sustained presence thresholding (default 5.0 seconds), head orientation/gaze proxy estimation, and temporal frame-differencing motion analysis.
2. **Acoustic Signal Processing (Librosa & SoundFile)**: Short-time energy (RMS), Zero-Crossing Rate (ZCR), dynamic noise-floor modeling, and Voice Activity Detection (VAD) for speech-to-silence ratio quantification.
3. **Multimodal Interval Fusion Engine**: Configurable weighted linear combination ($w_v \times \text{VideoScore} + w_a \times \text{AudioScore}$) producing an interpretable engagement index (0–100) per time window.
4. **Transparent Rule-Based AI Teaching Agent**: Deterministic reasoning engine mapping empirical anomalies (sustained drops, monologues, restlessness) to concrete pedagogical interventions and live classroom scripts.
5. **Relational Persistence & Automated Reporting**: SQLite database storage with instant CSV and ReportLab-compiled PDF report generation.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Input ["Classroom Inputs"]
        V[Classroom Video Stream\nMP4 / AVI / MOV]
        A[Classroom Audio Stream\nWAV / MP3 / OGG]
    end

    subgraph VisionPipeline ["Vision Processing (vision.py)"]
        V --> FD[CLAHE + Haar Cascade Face Detection]
        FD --> CT[Centroid Tracker\nAnonymous IDs: STU-01]
        CT --> SP[Sustained Presence Evaluator\n≥ 5.0s Threshold]
        FD --> HP[Head Orientation & Gaze Proxy\nEye Presence & Symmetry]
        V --> MD[Temporal Frame Differencing\nMotion Index & Inactivity]
        SP --> VINT[Interval Video Sub-Score 0-100]
        HP --> VINT
        MD --> VINT
    end

    subgraph AudioPipeline ["Audio Processing (audio.py)"]
        A --> PRE[16 kHz Mono Resampling]
        PRE --> RMS[Short-Time Energy RMS]
        PRE --> ZCR[Zero-Crossing Rate ZCR]
        RMS --> VAD[Dynamic Noise Floor & VAD]
        ZCR --> VAD
        VAD --> AINT[Interval Audio Sub-Score 0-100\nSpeech-to-Silence Ratio]
    end

    subgraph FusionEngine ["Multimodal Fusion (fusion.py)"]
        VINT --> FUS[Weighted Linear Interval Fusion\nw_v * Video + w_a * Audio]
        AINT --> FUS
        FUS --> EMA[Rolling Trend & Anomaly Tagger\nDrops < 40 | Peaks ≥ 78]
    end

    subgraph ReasoningAgent ["AI Pedagogical Agent (agent.py)"]
        EMA --> RULE[Deterministic Rule Engine\nRecap | Q&A | Restlessness | Pacing]
        RULE --> RECS[Actionable Teaching Strategies & Scripts]
    end

    subgraph StorageReports ["Persistence & Reporting"]
        SP --> DB[(SQLite Database\nskillwallet.db)]
        EMA --> DB
        RECS --> DB
        DB --> UI[Streamlit Interactive UI\napp.py]
        DB --> PDF[ReportLab Combined PDF Report]
        DB --> CSV[Attendance & Analytics CSVs]
    end
```

---

## 🌟 Key Capabilities & Features

### 1. Privacy-Preserving Attendance Verification
- **Zero Biometric Enrollment**: Never extracts or stores facial embeddings, facial geometry vectors, or student names.
- **Anonymous Tracking IDs**: Uses Euclidean distance centroid tracking to assign sequential tokens (`STU-01`, `STU-02`).
- **Configurable Sustained Presence Threshold**: Default 5.0 seconds (customizable from 1.0s to 30.0s). Distinguishes genuine attendees (`Present`) from passersby or doorway occlusions (`Partial / Fleeting`).
- **Temporal Metrics**: Accurately records first-seen timestamp (entry time), last-seen timestamp (exit time), and total visible seconds.
- **Manual Headcount Validator**: Allows the educator or proctor to input their physical roll call count and automatically calculates discrepancy ($\Delta$), detection coverage, and edge diagnostic feedback.

### 2. Multimodal Engagement Analytics
- **Video Proxies**:
  - *Face Visibility Ratio*: Active faces present relative to peak classroom capacity.
  - *Head Orientation Proxy*: Eye presence via Haar Cascade detection within facial upper ROI combined with aspect geometry (frontal forward gaze vs tilted downward or turned away).
  - *Motion Index*: Pixel-level absolute frame differencing ($|I_t - I_{t-1}|$) calibrated to detect healthy attentive motion vs extreme inactivity (freezing/sleeping) vs hyperactive restlessness.
- **Audio Proxies**:
  - *Short-Time Energy (RMS)* and *Zero-Crossing Rate (ZCR)*.
  - *Adaptive Voice Activity Detection (VAD)*: Dynamic noise-floor baseline separating vocal formants from room ventilation hum.
  - *Dialogue Dynamics*: Categorizes balanced discourse (35–75% speech) vs monologue (> 80% speech) vs dead silence (> 85% silence).
- **Multimodal Interval Fusion**:
  - Discretizes sessions into synchronized time windows (default 5.0 seconds).
  - Normalizes weights ($w_v + w_a = 1.0$) with dynamic UI sliders allowing real-time timeline recalculation without reprocessing video frames.
  - Automatic fallback to Video-Only mode ($w_v = 1.0, w_a = 0.0$) if an audio track is absent or unsupported.
- **Interactive Plotly Visualizations**:
  - Fused score timeline with 3-interval rolling trend line.
  - Shaded optimal engagement target band (55–85) and attention intervention zone (< 40).
  - Visual markers and tooltips for Engagement Drops and Interaction Peaks.
  - Subcomponent decomposition chart (Gaze vs Motion vs Speech vs Silence).

### 3. Rule-Based AI Teaching Agent
- Operates 100% deterministically without paid APIs or external LLM hallucination risks.
- Formulates structured pedagogical interventions with severity ratings (`HIGH`, `MEDIUM`, `INFO`), empirical trigger metrics, pedagogical rationales, recommended teaching strategies, and ready-to-speak classroom scripts:
  - `R-RECAP-01`: Concept check and recap prompt triggered by sustained engagement drops ($\ge 10\text{s}$).
  - `R-QA-02`: Think-Pair-Share and peer dialogue prompt triggered by extended teacher monologues.
  - `R-SILENCE-03`: Scaffolding hint prompt triggered by unplanned auditory silence.
  - `R-RESTLESS-04`: Pacing and physical micro-break prompt triggered by erratic motion and averted gaze.
  - `R-POSITIVE-05`: Instructional high-point reinforcement cataloging peak participatory windows.
  - `R-ATTENDANCE-06`: Sustained presence integrity alert flagging fleeting perimeter tracks.

### 4. Comprehensive Reporting & Persistence
- **SQLite Database (`skillwallet.db`)**: Normalized relational schema storing sessions, student attendance, engagement intervals, and AI recommendations.
- **ReportLab PDF Generator**: Compiles a publication-grade PDF report with session metadata, privacy notice, attendance table, multimodal metrics summary, and pedagogical recommendation cards.
- **Instant CSV Exports**: Export structured Attendance CSVs and Time-Series Analytics CSVs with a single click.
- **Pre-packaged Synthetic Demo Mode**: 1-click demonstration dataset simulating a Data Structures lecture with realistic attendance, engagement drop, and interaction peaks.

---

## 📂 Project Structure

```
skillwalet/
├── app.py                     # Streamlit application entry point
├── studio_app.py              # Classroom intelligence workspace UI
├── vision.py                  # Face detection, centroid tracking, gaze proxy, motion
├── audio.py                   # RMS energy, ZCR, VAD, and speech-silence dynamics
├── fusion.py                  # Multimodal interval fusion & Plotly visualization
├── agent.py                   # Rule-based transparent pedagogical AI agent
├── database.py                # SQLite persistence and schema management
├── utils.py                   # Time formatting, demo generator, ReportLab PDF builder
├── requirements.txt           # Project dependencies
├── install_dependencies.bat   # Windows dependency installation script
├── .gitignore                 # Git ignore rules for virtual environments and cache
└── README.md                  # Project documentation & capstone report
```

---

## 💻 Windows Setup & Running Instructions

### Prerequisites
- Windows 10 or 11 (64-bit)
- Python 3.14 (Installed and verified in PATH)
- A `.venv314` virtual environment in the project directory

### Step 1: Open Terminal in Workspace
Open PowerShell or Command Prompt in the workspace directory:
```powershell
cd c:\Users\Devansh\Desktop\skillwalet
```

### Step 2: Create and activate the Python 3.14 virtual environment
```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*(You will see `(.venv314)` in your PowerShell prompt.)*

### Step 3: Run the Streamlit Application
```powershell
python -m streamlit run app.py
```

### Step 4: Access the Web Interface
Your default browser will automatically open:
```
Local URL: http://localhost:8501
```

---

## 🧪 Testing with the Built-in Demo Dataset
To evaluate the complete multimodal capabilities without needing a pre-recorded classroom video:
1. Launch the app using `streamlit run app.py`.
2. In the left sidebar, click **"🚀 Load Sample Classroom Session"**.
3. The system will synthesize a complete 3-minute classroom session with 12 tracked students, synchronized interval engagement curves, flagged drops, and pedagogical recommendations.
4. Explore the seven workspace views (**Welcome**, **Dashboard**, **Live classroom studio**, **Attendance logs**, **Session dynamics**, **Student cohorts**, and **Architecture & specs**). The demo is labeled as synthetic; adjust weights and export attendance, interval, and PDF reports.

The Welcome view is the first screen. **Launch workspace** opens the dashboard; **Load sample classroom** creates a labeled synthetic session in the local database. Use **Live classroom studio** to process a real recording with the existing local vision, audio, fusion, and teaching-agent modules.

---

## 🔒 Privacy & Ethical AI Framework

SkillWallet was architected strictly under **Privacy by Design** principles:

| Privacy Principle | Implementation in SkillWallet |
| :--- | :--- |
| **No Biometric Identification** | Does not extract 128-d face embeddings, face templates, or associate names. Attendees are represented purely by random spatial tokens (e.g. `STU-01`). |
| **Ephemeral Memory Processing** | Video frames are analyzed in RAM buffers and immediately released. Raw video frames are never stored in the database. |
| **100% Local & Edge Execution** | All computer vision, signal processing, and rule reasoning execute locally on the host CPU. Zero network transmission. |
| **Proxy Metric Transparency** | Engagement scores are explicitly documented as **behavioral proxies** (visible face presence, gaze proxy, acoustic activity). They **do not claim to measure internal cognitive attention, mental comprehension, or aptitude**. |
| **Confirmed Data Erasure** | The system includes a confirmed "Delete All Data" control in the Privacy page to purge all database records. |

---

## ⚠️ Known Technical Limitations

1. **Camera Angle & Resolution**: Single wide-angle classroom cameras may suffer from perspective distortion or face scale reduction for students seated in far back rows (< 35px face width).
2. **Classroom Lighting Variations**: Severe backlighting (e.g., windows behind students) can degrade Haar cascade contrast. The pipeline mitigates this using adaptive histogram equalization (CLAHE).
3. **Acoustic Far-Field Reverberation**: Single laptop microphones capture room reverberation and echo. Dedicated lavalier or boundary microphones improve speech-to-silence precision.
4. **Proxy Behavioral Assumption**: A student looking downwards may be actively taking handwritten notes or reading rather than being disengaged. The pedagogical agent treats these as gentle prompts for check-ins rather than punitive evaluations.

---

## 🚀 Deployment & Future Work
- **Containerization**: Deployable via Docker with standard Python 3.12 slim images.
- **Embedded Edge Hardware**: Can run on NVIDIA Jetson Orin Nano or Raspberry Pi 5 with an overhead wide-angle USB/CSI camera.
- **RTSP IP Camera Ingestion**: Direct integration with existing campus lecture capture cameras over RTSP/HLS protocols.
- **Multi-Camera Stitching**: Multi-camera panoramic homography fusion to eliminate blind spots in lecture halls with > 100 students.

---

## 📜 Academic Attribution
- **Capstone Title:** SkillWallet — Multimodal Attendance & Engagement Agent
- **Domain:** Multimodal Human-Computer Interaction & Educational Data Mining
- **Student field:** B.Tech / B.E. Computer Science & Engineering (CSE)
