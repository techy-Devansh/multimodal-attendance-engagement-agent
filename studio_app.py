"""Premium Streamlit workspace for the SkillWallet classroom intelligence system."""

import datetime
import html
import os
import tempfile
import time
import uuid

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import agent
import audio
import database
import fusion
import utils
import vision


PAGES = [
    "Welcome",
    "Dashboard",
    "Live classroom studio",
    "Attendance logs",
    "Session dynamics",
    "Student cohorts",
    "Architecture & specs",
]

st.set_page_config(
    page_title="SkillWallet | Classroom intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)
database.init_db()


def safe(value):
    return html.escape(str(value if value is not None else ""))


def go_to(page):
    st.session_state["requested_route"] = page
    st.rerun()


if st.session_state.get("requested_route") in PAGES:
    st.session_state["nav_route"] = st.session_state.pop("requested_route")

all_sessions = database.get_all_sessions()
known_ids = {row["session_id"] for row in all_sessions}
if st.session_state.get("active_session_selector") not in known_ids | {"No session"}:
    st.session_state["active_session_selector"] = (
        all_sessions[0]["session_id"] if all_sessions else "No session"
    )


st.markdown(
    """
<style>
:root {
  --ink:#e8ede9; --muted:#9ba59f; --subtle:#66716b; --line:#2b332f;
  --base:#0b0e0d; --panel:#111614; --panel2:#151b18; --teal:#80cdb0;
  --lime:#b6c99a; --amber:#cda16e; --red:#c77d72;
}
html, body, [class*="css"] { font-family:'Segoe UI',system-ui,sans-serif; }
:root { color-scheme:dark; }
.stApp { background:var(--base); color:var(--ink); }
[data-testid="stHeader"] { background:#0b0e0d; }
[data-testid="stSidebar"] { background:#0e1210; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.15rem; }
[data-testid="stMain"] { background:var(--base); }
.block-container { max-width:1480px; padding:1.65rem 3rem 4rem; }
h1,h2,h3,h4 { font-family:'Segoe UI',system-ui,sans-serif; letter-spacing:-.035em; color:var(--ink); }
h1 { font-size:clamp(2.5rem,5.2vw,4.8rem)!important; line-height:1.03!important; font-weight:600!important; }
h2 { font-size:2rem!important; font-weight:600!important; }
h3 { font-size:1.18rem!important; font-weight:600!important; }
p, li { color:#adb6b0; }
code { color:var(--teal); }
.brand-mark { width:32px;height:32px;display:inline-grid;place-items:center;border:1px solid #496657;
  color:var(--teal);border-radius:7px;font-size:17px;background:#17201b; }
.brand-lockup { display:flex;align-items:center;gap:10px;margin:0 0 5px; }
.brand-name { font:650 16px 'Segoe UI',system-ui,sans-serif;letter-spacing:-.035em;color:#e8ede9; }
.brand-caption,.eyebrow,.mono { font-family:Consolas,'Courier New',monospace; }
.brand-caption { color:#7f8b84;font-size:9px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 22px 42px; }
.eyebrow { color:var(--teal);font-size:9px;letter-spacing:.16em;text-transform:uppercase; }
.page-intro { color:#a3ada6;font-size:14px;max-width:760px;line-height:1.7;margin:.65rem 0 1.35rem; }
.topline { display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);
  padding:0 0 14px;margin:0 0 26px;color:#87928b;font:9px Consolas,'Courier New',monospace;letter-spacing:.1em;text-transform:uppercase; }
.source-pill { display:inline-flex;align-items:center;gap:8px;border:1px solid #38433d;border-radius:4px;
  padding:6px 9px;color:#b1bbb4;font:9px Consolas,'Courier New',monospace;letter-spacing:.05em;text-transform:uppercase; }
.source-dot { width:6px;height:6px;border-radius:50%;background:var(--lime);display:inline-block; }
.source-demo { border-color:#67523b;color:#d7ba92; }
.source-demo .source-dot { background:var(--amber); }
.metric { background:#111614;border:1px solid #2a332e;border-radius:8px;
  padding:19px 20px;min-height:128px;position:relative;overflow:hidden; }
.metric:after { content:'';position:absolute;left:0;top:0;width:28px;height:1px;background:var(--teal); }
.metric-label { color:#929d96;font:9px Consolas,'Courier New',monospace;letter-spacing:.09em;text-transform:uppercase; }
.metric-value { color:#e8ede9;font:600 31px 'Segoe UI',system-ui,sans-serif;letter-spacing:-.055em;margin:13px 0 3px; }
.metric-note { color:#87928b;font-size:11px; }
.surface { background:#111614;border:1px solid #29322d;border-radius:9px;padding:20px 21px; }
.surface-title { font:600 15px 'Segoe UI',system-ui,sans-serif;color:#e6ebe7;letter-spacing:-.025em; }
.surface-note { color:#87928b;font:9px Consolas,'Courier New',monospace;letter-spacing:.08em;text-transform:uppercase;margin-top:5px; }
.hero { position:relative;overflow:hidden;border:1px solid #303a34;border-radius:10px;padding:clamp(30px,5vw,68px);
  background:#111614; }
.hero:before { content:'';position:absolute;inset:0 0 auto;height:2px;background:linear-gradient(90deg,var(--teal),transparent 42%);opacity:.72; }
.hero-copy { position:relative;z-index:2;max-width:740px; }
.hero-copy h1 [data-heading-text] { color:var(--ink)!important; }
.hero-copy h1 [data-heading-text] > span { color:var(--teal)!important; }
.hero-sub { color:#a5afa8;font-size:15px;line-height:1.8;max-width:620px;margin:19px 0 25px; }
.hero-meta { color:#8d9991;font:9px Consolas,'Courier New',monospace;letter-spacing:.13em;text-transform:uppercase;margin-bottom:18px; }
.product-preview { background:#0d110f;border:1px solid #303a34;border-radius:8px;overflow:hidden; }
.preview-head { display:flex;justify-content:space-between;align-items:center;padding:12px 15px;border-bottom:1px solid #29322d;color:#99a49d;font:9px Consolas,'Courier New',monospace;letter-spacing:.08em;text-transform:uppercase; }
.preview-lights { display:flex;gap:5px; }
.preview-lights i { width:6px;height:6px;border-radius:50%;background:#57635b; }
.preview-body { padding:18px 17px 16px; }
.preview-kicker { color:#8d9991;font:9px Consolas,'Courier New',monospace;letter-spacing:.1em;text-transform:uppercase; }
.preview-value { font:600 34px 'Segoe UI',system-ui,sans-serif;letter-spacing:-.06em;color:#e7ede8;margin-top:8px; }
.preview-rule { height:1px;background:#29322d;margin:16px 0 13px; }
.preview-state { color:#9da8a0;font-size:11px;line-height:1.65; }
.preview-state strong { color:#dce5de;font-weight:600; }
.preview-empty { min-height:270px;display:flex;flex-direction:column;justify-content:space-between; }
.preview-empty .preview-value { font-size:24px;line-height:1.3;max-width:240px; }
.preview-foot { color:#849087;font:9px Consolas,'Courier New',monospace;letter-spacing:.05em;text-transform:uppercase; }
.hero-meta-rule { width:42px;height:1px;background:var(--teal);margin:19px 0; }
.data-label { color:#849087;font:9px Consolas,'Courier New',monospace;letter-spacing:.07em; }
.section-head { display:flex;align-items:end;justify-content:space-between;margin:27px 0 13px;gap:12px; }
.section-head h3 { margin:0; }
.event-card { border-left:2px solid var(--amber);background:#151a17;padding:13px 15px;border-radius:0 5px 5px 0;margin:9px 0; }
.event-card.good { border-color:var(--teal); }
.event-title { color:#e1e8e2;font-size:12px;font-weight:600; }
.event-copy { color:#a0aba3;font-size:11px;line-height:1.6;margin-top:5px; }
.cohort { background:#121714;border:1px solid #303a34;border-radius:8px;padding:17px; }
.cohort-num { color:var(--teal);font:600 27px 'Segoe UI',system-ui,sans-serif; }
.cohort-name { color:#d5ddd7;font-size:12px;margin:6px 0; }
.cohort-rule { color:#8d9991;font:9px Consolas,'Courier New',monospace;text-transform:uppercase;letter-spacing:.07em; }
.flow-node { text-align:center;background:#121714;border:1px solid #303a34;border-radius:5px;padding:13px 10px;color:#d0d9d2;font:10px Consolas,'Courier New',monospace;letter-spacing:.025em; }
.flow-arrow { text-align:center;color:#7eaf96;padding:5px 0;font:13px Consolas,'Courier New',monospace; }
button[data-testid^="stBaseButton-"] { min-height:39px;border-radius:5px!important;border:1px solid #3a4840!important;background:#17201b!important;color:#e0e8e1!important;font-weight:600;transition:all .16s ease; }
button[data-testid^="stBaseButton-"]:hover { border-color:var(--teal)!important;color:#c8f1df!important;background:#1a2620!important; }
button[data-testid="stBaseButton-primary"] { background:#80cdb0!important;border-color:#80cdb0!important;color:#0c1410!important; }
button[data-testid="stBaseButton-primary"]:hover { background:#9addc2!important;color:#0a110d!important; }
.stTextInput input,.stSelectbox [data-baseweb="select"],.stNumberInput input { background:#111714;color:#e7ece8;border-color:#354038; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--teal); }
[data-testid="stDataFrame"] { border:1px solid #2e3732;border-radius:6px;overflow:hidden; }
[data-testid="stMetric"] { background:#111614;border:1px solid #303a34;padding:14px;border-radius:6px; }
.stTabs [data-baseweb="tab-list"] { gap:18px;border-bottom:1px solid #303a34; }
.stTabs [data-baseweb="tab"] { color:#9aa59e; }
.stTabs [aria-selected="true"] { color:#9cdbbc!important; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] { border-radius:4px;padding:4px 5px;border-left:2px solid transparent; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background:#171e1a; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] { background:#18221c;border-left-color:var(--teal); }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p { color:#b9e2ca; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] input { accent-color:var(--teal); }
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] { border-radius:5px; }
hr { border-color:#2b332f!important; }
@media(max-width:1000px) { .block-container{padding:1.4rem 1.5rem 3rem}.hero{padding:34px 28px} }
@media(max-width:700px) { .block-container{padding:1.15rem 1rem 2.5rem}h1{font-size:2.65rem!important}.metric{padding:15px;min-height:112px}.metric-value{font-size:26px}.hero-sub{font-size:14px}.topline{font-size:8px}.source-pill{font-size:8px} }
@media(prefers-reduced-motion:reduce) { *,*:before,*:after{animation-duration:.01ms!important;transition-duration:.01ms!important} }

/* Midnight signal design system */
:root {
  --ink:#f0f3ff; --muted:#a2acc4; --subtle:#737f9c; --line:rgba(151,174,226,.15);
  --base:#080b16; --panel:rgba(15,21,39,.84); --panel2:rgba(19,27,50,.78);
  --teal:#5fe3f2; --lime:#9d8aff; --amber:#f0bd78; --red:#fa8296;
  --cyan:#5fe3f2; --violet:#a88bff; --blue:#6d9cff;
}
:root { color-scheme:dark; }
html,body,[class*="css"] { font-family:"Segoe UI",Inter,system-ui,sans-serif; }
body { background:#080b16; }
.stApp {
  color:var(--ink);
  background:
    radial-gradient(ellipse at 87% 2%,rgba(82,80,209,.17),transparent 28%),
    radial-gradient(ellipse at 8% 30%,rgba(0,181,215,.075),transparent 30%),
    linear-gradient(145deg,#080b16 0%,#0b1020 53%,#090d19 100%);
}
[data-testid="stHeader"] { background:rgba(8,11,22,.86);backdrop-filter:blur(16px); }
[data-testid="stMain"] { background:transparent; }
.block-container { max-width:1510px;padding:1.45rem 3.1rem 4rem; }
h1,h2,h3,h4 { color:#f0f3ff;letter-spacing:-.04em; }
h1 { font-size:clamp(2.8rem,5.7vw,5.15rem)!important;line-height:.99!important;font-weight:650!important; }
h2 { font-size:2.05rem!important; }
h3 { font-size:1.2rem!important; }
p,li { color:#b0bad0; }
code { color:var(--cyan); }
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,rgba(13,19,37,.98),rgba(9,13,26,.99));
  border-right:1px solid rgba(131,157,218,.18);
  box-shadow:14px 0 46px rgba(0,0,0,.2);
}
[data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
.brand-mark { width:35px;height:35px;border-radius:10px;border:1px solid rgba(95,227,242,.5);
  color:#baf7ff;background:linear-gradient(145deg,rgba(39,194,228,.22),rgba(129,100,255,.16));
  box-shadow:0 0 26px rgba(60,194,236,.18); }
.brand-name { color:#eff4ff;font-size:17px; }
.brand-caption { color:#7887aa; }
.eyebrow { color:#7deafa;letter-spacing:.18em; }
.topline { border-color:rgba(151,174,226,.14);color:#8290b0; }
.page-intro { color:#aab5cd; }
.source-pill { border-radius:999px;background:rgba(95,227,242,.06);border-color:rgba(95,227,242,.27);color:#a7eefa; }
.source-demo { background:rgba(240,189,120,.08);border-color:rgba(240,189,120,.36);color:#f2c995; }
.source-demo .source-dot { background:var(--amber);box-shadow:0 0 10px rgba(240,189,120,.55); }
.source-dot { box-shadow:0 0 10px rgba(95,227,242,.55); }
.metric {
  min-height:136px;padding:20px 21px;border-radius:13px;
  border:1px solid rgba(148,174,231,.19);
  background:linear-gradient(145deg,rgba(23,33,60,.88),rgba(13,19,36,.85));
  box-shadow:0 15px 32px rgba(1,4,14,.18),inset 0 1px 0 rgba(255,255,255,.035);
  animation:card-arrive .55s cubic-bezier(.2,.8,.2,1) both;
  transition:transform .24s ease,border-color .24s ease,box-shadow .24s ease;
}
.metric:nth-child(2) { animation-delay:.06s; }
.metric:nth-child(3) { animation-delay:.12s; }
.metric:nth-child(4) { animation-delay:.18s; }
.metric:after { left:20px;top:0;width:58px;height:2px;background:linear-gradient(90deg,var(--cyan),rgba(168,139,255,.35)); }
.metric:hover { transform:translateY(-4px);border-color:rgba(95,227,242,.42);box-shadow:0 18px 38px rgba(2,7,23,.34),0 0 24px rgba(65,180,236,.08); }
.metric-label { color:#95a4c4; }
.metric-value { color:#f2f5ff;font-size:34px; }
.metric-note { color:#92a0bd; }
.surface,.cohort,.flow-node {
  background:linear-gradient(145deg,rgba(20,29,53,.81),rgba(12,18,35,.82));
  border:1px solid rgba(145,169,224,.17);border-radius:12px;
  box-shadow:0 16px 38px rgba(2,5,16,.19),inset 0 1px 0 rgba(255,255,255,.035);
  backdrop-filter:blur(12px);
  transition:transform .24s ease,border-color .24s ease,box-shadow .24s ease;
}
.surface { padding:22px 23px; }
.surface:hover,.cohort:hover { transform:translateY(-3px);border-color:rgba(113,200,241,.34);box-shadow:0 20px 42px rgba(2,5,16,.29),0 0 22px rgba(112,126,255,.07); }
.surface-title { color:#edf2ff;font-size:16px; }
.surface-note { color:#8392b2; }
.section-head { margin:32px 0 15px; }
.section-head h3 { color:#eef3ff; }
.event-card { background:rgba(15,22,42,.78);border-left:2px solid var(--violet);border-radius:2px 9px 9px 2px;padding:14px 16px; }
.event-card.good { border-color:var(--cyan); }
.event-title { color:#e4eaff; }
.event-copy { color:#a3afca; }
.cohort { padding:19px; }
.cohort-num { color:#74e9f4; }
.cohort-rule { color:#8795b5; }
.flow-node { color:#d5e0ff;border-radius:8px;border-color:rgba(135,154,220,.25); }
.flow-arrow { color:#8d9bff; }
.hero { background:transparent;border:0;padding:0; }
.hero:before { display:none; }
.hero-copy h1 [data-heading-text] { color:#f2f5ff!important; }
.hero-copy h1 [data-heading-text] > span { color:var(--cyan)!important; }
.hero-composition {
  position:relative;isolation:isolate;display:grid;grid-template-columns:minmax(0,1.08fr) minmax(360px,.92fr);
  align-items:center;gap:clamp(28px,5vw,76px);min-height:440px;margin:12px 0 19px;padding:clamp(28px,5vw,65px);
  overflow:hidden;border:1px solid rgba(146,171,231,.19);border-radius:20px;
  background:radial-gradient(ellipse at 82% 45%,rgba(83,94,213,.13),transparent 37%),linear-gradient(120deg,rgba(17,25,48,.94),rgba(11,17,34,.91) 58%,rgba(15,18,42,.9));
  box-shadow:0 28px 80px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.045);
}
.hero-composition:before { content:"";position:absolute;inset:0;z-index:-1;opacity:.34;pointer-events:none;
  background-image:linear-gradient(rgba(139,160,220,.075) 1px,transparent 1px),linear-gradient(90deg,rgba(139,160,220,.075) 1px,transparent 1px);
  background-size:42px 42px;mask-image:linear-gradient(90deg,transparent 5%,black 100%); }
.hero-composition:after { content:"";position:absolute;z-index:-1;width:500px;height:500px;top:-320px;left:45%;
  border-radius:50%;border:1px solid rgba(111,144,234,.09);box-shadow:0 0 0 38px rgba(111,144,234,.025),0 0 0 78px rgba(111,144,234,.02); }
.hero-copy { position:relative;z-index:2;max-width:740px; }
.hero-copy h1 { margin:.6rem 0 1.15rem; }
.hero-copy h1 span { color:var(--cyan);background:linear-gradient(95deg,#66e5f1,#a5a0ff 88%);background-clip:text;-webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.hero-meta { display:flex;align-items:center;gap:9px;color:#99aacb;font-size:10px; }
.hero-live-dot { width:7px;height:7px;border-radius:50%;background:#61e1ef;box-shadow:0 0 14px rgba(95,227,242,.85); }
.hero-sub { font-size:15px;max-width:580px;line-height:1.75;margin:0 0 25px; }
.hero-proof { display:flex;align-items:center;gap:11px;color:#8594b5;font:9px Consolas,'Courier New',monospace;letter-spacing:.08em; }
.hero-proof i { width:3px;height:3px;border-radius:50%;background:#a58cff;box-shadow:0 0 8px rgba(168,139,255,.6); }
.hero-visual { position:relative;z-index:1;min-width:0; }
.hero-data-card { min-height:310px; }
.hero-data-card .preview-body { padding:21px 22px 19px; }
.preview-source { color:#94a5c8;font:8px Consolas,'Courier New',monospace;letter-spacing:.08em; }
.session-name { overflow:hidden;color:#f0f4ff;font-size:19px;font-weight:600;letter-spacing:-.03em;text-overflow:ellipsis;white-space:nowrap;margin:0 0 17px; }
.signal-overview { display:flex;align-items:center;gap:21px;padding-bottom:15px;border-bottom:1px solid rgba(151,174,226,.13); }
.score-ring {
  width:98px;height:98px;flex:0 0 98px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:radial-gradient(circle at center,#111a32 58%,transparent 60%),conic-gradient(from 220deg,#5fe3f2 var(--score-progress,0%),#8b7bff var(--score-progress,0%),rgba(131,151,204,.16) var(--score-progress,0%));
  box-shadow:0 0 28px rgba(80,164,235,.1);
}
.score-ring-label { color:#8d9fbe;font:7px Consolas,'Courier New',monospace;letter-spacing:.08em; }
.score-ring strong { color:#f0f5ff;font-size:26px;line-height:1.1;margin-top:3px;letter-spacing:-.06em; }
.score-ring small { color:#8d9fbe;font:8px Consolas,'Courier New',monospace; }
.signal-stats { flex:1;display:grid;gap:13px; }
.signal-stats div { display:flex;flex-direction:column;gap:4px; }
.signal-stats span,.signal-caption { color:#8696b6;font:8px Consolas,'Courier New',monospace;letter-spacing:.08em; }
.signal-stats strong { color:#e7edff;font-size:16px;font-weight:600; }
.session-signal { display:block;width:100%;height:78px;margin:14px 0 0;overflow:visible; }
.signal-area { fill:rgba(95,227,242,.1); }
.signal-line { fill:none;stroke:#71e4f0;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px rgba(95,227,242,.4)); }
.signal-caption { display:flex;justify-content:space-between;margin:1px 0 14px; }
.signal-empty { height:92px;display:grid;place-items:center;color:#8e9cba;font-size:11px; }
.hero-data-card .preview-foot { border-top:1px solid rgba(151,174,226,.13);padding-top:12px;margin-top:2px; }
.visual-orbit { position:absolute;z-index:-1;border:1px solid rgba(127,140,246,.11);border-radius:50%;pointer-events:none;animation:ambient-drift 12s ease-in-out infinite; }
.orbit-one { width:360px;height:360px;right:-170px;top:-185px; }
.orbit-two { width:260px;height:260px;right:-120px;top:-135px;border-color:rgba(95,227,242,.12);animation-delay:-4s; }
.no-session .preview-body { display:flex;flex-direction:column;min-height:275px; }
.empty-signal-heading { color:#eff3ff;font-size:23px;line-height:1.25;font-weight:600;letter-spacing:-.035em;margin:2px 0 20px; }
.empty-signal-heading span { color:#aebbd7;font-size:17px;font-weight:400; }
.pipeline-visual { display:flex;align-items:center;margin:3px 0 19px; }
.pipeline-node { min-width:85px;display:grid;gap:5px;padding:10px 11px;border:1px solid rgba(95,227,242,.23);border-radius:9px;background:rgba(26,45,68,.44); }
.pipeline-node i { color:#6fe3ef;font:8px Consolas,'Courier New',monospace;font-style:normal; }
.pipeline-node b { color:#eaf1ff;font-size:12px;font-weight:600; }
.pipeline-node small { color:#8f9db9;font:6px Consolas,'Courier New',monospace;white-space:nowrap; }
.pipeline-node.violet { border-color:rgba(168,139,255,.27);background:rgba(42,34,72,.4); }
.pipeline-node.violet i { color:#b39cff; }
.pipeline-link { height:1px;flex:1;min-width:10px;background:linear-gradient(90deg,rgba(95,227,242,.55),rgba(168,139,255,.65));position:relative; }
.pipeline-link:after { content:"";position:absolute;width:4px;height:4px;right:0;top:-2px;border-radius:50%;background:#b49aff;box-shadow:0 0 8px #9b83ff; }
.hero-sub { color:#aeb9d1; }
.hero-meta { color:#98a7c7; }
.hero-meta-rule { background:linear-gradient(90deg,var(--cyan),var(--violet));height:2px; }
.product-preview {
  position:relative;isolation:isolate;overflow:hidden;
  border:1px solid rgba(140,167,235,.25);border-radius:16px;
  background:linear-gradient(145deg,rgba(19,29,55,.92),rgba(10,16,32,.95));
  box-shadow:0 25px 65px rgba(0,0,0,.32),0 0 48px rgba(86,97,225,.12),inset 0 1px 0 rgba(255,255,255,.06);
  backdrop-filter:blur(18px);transition:transform .3s ease,border-color .3s ease,box-shadow .3s ease;
}
.product-preview:hover { transform:translateY(-5px);border-color:rgba(95,227,242,.46);box-shadow:0 30px 70px rgba(0,0,0,.38),0 0 55px rgba(86,97,225,.17); }
.product-preview:before { content:"";position:absolute;z-index:-1;width:245px;height:245px;right:-110px;top:-115px;
  border-radius:50%;background:radial-gradient(circle,rgba(116,88,255,.36),rgba(116,88,255,0) 69%);filter:blur(3px); }
.product-preview:after { content:"";position:absolute;z-index:-1;left:-125px;bottom:-170px;width:280px;height:280px;
  border-radius:50%;background:radial-gradient(circle,rgba(30,194,225,.18),transparent 69%); }
.preview-head { padding:13px 17px;border-color:rgba(145,169,224,.14);color:#91a1c4; }
.preview-lights i:nth-child(1) { background:#62e1e9;box-shadow:0 0 8px rgba(98,225,233,.6); }
.preview-lights i:nth-child(2) { background:#9b83ff; }
.preview-lights i:nth-child(3) { background:#51617e; }
.preview-body { padding:24px 23px 20px; }
.preview-kicker { color:#9d8bff; }
.preview-value { color:#f1f5ff; }
.preview-rule { background:rgba(151,174,226,.16); }
.preview-state { color:#aab6d0; }
.preview-state strong { color:#e3eaff; }
.preview-foot,.data-label { color:#8392b2; }
.view-heading {
  position:relative;overflow:hidden;margin:0 0 18px;padding:20px 23px 19px 25px;
  border:1px solid rgba(144,168,225,.16);border-radius:12px;
  background:linear-gradient(110deg,rgba(17,26,49,.92),rgba(13,19,37,.67));
  box-shadow:0 12px 32px rgba(0,0,0,.14),inset 0 1px 0 rgba(255,255,255,.035);
}
.view-heading:before { content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,var(--view-accent,#61dff0),var(--view-accent-2,#9b83ff)); }
.view-heading:after { content:"";position:absolute;right:-90px;top:-145px;width:270px;height:270px;border-radius:50%;
  background:radial-gradient(circle,var(--view-glow,rgba(88,185,255,.14)),transparent 68%);pointer-events:none; }
.view-heading.dashboard { --view-accent:#61e1f1;--view-accent-2:#9a81ff;--view-glow:rgba(87,153,255,.18); }
.view-heading.studio { --view-accent:#5fe3f2;--view-accent-2:#58aaff;--view-glow:rgba(45,191,232,.16); }
.view-heading.attendance { --view-accent:#ac8bff;--view-accent-2:#e184fa;--view-glow:rgba(147,92,245,.17); }
.view-heading.dynamics { --view-accent:#5fe3f2;--view-accent-2:#a88bff;--view-glow:rgba(78,106,244,.18); }
.view-heading.cohorts { --view-accent:#65e8d1;--view-accent-2:#6eaeff;--view-glow:rgba(40,200,183,.13); }
.view-heading.architecture { --view-accent:#75b9ff;--view-accent-2:#a68aff;--view-glow:rgba(72,119,255,.16); }
.view-heading .view-index { color:#8191b2;font:9px Consolas,'Courier New',monospace;letter-spacing:.12em;text-transform:uppercase; }
.view-heading h1 { font-size:clamp(1.9rem,3vw,2.7rem)!important;line-height:1.08!important;margin:.55rem 0 .4rem; }
.view-heading .page-intro { font-size:13px;margin:0;max-width:780px;color:#aab6d0; }
.view-heading .eyebrow { color:var(--view-accent,#73e2f1); }
button[data-testid^="stBaseButton-"] {
  min-height:42px;border-radius:8px!important;border:1px solid rgba(136,164,224,.28)!important;
  background:linear-gradient(135deg,rgba(27,39,70,.95),rgba(19,27,50,.95))!important;
  color:#e4ecff!important;font-weight:600;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease,background .2s ease;
}
button[data-testid^="stBaseButton-"]:hover { transform:translateY(-2px);border-color:rgba(95,227,242,.7)!important;
  color:#d8fbff!important;background:linear-gradient(135deg,rgba(33,59,86,.98),rgba(30,38,77,.98))!important;
  box-shadow:0 8px 24px rgba(44,162,210,.17); }
button[data-testid="stBaseButton-primary"] {
  position:relative;overflow:hidden;background:linear-gradient(105deg,#53d8ed,#8c92ff 55%,#a984f5)!important;
  border:0!important;color:#091322!important;box-shadow:0 8px 28px rgba(87,174,237,.25);
}
button[data-testid="stBaseButton-primary"]:hover { color:#06111b!important;box-shadow:0 10px 34px rgba(100,185,248,.4); }
.stTextInput input,.stSelectbox [data-baseweb="select"],.stNumberInput input {
  background:rgba(13,20,39,.92);color:#eaf0ff;border-color:rgba(133,157,214,.24);border-radius:7px;
}
.stTextInput input:focus,.stNumberInput input:focus { border-color:var(--cyan);box-shadow:0 0 0 1px rgba(95,227,242,.25); }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--cyan);box-shadow:0 0 12px rgba(95,227,242,.55); }
[data-testid="stDataFrame"] { border:1px solid rgba(139,162,220,.2);border-radius:9px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.16); }
[data-testid="stMetric"] { background:rgba(17,25,48,.85);border:1px solid rgba(139,162,220,.2);padding:15px;border-radius:9px; }
.stTabs [data-baseweb="tab-list"] { gap:16px;border-bottom:1px solid rgba(139,162,220,.2); }
.stTabs [data-baseweb="tab"] { color:#9aa8c7;transition:color .18s ease; }
.stTabs [aria-selected="true"] { color:#82e8f4!important; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
  position:relative;border-radius:7px;padding:6px 8px;border:1px solid transparent;
  transition:background .18s ease,border-color .18s ease,transform .18s ease;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {
  background:rgba(74,101,161,.14);border-color:rgba(133,160,222,.14);transform:translateX(2px);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
  background:linear-gradient(100deg,rgba(57,180,215,.16),rgba(125,99,228,.13));
  border-color:rgba(95,227,242,.24);box-shadow:inset 0 0 20px rgba(58,148,210,.05);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"]:before {
  content:"";position:absolute;left:-1px;top:7px;bottom:7px;width:2px;border-radius:2px;
  background:linear-gradient(180deg,var(--cyan),var(--violet));box-shadow:0 0 12px rgba(95,227,242,.5);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p { color:#ddfaff; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {
  background-color:var(--cyan)!important;border-color:var(--cyan)!important;box-shadow:0 0 12px rgba(95,227,242,.5)!important;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div {
  background-color:#0b1224!important;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] input { accent-color:var(--cyan); }
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] { border-radius:7px; }
.stPlotlyChart { border:1px solid rgba(139,162,220,.16);border-radius:11px;background:rgba(12,18,35,.65);overflow:hidden; }
.stProgress > div > div { background:linear-gradient(90deg,var(--cyan),var(--violet)); }
hr { border-color:rgba(139,162,220,.16)!important; }
@keyframes card-arrive { from { opacity:0;transform:translateY(12px) } to { opacity:1;transform:translateY(0) } }
@keyframes ambient-drift { 0%,100% { transform:translate3d(0,0,0) } 50% { transform:translate3d(-10px,8px,0) } }
@keyframes signal-scan { 0% { opacity:.2;transform:translateY(-8px) } 50% { opacity:.7 } 100% { opacity:.2;transform:translateY(8px) } }
@media(max-width:1000px) { .block-container{padding:1.3rem 1.5rem 3rem}.view-heading{padding:19px 20px}.hero{padding:28px 24px} }
@media(max-width:1200px) { .hero-composition{grid-template-columns:1fr;gap:26px;min-height:unset}.hero-copy{max-width:780px}.hero-visual{max-width:620px;width:100%;margin:0 auto}.hero-data-card{min-height:0} }
@media(max-width:900px) { .hero-composition{grid-template-columns:1fr;gap:28px;min-height:unset}.hero-copy{max-width:760px}.hero-visual{max-width:600px;width:100%;margin:0 auto}.hero-data-card{min-height:0} }
@media(max-width:700px) { .block-container{padding:1.05rem .9rem 2.5rem}h1{font-size:2.65rem!important}.metric{padding:15px;min-height:112px}.metric-value{font-size:27px}.hero-sub{font-size:14px}.topline{font-size:8px}.source-pill{font-size:8px}.view-heading{padding:17px 17px 17px 20px}.view-heading h1{font-size:2rem!important}.product-preview{margin-top:5px}.hero-composition{padding:25px 19px;border-radius:15px}.hero-copy h1{font-size:2.75rem!important}.hero-proof{flex-wrap:wrap;font-size:8px}.signal-overview{gap:14px}.score-ring{width:84px;height:84px;flex-basis:84px}.pipeline-node{min-width:0;padding:9px 7px}.pipeline-node small{font-size:5px}.empty-signal-heading{font-size:21px} }
@media(prefers-reduced-motion:reduce) { *,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important} }
</style>
""",
    unsafe_allow_html=True,
)


def metric(label, value, note):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{safe(label)}</div>'
        f'<div class="metric-value">{safe(value)}</div><div class="metric-note">{safe(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section(title, note=""):
    st.markdown(
        f'<div class="section-head"><h3>{safe(title)}</h3>'
        f'<div class="surface-note">{safe(note)}</div></div>',
        unsafe_allow_html=True,
    )


def source_badge(is_demo):
    label = "Synthetic demonstration data" if is_demo else "Processed local session"
    class_name = "source-pill source-demo" if is_demo else "source-pill"
    st.markdown(
        f'<span class="{class_name}"><i class="source-dot"></i>{label}</span>',
        unsafe_allow_html=True,
    )


def style_figure(fig):
    palette = {
        "Video Sub-Score": "#5fe3f2",
        "Audio Sub-Score": "#a88bff",
        "Smoothed Trend": "#f0bd78",
        "Fused Engagement Score": "#f3f6ff",
        "Engagement Drop": "#fa8296",
        "Interaction Peak": "#8fdaff",
        "Gaze/Head Orientation": "#5fe3f2",
        "Motion Index": "#a88bff",
        "Speech Ratio (%)": "#5fe3f2",
        "Silence Ratio (%)": "#777f9c",
    }
    for trace in fig.data:
        color = palette.get(trace.name)
        if color and getattr(trace, "line", None) is not None:
            trace.line.color = color
        if color and getattr(trace, "marker", None) is not None:
            trace.marker.color = color
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aebbd7", family="Segoe UI, sans-serif", size=11),
        title_font=dict(color="#edf3ff", size=14),
        legend=dict(font=dict(color="#b6c2dd", size=10)),
        hoverlabel=dict(bgcolor="#151f39", bordercolor="#6f89d4", font=dict(color="#eef4ff", family="Segoe UI, sans-serif", size=11)),
        xaxis=dict(gridcolor="rgba(143,164,218,.12)", zerolinecolor="rgba(143,164,218,.18)", linecolor="rgba(143,164,218,.24)"),
        yaxis=dict(gridcolor="rgba(143,164,218,.12)", zerolinecolor="rgba(143,164,218,.18)", linecolor="rgba(143,164,218,.24)"),
        margin=dict(l=14, r=15, t=52, b=15),
    )
    return fig


with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><span class="brand-mark">◉</span><span class="brand-name">SkillWallet</span></div>'
        '<div class="brand-caption">Classroom intelligence / 01</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    if st.button("＋  Load sample classroom", use_container_width=True, help="Creates an explicitly labeled synthetic demo session in the local SQLite database."):
        with st.spinner("Preparing synthetic classroom data…"):
            result = utils.generate_demo_dataset()
        st.session_state["active_session_selector"] = result["session"]["session_id"]
        st.session_state["requested_route"] = "Dashboard"
        st.rerun()

    st.radio("Navigate", PAGES, key="nav_route", label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="eyebrow">Session context</div>', unsafe_allow_html=True)

    all_sessions = database.get_all_sessions()
    session_options = ["No session"] + [row["session_id"] for row in all_sessions]
    selected_id = st.selectbox(
        "Active session",
        session_options,
        key="active_session_selector",
        format_func=lambda value: (
            "Choose a session" if value == "No session"
            else next(
                (f"{row['session_name']} · {row['session_id'][:8]}" for row in all_sessions if row["session_id"] == value),
                value,
            )
        ),
        label_visibility="collapsed",
    )
    st.markdown(
        '<div style="color:#637069;font:9px DM Mono,monospace;line-height:1.7;margin-top:10px;">'
        'LOCAL-FIRST · NO CLOUD TRANSFER<br>ANONYMOUS TRACK TOKENS ONLY</div>',
        unsafe_allow_html=True,
    )


active_id = selected_id if selected_id != "No session" else None
active_session = database.get_session_by_id(active_id) if active_id else None
attendance_df = database.get_attendance_for_session(active_id) if active_id else pd.DataFrame()
intervals_df = database.get_intervals_for_session(active_id) if active_id else pd.DataFrame()
recommendations = database.get_recommendations_for_session(active_id) if active_id else []
route = st.session_state["nav_route"]


def page_header(index, title, description, is_demo=None):
    view_class = {
        "Dashboard": "dashboard",
        "Live classroom studio": "studio",
        "Attendance logs": "attendance",
        "Session dynamics": "dynamics",
        "Student cohorts": "cohorts",
        "Architecture & specifications": "architecture",
    }.get(title, "dashboard")
    st.markdown(
        f'<div class="topline"><span>SKILLWALLET / FIELD NOTES&nbsp;&nbsp; {index}</span>'
        f'<span>{"LOCAL SESSION" if active_session else "AWAITING SESSION"}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="view-heading {view_class}"><div class="view-index">WORKSPACE / {index}</div>'
        f'<div class="eyebrow">{safe(title)}</div><h1>{safe(title)}</h1>'
        f'<div class="page-intro">{safe(description)}</div></div>',
        unsafe_allow_html=True,
    )
    if active_session:
        source_badge(bool(active_session.get("is_demo", 0)) if is_demo is None else is_demo)


def no_session(copy="Load a synthetic session or analyze a classroom recording to populate this view."):
    st.markdown(
        f'<div class="surface" style="margin-top:20px"><div class="surface-title">No session selected</div>'
        f'<p style="font-size:13px;margin:9px 0 0">{safe(copy)}</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        if st.button("Explore the workspace", use_container_width=True):
            go_to("Dashboard")
    with right:
        if st.button("Open live classroom studio", use_container_width=True):
            go_to("Live classroom studio")


def render_landing():
    st.markdown(
        '<div class="topline"><span>SKILLWALLET&nbsp; / &nbsp;CLASSROOM INTELLIGENCE</span>'
        '<span>LOCAL PROCESSING&nbsp;&nbsp; · &nbsp;&nbsp;PRIVACY BY DESIGN</span></div>',
        unsafe_allow_html=True,
    )
    if active_session:
        is_demo = bool(active_session.get("is_demo", 0))
        label = "SYNTHETIC DEMONSTRATION" if is_demo else "LOCAL ANALYSIS"
        scores = intervals_df["fused_score"].dropna().astype(float).tolist() if "fused_score" in intervals_df else []
        score = round(sum(scores) / len(scores), 1) if scores else None
        attendance = (
            int((attendance_df["sustained_present"] == 1).sum())
            if "sustained_present" in attendance_df else None
        )
        score_label = f"{score:.1f}" if score is not None else "—"
        attendance_label = str(attendance) if attendance is not None else "—"
        if scores:
            low, high = min(scores), max(scores)
            span = max(high - low, 1)
            points = " ".join(
                f"{12 + (i * 276 / max(len(scores) - 1, 1)):.1f},{70 - ((value - low) / span * 46):.1f}"
                for i, value in enumerate(scores)
            )
            signal_viz = (
                f'<svg class="session-signal" viewBox="0 0 300 82" role="img" aria-label="Session interval score trend">'
                f'<path class="signal-area" d="M {points.replace(" ", " L ")} L 288 78 L 12 78 Z"/>'
                f'<polyline class="signal-line" points="{points}"/></svg>'
                '<div class="signal-caption"><span>INTERVAL SCORE TREND</span><span>SESSION DATA</span></div>'
            )
        else:
            signal_viz = '<div class="signal-empty">No interval signal is available for this session.</div>'
        session_panel = (
            '<div class="product-preview hero-data-card"><div class="preview-head"><span>SESSION SIGNAL</span>'
            f'<span class="preview-source">{safe(label)}</span></div><div class="preview-body">'
            f'<div class="session-name">{safe(active_session["session_name"])}</div>'
            f'<div class="signal-overview"><div class="score-ring" style="--score-progress:{min(max(score or 0, 0), 100)}%">'
            '<span class="score-ring-label">MEAN SCORE</span>'
            f'<strong>{safe(score_label)}</strong><small>/ 100</small></div>'
            '<div class="signal-stats"><div><span>SUSTAINED TRACKS</span>'
            f'<strong>{safe(attendance_label)}</strong></div><div><span>SESSION LENGTH</span>'
            f'<strong>{safe(utils.sec_to_mmss(active_session["video_duration_sec"]))}</strong></div></div></div>'
            f'{signal_viz}<div class="preview-foot">{safe(active_session["session_id"])}</div></div></div>'
        )
    else:
        session_panel = (
            '<div class="product-preview hero-data-card no-session"><div class="preview-head">'
            '<span>WORKSPACE SIGNAL PATH</span><span class="preview-source">NO SESSION DATA LOADED</span></div>'
            '<div class="preview-body"><div class="empty-signal-heading">Local by design.<br><span>Signals when you are ready.</span></div>'
            '<div class="pipeline-visual"><div class="pipeline-node"><i>01</i><b>Vision</b><small>ANONYMOUS PRESENCE</small></div>'
            '<div class="pipeline-link"></div><div class="pipeline-node violet"><i>02</i><b>Audio</b><small>ACOUSTIC RHYTHM</small></div>'
            '<div class="pipeline-link"></div><div class="pipeline-node"><i>03</i><b>Fusion</b><small>REVIEWABLE SIGNALS</small></div></div>'
            '<div class="preview-rule"></div><div class="preview-state">Analyze a local recording or load the clearly labeled sample session '
            'to view its real attendance and interval analytics.</div><div class="preview-foot">PROCESSING REMAINS ON THIS DEVICE</div></div></div>'
        )
    st.markdown(
        '<div class="hero-composition"><div class="hero-copy">'
        '<div class="hero-meta"><span class="hero-live-dot"></span> MULTIMODAL CLASSROOM INTELLIGENCE</div>'
        '<h1>Understand the classroom <span>beyond attendance.</span></h1>'
        '<p class="hero-sub">A privacy-conscious workspace for reviewing anonymous presence and classroom session dynamics. '
        'SkillWallet combines local video and audio analysis into interpretable signals for educator review—not student identity '
        'or cognitive assessment.</p><div class="hero-proof"><span>LOCAL VIDEO + AUDIO</span><i></i>'
        '<span>ANONYMOUS TRACKS</span><i></i><span>HUMAN REVIEW</span></div></div>'
        f'<div class="hero-visual">{session_panel}<div class="visual-orbit orbit-one"></div>'
        '<div class="visual-orbit orbit-two"></div></div></div>',
        unsafe_allow_html=True,
    )
    cta, note = st.columns([1, 2.2], vertical_alignment="center")
    with cta:
        if st.button("Launch workspace", type="primary", use_container_width=True):
            go_to("Dashboard")
    with note:
        st.markdown('<span class="data-label">ANONYMOUS TRACKING&nbsp; / &nbsp;LOCAL DATA&nbsp; / &nbsp;NO IDENTITY RECOGNITION</span>', unsafe_allow_html=True)
    section("Built for careful classroom review", "LOCAL-FIRST · EXPLAINABLE OUTPUTS")
    a, b = st.columns([1.2, 0.8], gap="large")
    with a:
        st.markdown(
            '<div class="surface"><div class="surface-title">One workspace, two synchronized inputs</div>'
            '<div class="surface-note">PROCESSING FLOW</div>'
            '<div style="display:flex;align-items:center;gap:10px;margin:22px 0 19px">'
            '<div style="flex:1;border:1px solid #35433a;border-radius:5px;padding:15px 12px">'
            '<div class="eyebrow">01 / Vision</div><div class="surface-title" style="margin-top:7px">Anonymous presence</div>'
            '<div class="event-copy">Face detection · spatial tracking</div></div>'
            '<div class="data-label">+</div>'
            '<div style="flex:1;border:1px solid #35433a;border-radius:5px;padding:15px 12px">'
            '<div class="eyebrow">02 / Audio</div><div class="surface-title" style="margin-top:7px">Speech activity</div>'
            '<div class="event-copy">Voice activity · silence ratio</div></div></div>'
            '<div class="preview-rule"></div><div class="preview-state">'
            '<strong>03 / Fusion</strong>&nbsp;&nbsp; Weighted interval signals and transparent review prompts.</div></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="surface"><div class="surface-title">Designed around privacy</div>'
            '<div class="event-card good"><div class="event-title">No identity recognition</div>'
            '<div class="event-copy">Tracks use anonymous sequential tokens. Names and face embeddings are not stored.</div></div>'
            '<div class="event-card"><div class="event-title">Signals are not diagnoses</div>'
            '<div class="event-copy">Observed behavior does not establish attention, comprehension, or learning.</div></div>'
            '<div class="surface-note">HUMAN REVIEW REMAINS ESSENTIAL</div></div>',
            unsafe_allow_html=True,
        )
    section("A complete classroom signal", "THREE LOCAL PROCESSING LAYERS")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="surface"><div class="eyebrow">01 / Vision</div><div class="surface-title">Anonymous presence</div><p>Face presence, spatial continuity and coarse orientation proxies, processed in memory.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="surface"><div class="eyebrow">02 / Audio</div><div class="surface-title">Acoustic rhythm</div><p>Voice activity and speech-to-silence balance from local signal processing.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="surface"><div class="eyebrow">03 / Fusion</div><div class="surface-title">Transparent signals</div><p>Weighted interval estimates, explainable event tags, and practical teaching prompts.</p></div>', unsafe_allow_html=True)


def render_dashboard():
    page_header("02 / OVERVIEW", "Dashboard", "A measured read on the session: who was present, how the room moved, and where discussion changed.")
    if not active_session:
        no_session("Choose a prior session, load the synthetic sample from the sidebar, or start a new analysis.")
        return
    if attendance_df.empty and intervals_df.empty:
        no_session("The selected session has no attendance or interval records.")
        return

    present = int((attendance_df["sustained_present"] == 1).sum()) if not attendance_df.empty else 0
    average = round(float(intervals_df["fused_score"].mean()), 1) if not intervals_df.empty else 0
    drops = int((intervals_df["event_tag"] == "ENGAGEMENT_DROP").sum()) if not intervals_df.empty else 0
    speech = round(float(intervals_df["speech_ratio"].mean()), 1) if not intervals_df.empty else 0
    cols = st.columns(4)
    with cols[0]: metric("Sustained presence", present, f"of {len(attendance_df)} anonymous tracks")
    with cols[1]: metric("Mean session signal", f"{average}", "multimodal estimate / 100")
    with cols[2]: metric("Attention shifts", drops, "intervals flagged for review")
    with cols[3]: metric("Speech activity", f"{speech}%", "average interval share")
    section("Session trace", f"{utils.sec_to_mmss(active_session['video_duration_sec'])} SESSION")
    if not intervals_df.empty:
        chart_col, signal_col = st.columns([1.7, 0.8], gap="large")
        with chart_col:
            fig = style_figure(fusion.create_engagement_timeline_chart(intervals_df))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with signal_col:
            st.markdown('<div class="surface"><div class="surface-title">Session readout</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="event-card {"good" if drops == 0 else ""}"><div class="event-title">{drops} attention shift{"s" if drops != 1 else ""}</div>'
                '<div class="event-copy">Low intervals are moments to revisit, not a judgment of learners.</div></div>'
                f'<div class="event-card good"><div class="event-title">{len(recommendations)} teaching prompt{"s" if len(recommendations) != 1 else ""}</div>'
                '<div class="event-copy">Deterministic recommendations linked to observed session patterns.</div></div>'
                f'<div class="surface-note">SOURCE / {"SYNTHETIC DEMO" if active_session.get("is_demo") else "LOCAL ANALYSIS"}</div></div>',
                unsafe_allow_html=True,
            )
            if recommendations:
                st.markdown(f"**Next prompt**  \n{recommendations[0].get('title', 'Review session dynamics')}")
                st.caption(recommendations[0].get("recommendation", ""))
    section("Anonymous attendance", "NO NAMES OR BIOMETRIC IDENTITIES")
    if not attendance_df.empty:
        st.dataframe(
            attendance_df[["track_id", "first_seen_sec", "last_seen_sec", "total_visible_sec", "status"]]
            .rename(columns={"track_id": "Track", "first_seen_sec": "First seen (s)", "last_seen_sec": "Last seen (s)", "total_visible_sec": "Visible (s)", "status": "Presence"}),
            use_container_width=True,
            hide_index=True,
        )


def render_studio():
    page_header("03 / CAPTURE", "Live classroom studio", "Bring a recording into the local analysis pipeline. Nothing is sent to a cloud service.")
    st.markdown(
        '<div class="surface"><div class="surface-title">Local analysis pipeline</div>'
        '<div class="surface-note">Video · optional synchronized audio · local SQLite persistence</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.markdown("#### Media inputs")
            uploaded_video = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"], help="Video is analyzed locally; the original is not stored in the database.")
            uploaded_audio = st.file_uploader("Synchronized audio (optional)", type=["wav", "mp3", "ogg", "flac"])
            session_name = st.text_input("Session / course label", value="Classroom session")
        with right:
            st.markdown("#### Analysis parameters")
            presence_threshold = st.slider("Sustained presence threshold", 1.0, 30.0, 5.0, 0.5, format="%.1f s")
            w_video, w_audio = st.columns(2)
            with w_video:
                video_weight = st.slider("Vision weight", 0.0, 1.0, 0.60, 0.05)
            with w_audio:
                audio_weight = st.slider("Audio weight", 0.0, 1.0, 0.40, 0.05)
            interval = st.select_slider("Interval window", options=[3.0, 5.0, 10.0, 15.0], value=5.0, format_func=lambda value: f"{value:g} seconds")
            sample_fps = st.selectbox("Vision sampling", [4.0, 2.0, 6.0], index=0, format_func=lambda value: f"{value:g} FPS")

        if st.button("Run local analysis  ↗", type="primary", use_container_width=True):
            if uploaded_video is None:
                st.error("Choose a classroom video before starting the analysis.")
            else:
                progress = st.progress(0.0, text="Preparing the local processing pipeline…")
                status = st.status("Analyzing classroom media on this device", expanded=True)
                video_path = None
                audio_path = None
                try:
                    suffix = os.path.splitext(uploaded_video.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                        temp_video.write(uploaded_video.getvalue())
                        video_path = temp_video.name
                    if uploaded_audio is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as temp_audio:
                            temp_audio.write(uploaded_audio.getvalue())
                            audio_path = temp_audio.name

                    status.write("Vision analysis · anonymous presence, orientation and motion")
                    video_result = vision.process_video_stream(
                        video_path,
                        presence_threshold_sec=presence_threshold,
                        interval_sec=interval,
                        target_sample_fps=sample_fps,
                        progress_callback=lambda pct, text: progress.progress(pct * 0.55, text=text),
                    )
                    st.session_state["preview_frames"] = video_result.get("preview_frames", [])
                    status.write("Audio signal analysis · local speech and silence estimation")
                    audio_result = audio.analyze_audio_stream(
                        audio_path=audio_path,
                        expected_duration_sec=video_result["meta"]["duration_sec"],
                        interval_sec=interval,
                    )
                    status.write("Multimodal fusion and transparent teaching-agent rules")
                    fused_intervals = fusion.fuse_multimodal_intervals(
                        video_result["intervals"], audio_result["intervals"],
                        video_weight=video_weight, audio_weight=audio_weight,
                        has_audio=audio_result["has_audio"],
                    )
                    recommendations_new = agent.PedagogicalAgent().analyze_session(
                        pd.DataFrame(fused_intervals),
                        pd.DataFrame(video_result["attendance"]),
                        presence_threshold_sec=presence_threshold,
                    )
                    session_id = f"SESS-{uuid.uuid4().hex[:8].upper()}"
                    session_header = {
                        "session_id": session_id,
                        "session_name": session_name,
                        "filename": uploaded_video.name,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "video_duration_sec": video_result["meta"]["duration_sec"],
                        "fps": video_result["meta"]["fps"],
                        "total_frames": video_result["meta"]["total_frames"],
                        "has_audio": int(audio_result["has_audio"]),
                        "audio_duration_sec": audio_result.get("duration_sec", 0.0),
                        "presence_threshold_sec": presence_threshold,
                        "video_weight": video_weight,
                        "audio_weight": audio_weight,
                        "interval_sec": interval,
                        "is_demo": 0,
                        "processing_time_sec": video_result["meta"]["processing_time_sec"],
                    }
                    status.write("Saving session records to the local database")
                    database.save_session(session_header)
                    database.save_attendance_batch(session_id, video_result["attendance"])
                    database.save_intervals_batch(session_id, fused_intervals)
                    database.save_recommendations_batch(session_id, recommendations_new)
                    st.session_state["requested_route"] = "Dashboard"
                    st.session_state["active_session_selector"] = session_id
                    progress.progress(1.0, text="Analysis complete")
                    status.update(label="Analysis complete · session saved locally", state="complete", expanded=False)
                    st.rerun()
                except Exception as error:
                    status.update(label="Analysis could not be completed", state="error", expanded=True)
                    st.error(f"Local analysis failed: {error}")
                finally:
                    utils.clean_temp_file(video_path)
                    utils.clean_temp_file(audio_path)

    if st.session_state.get("preview_frames"):
        section("Anonymous spatial tracking preview", "FRAMES HELD IN SESSION MEMORY")
        frame_cols = st.columns(len(st.session_state["preview_frames"]))
        for index, frame in enumerate(st.session_state["preview_frames"]):
            with frame_cols[index]:
                st.image(frame, caption=f"Analysis frame {index + 1}", use_container_width=True)
    section("Session archive", "STORED LOCALLY")
    sessions = database.get_all_sessions()
    if sessions:
        archive_df = pd.DataFrame(sessions)[["session_name", "session_id", "created_at", "video_duration_sec", "is_demo"]]
        archive_df["data source"] = archive_df["is_demo"].map({1: "Synthetic demo", 0: "Local analysis"})
        st.dataframe(archive_df.drop(columns="is_demo"), use_container_width=True, hide_index=True)
    else:
        st.caption("No sessions have been saved yet.")


def render_attendance():
    page_header("04 / PRESENCE", "Attendance logs", "An anonymous register of visible tracks with configurable sustained-presence context.")
    if not active_session:
        no_session()
        return
    if attendance_df.empty:
        no_session("No anonymous tracks were recorded for the selected session.")
        return
    sustained = int((attendance_df["sustained_present"] == 1).sum())
    threshold = float(active_session["presence_threshold_sec"])
    a, b, c = st.columns(3)
    with a: metric("Anonymous tracks", len(attendance_df), "sequential spatial tokens")
    with b: metric("Sustained presence", sustained, f"≥ {threshold:g} seconds")
    with c: metric("Brief detections", len(attendance_df) - sustained, "may include occlusion or brief crossings")
    st.markdown(
        '<div class="event-card good"><div class="event-title">Privacy boundary</div>'
        '<div class="event-copy">Track IDs are not identities. The system does not establish a person’s name or store biometric templates. Verify these records against an instructor’s own process.</div></div>',
        unsafe_allow_html=True,
    )
    section("Filter the register", "ANONYMOUS ATTENDANCE")
    filter_col, search_col, count_col = st.columns([1, 1.3, 0.7])
    with filter_col:
        status_filter = st.selectbox("Presence status", ["All", "Present", "Partial / Fleeting"])
    with search_col:
        search = st.text_input("Find track", placeholder="e.g. STU-03")
    view_df = attendance_df.copy()
    if status_filter != "All":
        view_df = view_df[view_df["status"] == status_filter]
    if search.strip():
        view_df = view_df[view_df["track_id"].str.contains(search.strip(), case=False, na=False)]
    with count_col:
        st.markdown(f'<div class="surface-note" style="margin-top:35px">{len(view_df)} RECORDS</div>', unsafe_allow_html=True)
    view_df["First seen"] = view_df["first_seen_sec"].map(utils.sec_to_mmss)
    view_df["Last seen"] = view_df["last_seen_sec"].map(utils.sec_to_mmss)
    view_df["Visible duration"] = view_df["total_visible_sec"].map(lambda value: f"{value:.1f} s")
    st.dataframe(
        view_df[["track_id", "First seen", "Last seen", "Visible duration", "status"]]
        .rename(columns={"track_id": "Anonymous track", "status": "Presence classification"}),
        use_container_width=True, hide_index=True,
    )
    csv_bytes = utils.export_dataframe_to_csv_bytes(view_df)
    st.download_button("Download attendance CSV", csv_bytes, f"skillwallet_attendance_{active_id}.csv", "text/csv")
    manual = st.number_input("Instructor headcount for comparison", min_value=0, max_value=200, value=sustained)
    difference = sustained - manual
    st.caption(f"Difference from sustained tracks: {difference:+d}. This is a review aid, not identity verification.")
    pdf_bytes = utils.generate_pdf_report(active_session, attendance_df, intervals_df, recommendations)
    st.download_button("Download session PDF report", pdf_bytes, f"skillwallet_report_{active_id}.pdf", "application/pdf")


def render_dynamics():
    page_header("05 / DYNAMICS", "Session dynamics", "Explore interval-level behavioral proxies and recalibrate fusion weights without rerunning video.")
    if not active_session:
        no_session()
        return
    if intervals_df.empty:
        no_session("No interval analytics were recorded for the selected session.")
        return
    st.markdown(
        '<div class="event-card"><div class="event-title">Interpretation note</div>'
        '<div class="event-copy">These values are observed behavioral proxies—not validated measures of attention, understanding, or learning. Use them as prompts for human review.</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Recalibrate signal weights", expanded=False):
        left, right, note = st.columns([1, 1, 1.3])
        with left:
            vision_weight = st.slider("Vision", 0.0, 1.0, float(active_session.get("video_weight", .6)), .05, key=f"vw_{active_id}")
        with right:
            audio_weight = st.slider("Audio", 0.0, 1.0, float(active_session.get("audio_weight", .4)), .05, key=f"aw_{active_id}")
        with note:
            st.caption("Weights are normalized by the fusion engine. Sessions without decodable audio use vision-only scoring.")
        if st.button("Apply weights", key=f"apply_{active_id}"):
            database.update_session_weights(active_id, vision_weight, audio_weight)
            st.rerun()
        intervals_view = fusion.recalculate_fusion_df(
            intervals_df, vision_weight, audio_weight, has_audio=bool(active_session.get("has_audio", 1))
        )
    if "intervals_view" not in locals():
        intervals_view = fusion.recalculate_fusion_df(
            intervals_df,
            float(active_session.get("video_weight", .6)),
            float(active_session.get("audio_weight", .4)),
            has_audio=bool(active_session.get("has_audio", 1)),
        )
    avg = round(float(intervals_view["fused_score"].mean()), 1)
    d1, d2, d3 = st.columns(3)
    with d1: metric("Mean fused signal", avg, "weighted behavioral estimate / 100")
    with d2: metric("Interaction peaks", int((intervals_view["event_tag"] == "INTERACTION_PEAK").sum()), "intervals ≥ configured peak")
    with d3: metric("Review moments", int((intervals_view["event_tag"] == "ENGAGEMENT_DROP").sum()), "intervals below 40")
    tab_timeline, tab_components, tab_ledger = st.tabs(["Session timeline", "Signal components", "Interval ledger"])
    with tab_timeline:
        st.plotly_chart(style_figure(fusion.create_engagement_timeline_chart(intervals_view)), use_container_width=True, config={"displayModeBar": False})
    with tab_components:
        st.plotly_chart(style_figure(fusion.create_metrics_breakdown_chart(intervals_view)), use_container_width=True, config={"displayModeBar": False})
    with tab_ledger:
        event_filter = st.selectbox("Event filter", ["All intervals", "ENGAGEMENT_DROP", "INTERACTION_PEAK", "MONOLOGUE", "LOW_ACTIVITY"])
        table = intervals_view.copy()
        if event_filter != "All intervals":
            table = table[table["event_tag"] == event_filter]
        table["Time"] = table.apply(lambda row: f"{utils.sec_to_mmss(row['start_sec'])}–{utils.sec_to_mmss(row['end_sec'])}", axis=1)
        st.dataframe(
            table[["interval_idx", "Time", "fused_score", "video_score", "audio_score", "head_orientation_score", "motion_index", "speech_ratio", "event_tag"]]
            .rename(columns={"interval_idx": "Interval", "fused_score": "Fused", "video_score": "Vision", "audio_score": "Audio", "head_orientation_score": "Orientation proxy", "motion_index": "Motion", "speech_ratio": "Speech %", "event_tag": "Event"}),
            use_container_width=True, hide_index=True,
        )
        st.download_button("Download interval CSV", utils.export_dataframe_to_csv_bytes(table), f"skillwallet_dynamics_{active_id}.csv", "text/csv")
    if recommendations:
        section("Transparent teaching prompts", "RULE-BASED · HUMAN REVIEW")
        for item in recommendations:
            severity = safe(item.get("severity", "INFO"))
            st.markdown(
                f'<div class="event-card {"good" if severity == "INFO" else ""}"><div class="surface-note">{severity} / {safe(item.get("rule_id"))} / {safe(item.get("time_window"))}</div>'
                f'<div class="event-title" style="margin-top:7px">{safe(item.get("title"))}</div>'
                f'<div class="event-copy">{safe(item.get("trigger_metric"))}<br><br>{safe(item.get("recommendation"))}</div></div>',
                unsafe_allow_html=True,
            )
        if st.button("Re-evaluate teaching prompts on current weights"):
            updated = agent.PedagogicalAgent().analyze_session(
                intervals_view, attendance_df, presence_threshold_sec=active_session.get("presence_threshold_sec", 5.0)
            )
            database.save_recommendations_batch(active_id, updated)
            st.rerun()


def render_cohorts():
    page_header("06 / COHORTS", "Student cohorts", "A privacy-preserving view of presence-duration groups. Per-student engagement is not measured or inferred.")
    if not active_session:
        no_session()
        return
    if attendance_df.empty:
        no_session("No presence records are available for this session.")
        return
    duration = attendance_df["total_visible_sec"]
    threshold = float(active_session.get("presence_threshold_sec", 5.0))
    sustained = attendance_df["sustained_present"] == 1
    late = (attendance_df["first_seen_sec"] > max(10.0, threshold * 2)) & sustained
    groups = [
        ("Sustained presence", int(sustained.sum()), f"Visible ≥ {threshold:g} s", "#70e1c0"),
        ("Late arrival pattern", int(late.sum()), "First seen after session start", "#c0d98a"),
        ("Brief / partial tracks", int((~sustained).sum()), f"Visible < {threshold:g} s", "#e3ad68"),
    ]
    cols = st.columns(3)
    for column, (name, count, rule, accent) in zip(cols, groups):
        with column:
            st.markdown(
                f'<div class="cohort"><div class="surface-note">ANONYMOUS PRESENCE GROUP</div>'
                f'<div class="cohort-num" style="color:{accent}">{count:02d}</div>'
                f'<div class="cohort-name">{safe(name)}</div><div class="cohort-rule">{safe(rule)}</div></div>',
                unsafe_allow_html=True,
            )
    section("Distribution of visible time", "SESSION-LEVEL PRESENCE")
    cohort_names = [name for name, _, _, _ in groups]
    cohort_counts = [count for _, count, _, _ in groups]
    cohort_figure = go.Figure(
        go.Bar(
            x=cohort_names,
            y=cohort_counts,
            marker_color=["#70e1c0", "#c0d98a", "#e3ad68"],
            hovertemplate="%{x}<br>%{y} anonymous tracks<extra></extra>",
        )
    )
    cohort_figure.update_layout(
        title="Anonymous track duration groups",
        yaxis_title="Tracks",
        showlegend=False,
        bargap=0.55,
    )
    st.plotly_chart(style_figure(cohort_figure), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div class="event-card"><div class="event-title">A careful boundary</div>'
        '<div class="event-copy">Cohorts here describe track duration only. They are not student identities, ability groups, or individual engagement classifications. An anonymous token can also split or disappear when camera conditions change.</div></div>',
        unsafe_allow_html=True,
    )
    rows = attendance_df.copy()
    rows["Presence cohort"] = rows.apply(
        lambda row: "Brief / partial" if not row["sustained_present"] else "Late arrival pattern" if row["first_seen_sec"] > max(10.0, threshold * 2) else "Sustained presence",
        axis=1,
    )
    cohort_filter = st.selectbox("Show presence group", ["All groups"] + [item[0] for item in groups])
    if cohort_filter != "All groups":
        rows = rows[rows["Presence cohort"] == cohort_filter]
    st.dataframe(rows[["track_id", "Presence cohort", "first_seen_sec", "total_visible_sec"]].rename(columns={"track_id": "Anonymous track", "first_seen_sec": "First seen (s)", "total_visible_sec": "Visible duration (s)"}), use_container_width=True, hide_index=True)


def render_architecture():
    page_header("07 / SYSTEM", "Architecture & specifications", "A transparent, local-first pipeline with explicit privacy boundaries and interpretable outputs.")
    left, right = st.columns([1.4, 0.8], gap="large")
    with left:
        st.markdown('<div class="surface"><div class="surface-title">Processing topology</div><div class="surface-note">IN-MEMORY ANALYSIS → LOCAL SQLITE</div>', unsafe_allow_html=True)
        for label in ["CLASSROOM VIDEO  /  MP4 · AVI · MOV", "VISION  /  OpenCV face presence + centroid tracking", "ANONYMOUS TRACKS  /  STU-01 · STU-02 · …", "CLASSROOM AUDIO  /  WAV · MP3 · OGG (OPTIONAL)", "SIGNAL ANALYSIS  /  RMS · ZCR · voice activity", "FUSION  /  normalized interval weights", "RULE AGENT  /  explainable pedagogical prompts", "PERSISTENCE  /  local SQLite · CSV · PDF"]:
            st.markdown(f'<div class="flow-node">{safe(label)}</div><div class="flow-arrow">↓</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        stats = database.get_db_stats()
        st.markdown('<div class="surface"><div class="surface-title">Runtime specification</div>', unsafe_allow_html=True)
        for label, value in [
            ("Application", "Python · Streamlit"),
            ("Vision", "OpenCV · Haar cascades"),
            ("Audio", "SoundFile · NumPy · VAD"),
            ("Analytics", "Pandas · Plotly"),
            ("Persistence", "SQLite"),
            ("Database size", f"{stats['size_kb']} KB"),
            ("Saved sessions", stats["sessions"]),
            ("Attendance records", stats["attendance_records"]),
            ("Interval records", stats["intervals"]),
        ]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #252e29;padding:10px 0"><span class="data-label">{safe(label)}</span><span style="font-size:11px;color:#cad3cc;text-align:right">{safe(value)}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="event-card good"><div class="event-title">Privacy, by construction</div>'
            '<div class="event-copy">No face embeddings, names, or external LLM APIs. Frames are processed in memory. Persisted analytics remain on the local machine.</div></div>'
            '<div class="event-card"><div class="event-title">Proxy metric disclaimer</div>'
            '<div class="event-copy">Orientation, motion and speech ratios do not establish attention, comprehension, or student intent.</div></div>',
            unsafe_allow_html=True,
        )
    if active_session:
        section("Active session specification", "SESSION METADATA")
        st.dataframe(pd.DataFrame([{
            "Session": active_session["session_name"],
            "Created": active_session["created_at"],
            "Duration (s)": active_session["video_duration_sec"],
            "Sampling FPS": active_session["fps"],
            "Interval (s)": active_session["interval_sec"],
            "Video weight": active_session["video_weight"],
            "Audio weight": active_session["audio_weight"],
            "Audio available": bool(active_session["has_audio"]),
            "Source": "Synthetic demo" if active_session["is_demo"] else "Local analysis",
        }]), use_container_width=True, hide_index=True)


if route == "Welcome":
    render_landing()
elif route == "Dashboard":
    render_dashboard()
elif route == "Live classroom studio":
    render_studio()
elif route == "Attendance logs":
    render_attendance()
elif route == "Session dynamics":
    render_dynamics()
elif route == "Student cohorts":
    render_cohorts()
else:
    render_architecture()
