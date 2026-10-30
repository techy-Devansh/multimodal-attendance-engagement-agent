"""Streamlit entry point for the SkillWallet classroom intelligence workspace."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("studio_app.py")))
