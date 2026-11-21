"""From the project root: python -m streamlit run src/dashboard/app.py"""

from pathlib import Path
import sys

# Streamlit also supports launching by absolute filename from a different cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dashboard.dashboard_ui import render_dashboard

render_dashboard(None)
