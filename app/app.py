import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app_pages import (
    project_summary, eda, train_model, predict, metrics,
    community_impact, recommendations
)

st.set_page_config(
    page_title="LUKAS NextGen – Youth Engagement Analytics",
    page_icon="📊",
    layout="wide",
)

PAGES = {
    "📘 Project Summary": project_summary.render,
    "🔎 EDA": eda.render,
    "🧠 Train Model": train_model.render,
    "📈 Predict": predict.render,
    "🧪 Metrics": metrics.render,
    "🌱 Community & Impact": community_impact.render,
    "✨ Recommendations": recommendations.render,
}

with st.sidebar:
    st.title("Navigation")
    choice = st.radio("Go to", list(PAGES.keys()))
    st.markdown("---")
    st.markdown("**Community links**")
    st.link_button("Lukasgemeinde Karlsruhe", "https://www.lukasgemeinde-karlsruhe.de")
    st.link_button("Wir für Lukas", "https://www.wir-fuer-lukas.de")

PAGES[choice]()