import streamlit as st

def render():
    """Render the Metrics page."""
    st.title("🧪 Metrics")
    st.info("Model metrics will be displayed here after training.")
    st.markdown("""
    **Expected metrics:**
    - Training accuracy/R²
    - Test accuracy/R²
    - F1 scores (for classification)
    - MAE/RMSE (for regression)
    """)