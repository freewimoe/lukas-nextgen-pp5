import streamlit as st

def render():
    """Render the Recommendations page."""
    st.title("✨ Recommendations")
    st.caption("Draft youth activities ranked by expected attendance and bonding potential.")
    
    uploaded = st.file_uploader("Upload event drafts CSV", type=["csv"]) 
    if uploaded:
        st.info("Recommendation engine will be implemented when model training is ready.")
    else:
        st.info("Download the template from the README and upload your drafts.")
        
    st.markdown("""
    **How it will work:**
    1. Upload CSV with planned events
    2. AI predicts attendance and bonding potential
    3. Get ranked recommendations for optimal youth engagement
    """)