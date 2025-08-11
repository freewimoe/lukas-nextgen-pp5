import streamlit as st
import pandas as pd
import plotly.express as px

def render():
    """Render the EDA page."""
    st.title("🔎 Exploratory Data Analysis")
    st.caption("Upload events data to explore patterns.")
    
    data_file = st.file_uploader("Upload EVENTS CSV", type=["csv"]) 
    if not data_file:
        st.info("Upload a CSV with columns like attendance, weekday, event_type, price_eur…")
        return
    
    try:
        df = pd.read_csv(data_file)
        st.write("Shape:", df.shape)
        st.dataframe(df.head())

        num_cols = df.select_dtypes("number").columns.tolist()
        if num_cols:
            col = st.selectbox("Numeric column", num_cols)
            fig = px.histogram(df, x=col, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

        if len(num_cols) > 1:
            st.subheader("Correlation (numeric)")
            st.dataframe(df[num_cols].corr().style.background_gradient(cmap="Blues"))
    except Exception as e:
        st.error(f"Error loading data: {e}")