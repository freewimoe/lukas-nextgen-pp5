import streamlit as st

def render():
    """Render the Project Summary page"""
    st.title("📘 LUKAS NextGen - Project Summary")
    
    st.markdown("""
    ## 🎯 **Project Objectives**
    
    This **PP5 Portfolio Project** develops a predictive analytics solution to optimize 
    **youth engagement** and **financial sustainability** for the Lukasgemeinde Karlsruhe.
    
    ### **Key Goals:**
    1. **Predict youth engagement levels** and identify retention factors
    2. **Forecast financial sustainability** for building maintenance & operations  
    3. **Strengthen community bonding** through evidence-based programming
    4. **Create actionable recommendations** for innovative funding concepts
    
    ---
    
    ## 🗂️ **How to Use This Dashboard:**
    
    1. **📊 EDA** - Explore the data patterns and insights
    2. **🤖 Train Model** - Build and train machine learning models
    3. **🔮 Predict** - Make predictions for youth engagement scenarios
    4. **📈 Metrics** - Evaluate model performance and accuracy
    5. **💡 Recommendations** - Get data-driven suggestions for improvement
    6. **🤝 Community Impact** - Understand the broader community benefits
    
    ---
    
    ## 📊 **Technical Stack**
    - **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
    - **Machine Learning:** Scikit-learn, Optuna (hyperparameter tuning)
    - **Visualization:** Plotly, Streamlit Dashboard
    - **Development:** Jupyter Notebooks, Python 3.12
    
    ---
    
    ## 🌐 **Project Context**
    
    **Theme:** www.wir-fuer-lukas.de - Innovative Konzepte für die Lukasgemeinde Karlsruhe
    
    This project is part of the **Code Institute Full Stack Development Diploma** 
    and focuses on using predictive analytics to strengthen youth ministry and 
    community sustainability in Karlsruhe.
    """)
    
    # Success metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Target Youth Engagement",
            value="75%",
            delta="15% improvement"
        )
    
    with col2:
        st.metric(
            label="Financial Sustainability",
            value="€12,000/month",
            delta="€2,500 increase"
        )
    
    with col3:
        st.metric(
            label="Community Programs",
            value="8 events/month",
            delta="3 new programs"
        )