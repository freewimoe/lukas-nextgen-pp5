import streamlit as st

def render():
    """Render the Community Impact page"""
    st.title("🤝 Community & Impact")
    
    st.markdown("""
    ## 🌱 **Why This Matters**
    
    We help young people grow in **agency, resilience, and spirituality** through 
    data-driven community programs that create lasting positive impact.
    
    ### **Community Benefits:**
    - **Youth Development**: Evidence-based programs that truly engage young people
    - **Financial Sustainability**: Optimized resource allocation for maximum impact
    - **Community Bonding**: Stronger connections between generations
    - **Innovation**: Cutting-edge approaches to traditional ministry
    """)
    
    # Impact metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Expected Impact")
        st.write("• **75% youth retention** (up from 60%)")
        st.write("• **€36,000 annual savings** through optimization")
        st.write("• **12 new community programs** per year")
        st.write("• **450+ youth** directly benefited")
    
    with col2:
        st.subheader("🎯 Target Demographics")
        st.write("• **Ages 13-30** in Karlsruhe region")
        st.write("• **All backgrounds** welcome")
        st.write("• **8 districts** covered")
        st.write("• **Multi-cultural** programming")
    
    st.markdown("---")
    
    st.subheader("🔗 Get Involved")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.link_button(
            "💬 Give Feedback", 
            "https://your-form-link.example",
            help="Share your ideas for youth programs"
        )
    
    with col2:
        st.link_button(
            "🏛️ Lukasgemeinde Karlsruhe", 
            "https://www.lukasgemeinde-karlsruhe.de",
            help="Visit our main church website"
        )
    
    with col3:
        st.link_button(
            "🚀 Wir für Lukas", 
            "https://www.wir-fuer-lukas.de",
            help="Innovation initiative for community development"
        )
    
    st.markdown("""
    ---
    
    ### 🔒 **Data Privacy & Ethics**
    
    All data is **anonymized** and used exclusively to improve youth programs. 
    We follow **GDPR compliance** and **ethical AI practices** to protect 
    participant privacy while maximizing community benefit.
    
    **Contact:** For questions about this project or data usage, please reach out 
    through the Lukasgemeinde Karlsruhe website.
    """)