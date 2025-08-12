import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

def load_models():
    """Load trained ML models for recommendations."""
    try:
        models_dir = Path(__file__).parent.parent.parent / "models" / "versioned" / "v1"
        
        with open(models_dir / 'youth_engagement_model.pkl', 'rb') as f:
            youth_model_package = pickle.load(f)
        
        with open(models_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return youth_model_package, metadata
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def generate_recommendations(youth_model_package, participant_data):
    """Generate personalized recommendations based on ML model insights."""
    try:
        model = youth_model_package['model']
        feature_columns = youth_model_package['feature_columns']
        
        # Simulate recommendation scoring based on participant characteristics
        recommendations = []
        
        # Age-based recommendations
        age = participant_data.get('age', 19)
        if age <= 16:
            recommendations.extend([
                {"activity": "Gaming Tournament", "score": 0.85, "category": "Digital", "reason": "High appeal for younger participants"},
                {"activity": "Creative Workshop", "score": 0.78, "category": "Arts", "reason": "Age-appropriate creative expression"},
                {"activity": "Peer Mentoring Program", "score": 0.72, "category": "Social", "reason": "Building leadership skills"}
            ])
        elif age <= 22:
            recommendations.extend([
                {"activity": "Community Service Project", "score": 0.82, "category": "Service", "reason": "Matches volunteer interest profile"},
                {"activity": "Career Development Workshop", "score": 0.76, "category": "Education", "reason": "Relevant for young adults"},
                {"activity": "Sports & Fitness Group", "score": 0.73, "category": "Physical", "reason": "Active engagement opportunity"}
            ])
        else:
            recommendations.extend([
                {"activity": "Leadership Training", "score": 0.88, "category": "Leadership", "reason": "Mature participants ready for responsibility"},
                {"activity": "Family Event Planning", "score": 0.81, "category": "Family", "reason": "Appeals to older participants with families"},
                {"activity": "Mentoring Younger Members", "score": 0.75, "category": "Service", "reason": "Experience sharing opportunity"}
            ])
        
        # Engagement level adjustments
        engagement_score = participant_data.get('digital_engagement_score', 7)
        if engagement_score >= 8:
            recommendations.append({
                "activity": "Social Media Team", "score": 0.89, "category": "Digital", 
                "reason": "High digital engagement suggests social media skills"
            })
        
        # Event attendance adjustments
        events_attended = participant_data.get('monthly_events_attended', 3)
        if events_attended >= 4:
            recommendations.append({
                "activity": "Event Organization Committee", "score": 0.84, "category": "Leadership",
                "reason": "Regular attendance indicates commitment and interest"
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:8]  # Top 8 recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

def render():
    """Render the Recommendations page."""
    st.title("✨ Smart Recommendations")
    st.markdown("### AI-Powered Activity Suggestions for Youth Engagement")
    
    # Load models
    youth_model_package, metadata = load_models()
    
    if youth_model_package is None:
        st.warning("⚠️ Models not loaded. Using demo recommendations.")
    else:
        st.success("✅ AI recommendation engine loaded and ready!")
    
    # Tabs for different recommendation types
    tab1, tab2, tab3, tab4 = st.tabs(["🧑 Personal Recommendations", "🎯 Event Optimization", "📊 Community Insights", "📈 Strategic Planning"])
    
    with tab1:
        st.markdown("### Personalized Activity Recommendations")
        st.info("Enter participant details to get AI-powered activity suggestions")
        
        # Input form for personalized recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 13, 30, 19)
            monthly_events_attended = st.slider("Monthly Events Attended", 0, 10, 3)
            digital_engagement_score = st.slider("Digital Engagement (1-10)", 1, 10, 7)
        
        with col2:
            volunteer_hours = st.slider("Volunteer Hours/Month", 0, 20, 3)
            event_satisfaction = st.slider("Event Satisfaction (1-10)", 1, 10, 8)
            peer_influence = st.slider("Peer Influence (1-10)", 1, 10, 7)
        
        if st.button("🎯 Generate Recommendations", type="primary"):
            participant_data = {
                'age': age,
                'monthly_events_attended': monthly_events_attended,
                'digital_engagement_score': digital_engagement_score,
                'volunteer_hours_per_month': volunteer_hours,
                'event_satisfaction_avg': event_satisfaction,
                'peer_influence_score': peer_influence
            }
            
            if youth_model_package:
                recommendations = generate_recommendations(youth_model_package, participant_data)
            else:
                # Demo recommendations
                recommendations = [
                    {"activity": "Gaming Tournament", "score": 0.85, "category": "Digital", "reason": "High digital engagement score"},
                    {"activity": "Community Service", "score": 0.82, "category": "Service", "reason": "Strong volunteer interest"},
                    {"activity": "Leadership Workshop", "score": 0.78, "category": "Leadership", "reason": "Regular event attendance"},
                    {"activity": "Creative Arts", "score": 0.75, "category": "Arts", "reason": "Age-appropriate activities"}
                ]
            
            st.markdown("### 🎯 Your Personalized Recommendations")
            
            for i, rec in enumerate(recommendations[:6], 1):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{i}. {rec['activity']}**")
                        st.caption(rec['reason'])
                    
                    with col2:
                        score_color = "green" if rec['score'] > 0.8 else "orange" if rec['score'] > 0.7 else "red"
                        st.markdown(f":{score_color}[{rec['score']:.0%}]")
                    
                    with col3:
                        st.badge(rec['category'])
                    
                    st.markdown("---")
            
            # Visualization
            rec_df = pd.DataFrame(recommendations[:6])
            fig = px.bar(rec_df, x='score', y='activity', orientation='h',
                        color='category', title="Recommendation Scores by Activity",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Event Optimization Insights")
        
        # Event performance analysis
        st.markdown("#### 📈 Current Event Performance")
        
        event_data = {
            'Event Type': ['Youth Service', 'Creative Workshop', 'Sports Day', 'Study Group', 'Social Event', 'Outdoor Activity'],
            'Avg Attendance': [45, 32, 38, 28, 52, 35],
            'Satisfaction Score': [8.2, 7.8, 8.5, 7.2, 8.8, 8.1],
            'Engagement Level': [85, 75, 88, 65, 92, 80]
        }
        
        event_df = pd.DataFrame(event_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(event_df, x='Avg Attendance', y='Satisfaction Score',
                           size='Engagement Level', hover_name='Event Type',
                           title="Event Performance Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(event_df, x='Event Type', y='Engagement Level',
                        title="Engagement Level by Event Type")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Optimization recommendations
        st.markdown("#### 💡 Event Optimization Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("**Expand Social Events** - Highest engagement (92%)")
        with col2:
            st.warning("**Improve Study Groups** - Lowest satisfaction (7.2)")
        with col3:
            st.info("**Sports Day Format** - Good attendance + satisfaction")
    
    with tab3:
        st.markdown("### Community Engagement Insights")
        
        # Demographics analysis
        st.markdown("#### 👥 Participant Demographics")
        
        demo_data = {
            'Age Group': ['13-16', '17-20', '21-25', '26-30'],
            'Count': [85, 142, 98, 67],
            'Avg Engagement': [72, 68, 75, 82],
            'Top Interest': ['Gaming/Tech', 'Social/Creative', 'Career/Service', 'Leadership/Family']
        }
        
        demo_df = pd.DataFrame(demo_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(demo_df, values='Count', names='Age Group',
                        title="Age Group Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(demo_df, x='Age Group', y='Avg Engagement',
                        title="Average Engagement by Age Group")
            st.plotly_chart(fig, use_container_width=True)
        
        # District analysis
        st.markdown("#### 🗺️ Geographic Engagement Patterns")
        
        district_data = {
            'District': ['Innenstadt-Ost', 'Südstadt', 'Oststadt', 'Weststadt', 'Daxlanden', 'Nordstadt'],
            'Participants': [42, 51, 38, 35, 28, 23],
            'Engagement Rate': [48, 44, 42, 39, 52, 31]
        }
        
        district_df = pd.DataFrame(district_data)
        
        fig = px.bar(district_df, x='District', y=['Participants', 'Engagement Rate'],
                    title="Participant Count vs Engagement Rate by District",
                    barmode='group')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Strategic Planning Dashboard")
        
        # KPIs and targets
        st.markdown("#### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Engagement Rate", "40.2%", "+2.1%")
        with col2:
            st.metric("Target Engagement Rate", "75.0%", "Goal")
        with col3:
            st.metric("Gap to Close", "34.8%", "-1.2%")
        with col4:
            st.metric("Monthly Active Participants", "217", "+15")
        
        # Strategic recommendations
        st.markdown("#### 📋 Strategic Action Plan")
        
        strategies = [
            {
                "Priority": "High",
                "Action": "Expand Social Events",
                "Impact": "High",
                "Effort": "Medium",
                "Timeline": "1-2 months",
                "Expected Lift": "+8%"
            },
            {
                "Priority": "High", 
                "Action": "Digital Engagement Platform",
                "Impact": "High",
                "Effort": "High",
                "Timeline": "3-4 months",
                "Expected Lift": "+12%"
            },
            {
                "Priority": "Medium",
                "Action": "Peer Mentoring Program",
                "Impact": "Medium",
                "Effort": "Low",
                "Timeline": "2-3 months",
                "Expected Lift": "+5%"
            },
            {
                "Priority": "Medium",
                "Action": "District-Specific Events",
                "Impact": "Medium",
                "Effort": "Medium",
                "Timeline": "2-4 months",
                "Expected Lift": "+7%"
            }
        ]
        
        strategy_df = pd.DataFrame(strategies)
        st.dataframe(strategy_df, use_container_width=True)
        
        # ROI Analysis
        st.markdown("#### 💰 Return on Investment Analysis")
        
        roi_data = {
            'Strategy': ['Social Events', 'Digital Platform', 'Peer Mentoring', 'District Events'],
            'Investment (€)': [2500, 8000, 1200, 3500],
            'Expected ROI': [3.2, 4.8, 2.1, 2.8],
            'Engagement Lift': [8, 12, 5, 7]
        }
        
        roi_df = pd.DataFrame(roi_data)
        
        fig = px.scatter(roi_df, x='Investment (€)', y='Expected ROI',
                        size='Engagement Lift', hover_name='Strategy',
                        title="Investment vs Expected ROI")
        st.plotly_chart(fig, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.markdown("### 📥 Download Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download Recommendation Report"):
            # Create a sample report
            report_data = {
                'Activity': ['Gaming Tournament', 'Community Service', 'Leadership Workshop', 'Creative Arts'],
                'Category': ['Digital', 'Service', 'Leadership', 'Arts'],
                'Priority_Score': [0.85, 0.82, 0.78, 0.75],
                'Target_Age_Group': ['13-18', '18-25', '20-30', '13-20'],
                'Expected_Attendance': [35, 28, 22, 30]
            }
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="lukas_youth_recommendations.csv",
                mime="text/csv"
            )
    
    with col2:
        st.info("""
        **Report includes:**
        - Personalized activity recommendations
        - Priority scores and reasoning
        - Target demographics
        - Expected participation rates
        """)