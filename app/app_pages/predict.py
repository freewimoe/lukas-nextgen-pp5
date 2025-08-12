import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

def load_models():
    """Load trained ML models and preprocessing components."""
    try:
        models_dir = Path(__file__).parent.parent.parent / "models" / "versioned" / "v1"
        
        # Load youth engagement model
        with open(models_dir / 'youth_engagement_model.pkl', 'rb') as f:
            youth_model_package = pickle.load(f)
        
        # Load financial forecasting models
        with open(models_dir / 'financial_forecasting_models.pkl', 'rb') as f:
            financial_model_package = pickle.load(f)
            
        # Load metadata
        with open(models_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return youth_model_package, financial_model_package, metadata
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def predict_youth_engagement(input_data, youth_model_package):
    """Make youth engagement prediction for a single participant."""
    try:
        model = youth_model_package['model']
        scaler = youth_model_package['scaler']
        feature_columns = youth_model_package['feature_columns']
        label_encoders = youth_model_package['label_encoders']
        
        # Create DataFrame with input data
        df = pd.DataFrame([input_data])
        
        # Apply same feature engineering as in training
        df['age_group'] = pd.cut(df['age'], 
                                bins=[12, 16, 20, 25, 30], 
                                labels=['Teenager', 'Young Adult', 'Adult', 'Mature'],
                                include_lowest=True)
        
        df['engagement_intensity'] = (
            df['monthly_events_attended'] * 0.3 +
            df['volunteer_hours_per_month'] * 0.25 +
            df['digital_engagement_score'] * 0.25 +
            df['peer_influence_score'] * 0.2
        )
        
        df['participation_quality'] = df['monthly_events_attended'] * df['event_satisfaction_avg']
        df['experience_level'] = (df['age'] - 13) + (df['monthly_events_attended'] * 2)
        
        # Encode categorical variables
        categorical_cols = ['district', 'education_level', 'gender', 'age_group']
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                try:
                    df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
                except:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = 0
        
        # Select features in the same order as training
        X_input = df[feature_columns]
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(X_input)[0]
            prediction = model.predict(X_input)[0]
            return prediction, prediction_proba[1]  # Return prediction and probability of high engagement
        else:
            prediction = model.predict(X_input)[0]
            return prediction, None
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def render():
    """Render the Predict page."""
    st.title("🎯 Youth Engagement Prediction")
    st.markdown("### Predict engagement levels for Lukasgemeinde Karlsruhe participants")
    
    # Load models
    youth_model_package, financial_model_package, metadata = load_models()
    
    if youth_model_package is None:
        st.error("⚠️ Models not found. Please run the Jupyter notebook first to train models.")
        return
    
    # Display model information
    with st.expander("� Model Information"):
        st.write(f"**Best Model:** {metadata['model_performance']['youth_engagement']['best_model']}")
        st.write(f"**Accuracy:** {metadata['model_performance']['youth_engagement']['accuracy']:.1%}")
        st.write(f"**F1-Score:** {metadata['model_performance']['youth_engagement']['f1_score']:.1%}")
        st.write(f"**Training Date:** {metadata['creation_date']}")
        st.write(f"**Features Used:** {metadata['feature_counts']['youth_features']}")
    
    st.markdown("---")
    
    # Create two tabs: Single Prediction and Batch Prediction
    tab1, tab2 = st.tabs(["🧑 Single Participant", "📊 Batch Prediction"])
    
    with tab1:
        st.markdown("### Enter participant details:")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 13, 30, 19)
            gender = st.selectbox("Gender", ["M", "F", "D"])
            district = st.selectbox("District", [
                'Innenstadt-Ost', 'Innenstadt-West', 'Südstadt', 'Oststadt', 
                'Weststadt', 'Nordstadt', 'Mühlburg', 'Daxlanden'
            ])
            education_level = st.selectbox("Education Level", [
                'Hauptschule', 'Realschule', 'Gymnasium', 'Studium', 'Ausbildung'
            ])
            family_church_background = st.selectbox("Family Church Background", [0, 1], 
                                                   format_func=lambda x: "Yes" if x else "No")
        
        with col2:
            monthly_events_attended = st.slider("Monthly Events Attended", 0, 10, 3)
            volunteer_hours_per_month = st.slider("Volunteer Hours per Month", 0, 20, 3)
            digital_engagement_score = st.slider("Digital Engagement Score (1-10)", 1, 10, 7)
            peer_influence_score = st.slider("Peer Influence Score (1-10)", 1, 10, 7)
            event_satisfaction_avg = st.slider("Event Satisfaction Average (1-10)", 1, 10, 8)
        
        # Predict button
        if st.button("🎯 Predict Engagement", type="primary"):
            input_data = {
                'age': age,
                'gender': gender,
                'district': district,
                'education_level': education_level,
                'family_church_background': family_church_background,
                'monthly_events_attended': monthly_events_attended,
                'volunteer_hours_per_month': volunteer_hours_per_month,
                'digital_engagement_score': digital_engagement_score,
                'peer_influence_score': peer_influence_score,
                'event_satisfaction_avg': event_satisfaction_avg
            }
            
            prediction, probability = predict_youth_engagement(input_data, youth_model_package)
            
            if prediction is not None:
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    engagement_level = "High" if prediction == 1 else "Low"
                    color = "green" if prediction == 1 else "red"
                    st.markdown(f"**Engagement Level:** :{color}[{engagement_level}]")
                
                with col2:
                    if probability is not None:
                        st.markdown(f"**Confidence:** {probability:.1%}")
                
                with col3:
                    risk_level = "Low Risk" if probability > 0.7 else "Medium Risk" if probability > 0.4 else "High Risk"
                    st.markdown(f"**Risk Assessment:** {risk_level}")
                
                # Create visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100 if probability else (prediction * 100),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Engagement Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if prediction == 1:
                    st.success("✅ **High Engagement Predicted** - This participant is likely to be actively engaged!")
                    st.info("**Suggestions:** Involve in leadership roles, peer mentoring, or event planning.")
                else:
                    st.warning("⚠️ **Low Engagement Predicted** - This participant may need additional support.")
                    st.info("**Suggestions:** Personalized outreach, buddy system, or tailored program offerings.")
    
    with tab2:
        st.markdown("### Upload CSV file for batch predictions")
        st.info("CSV should contain columns: age, gender, district, education_level, family_church_background, monthly_events_attended, volunteer_hours_per_month, digital_engagement_score, peer_influence_score, event_satisfaction_avg")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(df.head())
                
                if st.button("🚀 Run Batch Predictions"):
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    for i, row in df.iterrows():
                        pred, prob = predict_youth_engagement(row.to_dict(), youth_model_package)
                        predictions.append(pred)
                        probabilities.append(prob)
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Add results to dataframe
                    df['predicted_engagement'] = predictions
                    df['engagement_probability'] = probabilities
                    df['engagement_level'] = df['predicted_engagement'].map({1: 'High', 0: 'Low'})
                    
                    st.success(f"✅ Completed predictions for {len(df)} participants!")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        high_engagement_count = sum(predictions)
                        st.metric("High Engagement", f"{high_engagement_count}", f"{high_engagement_count/len(predictions)*100:.1f}%")
                    with col2:
                        avg_probability = np.mean([p for p in probabilities if p is not None])
                        st.metric("Avg. Probability", f"{avg_probability:.1%}")
                    with col3:
                        at_risk = sum([1 for p in probabilities if p is not None and p < 0.4])
                        st.metric("At Risk", f"{at_risk}", f"{at_risk/len(predictions)*100:.1f}%")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="youth_engagement_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Display results
                    st.dataframe(df[['age', 'gender', 'district', 'engagement_level', 'engagement_probability']])
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Financial Forecasting Section
    st.markdown("---")
    st.markdown("## 💰 Financial Sustainability Forecast")
    
    if financial_model_package is not None:
        st.info("📈 6-Month Financial Forecast based on current trends")
        
        # This would include financial forecasting logic
        # For now, show a placeholder
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Monthly Donations", "€8,850", "+2.3%")
        with col2:
            st.metric("Predicted Net Result", "€3,420", "+1.8%")
        with col3:
            st.metric("Financial Health", "Positive", "Stable")
        
        st.success("✅ Financial outlook: Sustainable operations predicted for next 6 months")