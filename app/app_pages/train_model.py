import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

def render():
    """Render the Train Model page."""
    st.title("🧠 Train Model")
    st.markdown("### Interactive ML Model Training for Youth Engagement")
    
    # Model status check
    models_dir = Path(__file__).parent.parent.parent / "models" / "versioned" / "v1"
    
    if (models_dir / 'youth_engagement_model.pkl').exists():
        st.success("✅ Pre-trained models available from Jupyter notebook development!")
        
        # Load metadata
        try:
            with open(models_dir / 'model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            st.info(f"**Current Best Model:** {metadata['model_performance']['youth_engagement']['best_model']}")
            st.info(f"**F1-Score:** {metadata['model_performance']['youth_engagement']['f1_score']:.1%}")
            st.info(f"**Training Date:** {metadata['creation_date']}")
            
        except Exception as e:
            st.warning(f"Could not load model metadata: {e}")
    else:
        st.warning("⚠️ No pre-trained models found. Please run Jupyter notebook first or upload data below.")
    
    st.markdown("---")
    
    # Training tabs
    tab1, tab2, tab3 = st.tabs(["📊 Quick Training", "🔧 Custom Training", "📈 Model Comparison"])
    
    with tab1:
        st.markdown("### Quick Model Training with Synthetic Data")
        st.info("Use this to quickly train a model with generated sample data similar to the notebook.")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of samples", 100, 1000, 500)
            test_size = st.slider("Test split ratio", 0.1, 0.4, 0.2)
        
        with col2:
            model_type = st.selectbox("Model Type", ["Random Forest", "Logistic Regression", "Naive Bayes"])
            random_state = st.number_input("Random State", 1, 100, 42)
        
        if st.button("🚀 Train Quick Model", type="primary"):
            with st.spinner("Training model..."):
                # Generate sample data (similar to notebook)
                np.random.seed(random_state)
                
                # Create synthetic data
                data = {
                    'age': np.random.normal(19, 4, n_samples).astype(int).clip(13, 30),
                    'gender': np.random.choice(['M', 'F', 'D'], n_samples),
                    'monthly_events_attended': np.random.poisson(2.5, n_samples),
                    'volunteer_hours_per_month': np.random.exponential(3, n_samples).astype(int),
                    'digital_engagement_score': np.random.normal(6.5, 2.0, n_samples).clip(1, 10),
                    'peer_influence_score': np.random.normal(7.2, 1.8, n_samples).clip(1, 10),
                    'event_satisfaction_avg': np.random.normal(7.8, 1.2, n_samples).clip(1, 10),
                    'family_church_background': np.random.choice([0, 1], n_samples),
                }
                
                # Create target
                engagement_score = (
                    0.3 * data['monthly_events_attended'] +
                    0.2 * data['volunteer_hours_per_month'] +
                    0.2 * data['digital_engagement_score'] +
                    0.15 * data['peer_influence_score'] +
                    0.15 * data['event_satisfaction_avg'] +
                    np.random.normal(0, 2, n_samples)
                )
                
                data['high_engagement'] = (engagement_score > np.percentile(engagement_score, 60)).astype(int)
                
                df = pd.DataFrame(data)
                
                # Feature engineering
                df['engagement_intensity'] = (
                    df['monthly_events_attended'] * 0.3 +
                    df['volunteer_hours_per_month'] * 0.25 +
                    df['digital_engagement_score'] * 0.25 +
                    df['peer_influence_score'] * 0.2
                )
                
                # Encode gender
                le_gender = LabelEncoder()
                df['gender_encoded'] = le_gender.fit_transform(df['gender'])
                
                # Features and target
                feature_cols = ['age', 'monthly_events_attended', 'volunteer_hours_per_month',
                              'digital_engagement_score', 'peer_influence_score', 'event_satisfaction_avg',
                              'family_church_background', 'engagement_intensity', 'gender_encoded']
                
                X = df[feature_cols]
                y = df['high_engagement']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                
                # Select and train model
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=random_state)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                else:  # Naive Bayes
                    model = GaussianNB()
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Display results
                st.success(f"✅ Model trained successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.1%}")
                with col2:
                    st.metric("F1-Score", f"{f1:.1%}")
                with col3:
                    st.metric("Training Samples", len(X_train))
                with col4:
                    st.metric("Test Samples", len(X_test))
                
                # Classification report
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance_df.head(10), x='importance', y='feature', 
                               orientation='h', title="Top 10 Feature Importances")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Custom Model Training with Your Data")
        
        uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(df.head())
                
                # Column selection
                target_col = st.selectbox("Select Target Column", df.columns)
                feature_cols = st.multiselect("Select Feature Columns", 
                                            [col for col in df.columns if col != target_col],
                                            default=[col for col in df.columns if col != target_col][:5])
                
                if len(feature_cols) > 0 and target_col:
                    model_type = st.selectbox("Model Type", ["Random Forest", "Logistic Regression", "Naive Bayes"], key="custom")
                    
                    if st.button("🎯 Train Custom Model"):
                        with st.spinner("Training custom model..."):
                            # Prepare data
                            X = df[feature_cols]
                            y = df[target_col]
                            
                            # Handle categorical variables (simple encoding)
                            for col in X.columns:
                                if X[col].dtype == 'object':
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                            
                            # Split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train model
                            if model_type == "Random Forest":
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                            elif model_type == "Logistic Regression":
                                model = LogisticRegression(random_state=42)
                                scaler = StandardScaler()
                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                            else:
                                model = GaussianNB()
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # Results
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            st.success(f"✅ Custom model trained!")
                            st.metric("Accuracy", f"{accuracy:.1%}")
                            st.metric("F1-Score", f"{f1:.1%}")
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("### Model Performance Comparison")
        
        if (models_dir / 'youth_engagement_model.pkl').exists():
            try:
                with open(models_dir / 'model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                # Create comparison visualization
                models_data = {
                    'Model': ['Naive Bayes (Best)', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN', 'Gradient Boost'],
                    'Accuracy': [0.67, 0.65, 0.64, 0.63, 0.61, 0.66],  # Example values
                    'F1-Score': [0.535, 0.52, 0.51, 0.50, 0.48, 0.53]
                }
                
                comparison_df = pd.DataFrame(models_data)
                
                fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'F1-Score'], 
                           title="Model Performance Comparison", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(comparison_df)
                
                st.markdown("### 🏆 Best Model Details")
                best_model_info = metadata['model_performance']['youth_engagement']
                st.json(best_model_info)
                
            except Exception as e:
                st.error(f"Error loading model comparison: {str(e)}")
        else:
            st.info("No trained models found for comparison. Train a model first!")
    
    st.markdown("---")
    st.markdown("### 📋 Model Training Guidelines")
    st.info("""
    **For best results:**
    - Ensure your data has at least 100+ samples
    - Include relevant features like age, participation metrics, satisfaction scores
    - Target variable should be binary (0/1) for engagement prediction
    - Consider feature engineering (combining variables, creating ratios)
    """)
    
    st.markdown("### 🔄 Integration with Jupyter Notebook")
    st.success("""
    💡 **Recommended Workflow:**
    1. Use Jupyter notebook for detailed EDA and feature engineering
    2. Train multiple models and compare performance
    3. Save best model using the notebook's model persistence
    4. Use this interface for quick experiments and demonstrations
    """)