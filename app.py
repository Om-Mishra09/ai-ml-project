import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="No-Show Predictor", page_icon="🏥", layout="wide")
st.title("🏥 Clinical Appointment No-Show Predictor")
st.markdown("Upload patient appointment data to predict the risk of a no-show. This helps clinics prioritize interventions and reduce wasted time.")

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Kaggle Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head()) # Shows the first 5 rows

    # --- RUN PREDICTIONS ---
    st.write("---")
    if st.button("Run Predictions", type="primary"):
        with st.spinner("Analyzing patient data with the Decision Tree model..."):
            
            try:
                import joblib
                
                # 1. LOAD THE TRAINED MODEL AND SCALER
                model = joblib.load('noshow_model.pkl')
                scaler = joblib.load('scaler.pkl') # NEW: Load the scaler
                
                # 2. PREPROCESS THE UPLOADED DATA
                X = df.copy()
                
                # Align column typos to match the training script
                X.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
                
                # Calculate WaitDays
                sched_dt = pd.to_datetime(X['ScheduledDay'])
                appt_dt = pd.to_datetime(X['AppointmentDay'])
                X['WaitDays'] = (appt_dt.dt.normalize() - sched_dt.dt.normalize()).dt.days
                
                # Prevent negative WaitDays (fixes anomalies in new uploaded data)
                X.loc[X['WaitDays'] < 0, 'WaitDays'] = 0
                
                # Encode Gender (Must match training script exactly: M=1, F=0)
                X['Gender'] = X['Gender'].map({'M': 1, 'F': 0})
                
                # Enforce the exact column order the Scaler expects
                expected_cols = ['Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'WaitDays']
                X = X[expected_cols]
                
                # 3. SCALE THE DATA
                X_scaled = scaler.transform(X)
                
                # 4. RUN REAL PREDICTIONS ON SCALED DATA
                probabilities = model.predict_proba(X_scaled)[:, 1] 
                
                # Assign results back to the original dataframe for display
                df['No-Show Probability'] = (probabilities * 100).round(2).astype(str) + '%'
                df['Risk Level'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in probabilities]
                
                st.success("Analysis Complete!")
                
                # --- DISPLAY PREDICTIONS ---
                st.subheader("Prediction Results")
                
                def highlight_high_risk(val):
                    color = '#ffcccc' if val == 'High Risk' else ''
                    return f'background-color: {color}'
                
                # Display the results
                cols = ['Risk Level', 'No-Show Probability'] + [c for c in df.columns if c not in ['Risk Level', 'No-Show Probability']]
                st.dataframe(df[cols].head(50).style.map(highlight_high_risk, subset=['Risk Level']))

                # --- DISPLAY REAL FEATURE IMPORTANCE ---
                st.write("---")
                st.subheader("Key Contributing Factors")
                st.markdown("These are the actual factors driving the model's predictions based on the Decision Tree:")
                
                importances = model.feature_importances_
                feature_names = X.columns
                
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values(by='Importance', ascending=True) 
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(importance_df['Feature'], importance_df['Importance'], color='#4CAF50')
                ax.set_xlabel('Importance Score')
                ax.set_title('Decision Tree Feature Importances')
                st.pyplot(fig)
                
            except FileNotFoundError:
                st.error("🚨 Error: 'noshow_model.pkl' or 'scaler.pkl' not found! Make sure both files are in the same folder as this script.")
            except Exception as e:
                st.error(f"🚨 An error occurred during prediction: {e}")
else:
    st.info("👈 Please upload the appointment CSV file in the sidebar to get started.")