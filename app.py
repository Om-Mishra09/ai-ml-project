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
                
                # 1. LOAD THE TRAINED MODEL
                # Ensure 'noshow_model.pkl' is in the same directory as this script
                model = joblib.load('noshow_model.pkl')
                
                # 2. PREPROCESS THE UPLOADED DATA (Must match Colab training exactly)
                # We make a copy so we don't ruin the original df used for displaying
                X = df.copy()
                
                # Calculate LeadDays
                # Calculate LeadDays (Safe Datetime handling)
                sched_dt = pd.to_datetime(X['ScheduledDay'])
                appt_dt = pd.to_datetime(X['AppointmentDay'])
                
                # Normalize removes the time (sets to midnight) so we only calculate the difference in pure days
                X['LeadDays'] = (appt_dt.dt.normalize() - sched_dt.dt.normalize()).dt.days
                # Encode Gender
                X['Gender'] = X['Gender'].map({'M': 0, 'F': 1})
                
                # Drop columns the model hasn't seen
                cols_to_drop = ['PatientId', 'AppointmentID', 'Neighbourhood', 'ScheduledDay', 'AppointmentDay']
                # If the Kaggle dataset still has the answer key, drop it so we don't cheat!
                if 'No-show' in X.columns:
                    cols_to_drop.append('No-show')
                
                X = X.drop(columns=cols_to_drop, errors='ignore')
                
                # 3. RUN REAL PREDICTIONS
                # predict_proba gets the actual % confidence, [:, 1] gets the probability of class 1 (No-Show)
                probabilities = model.predict_proba(X)[:, 1] 
                
                # Assign results back to the original dataframe for display
                df['No-Show Probability'] = (probabilities * 100).round(2).astype(str) + '%'
                df['Risk Level'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in probabilities]
                
                st.success("Analysis Complete!")
                
                # --- DISPLAY PREDICTIONS ---
                st.subheader("Prediction Results")
                
                def highlight_high_risk(val):
                    color = '#ffcccc' if val == 'High Risk' else ''
                    return f'background-color: {color}'
                
                cols = ['Risk Level', 'No-Show Probability'] + [c for c in df.columns if c not in ['Risk Level', 'No-Show Probability']]
                st.dataframe(df[cols].head(50).style.map(highlight_high_risk, subset=['Risk Level']))

                # --- DISPLAY REAL FEATURE IMPORTANCE ---
                st.write("---")
                st.subheader("Key Contributing Factors")
                st.markdown("These are the actual factors driving the model's predictions based on the Decision Tree:")
                
                # Extract real feature importances from the loaded model
                importances = model.feature_importances_
                feature_names = X.columns
                
                # Create a dataframe and sort it
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values(by='Importance', ascending=True) # Ascending for horizontal bar chart
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(importance_df['Feature'], importance_df['Importance'], color='#4CAF50')
                ax.set_xlabel('Importance Score')
                ax.set_title('Decision Tree Feature Importances')
                st.pyplot(fig)
                
            except FileNotFoundError:
                st.error("🚨 Error: 'noshow_model.pkl' not found! Make sure the Data Scientist's model file is in the same folder as this script.")
            except Exception as e:
                st.error(f"🚨 An error occurred during prediction: {e}")
else:
    st.info("👈 Please upload the appointment CSV file in the sidebar to get started.")