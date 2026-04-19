import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

# Import project-specific modules
from agents.graph import generate_care_plan
from utils.pdf_export import create_pdf

# --- PAGE SETUP ---
st.set_page_config(page_title="No-Show Predictor & Care Agent", page_icon="🏥", layout="wide")
st.title("🏥 Clinical Appointment No-Show Predictor & AI Care Agent")
st.markdown("""
Predict patient no-show risks and generate automated care coordination plans.
1. **Upload** data 2. **Analyze** risks 3. **Generate** agentic care strategies.
""")

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Kaggle Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # --- RUN PREDICTIONS ---
    st.write("---")
    if st.button("Run ML Risk Prediction", type="primary"):
        with st.spinner("Analyzing patient data with Decision Tree..."):
            try:
                # 1. Load Model & Scaler
                model = joblib.load('noshow_model.pkl')
                scaler = joblib.load('scaler.pkl')
                
                # 2. Preprocessing
                X = df.copy()
                X.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
                
                sched_dt = pd.to_datetime(X['ScheduledDay'])
                appt_dt = pd.to_datetime(X['AppointmentDay'])
                X['WaitDays'] = (appt_dt.dt.normalize() - sched_dt.dt.normalize()).dt.days
                X.loc[X['WaitDays'] < 0, 'WaitDays'] = 0
                X['Gender'] = X['Gender'].map({'M': 1, 'F': 0})
                
                expected_cols = ['Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'WaitDays']
                X = X[expected_cols]
                
                # 3. Scaling & Prediction
                X_scaled = scaler.transform(X)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                # 4. Store Results in Session State for Persistence
                df['No-Show Probability'] = (probabilities * 100).round(2)
                df['Risk Level'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in probabilities]
                st.session_state['results_df'] = df
                st.session_state['feature_importances'] = dict(zip(expected_cols, model.feature_importances_))
                
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"🚨 Prediction Error: {e}")

    # --- DISPLAY RESULTS & AGENTIC CARE ---
    if 'results_df' in st.session_state:
        res_df = st.session_state['results_df']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Prediction Results (Top 50)")
            def highlight_high_risk(val):
                return 'background-color: #ffcccc' if val == 'High Risk' else ''
            
            display_cols = ['Risk Level', 'No-Show Probability'] + [c for c in res_df.columns if c not in ['Risk Level', 'No-Show Probability']]
            st.dataframe(res_df[display_cols].head(50).style.map(highlight_high_risk, subset=['Risk Level']))

        with col2:
            st.subheader("🤖 Agentic Care Coordination")
            st.info("Select a patient to generate a custom Care Plan using LangGraph & Groq.")
            
            # Filter for high-risk patients to suggest coordination
            high_risk_patients = res_df[res_df['Risk Level'] == 'High Risk']
            
            if not high_risk_patients.empty:
                selected_idx = st.selectbox("Select Patient Index:", high_risk_patients.index)
                patient_row = res_df.loc[selected_idx]
                
                if st.button("Generate AI Care Plan"):
                    with st.spinner("Agent pipeline running (Analyze -> Intervene -> Compile)..."):
                        # Prepare data for Task 2 Agent
                        patient_data_dict = {
                        "Age": patient_row['Age'],
                        "Gender": 1 if patient_row['Gender'] == 'M' else 0,
                        "Scholarship": patient_row['Scholarship'],
                        # Use the column name exactly as it appears in the results dataframe
                        "Hypertension": patient_row['Hypertension'] if 'Hypertension' in patient_row else patient_row['Hipertension'],
                        "Diabetes": patient_row['Diabetes'],
                        "Alcoholism": patient_row['Alcoholism'],
                        "Handicap": patient_row['Handicap'] if 'Handicap' in patient_row else patient_row['Handcap'],
                        "SMS_received": patient_row['SMS_received'],
                        "WaitDays": patient_row.get('WaitDays', 0)
                        }
                        
                        # Task 2: Call LangGraph
                        agent_resp = generate_care_plan(
                            patient_data=patient_data_dict,
                            risk_score=float(patient_row['No-Show Probability']),
                            risk_level=patient_row['Risk Level'],
                            feature_importances=st.session_state['feature_importances']
                        )
                        
                        if agent_resp.get("error"):
                            st.error(agent_resp["error"])
                        else:
                            report = agent_resp["final_report"]
                            st.markdown("### 📄 AI Care Strategy")
                            st.write(report["risk_analysis"])
                            st.write(report["intervention_plan"])
                            
                            # Task 3: PDF Generation
                            pdf_output = create_pdf(report)
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_output,
                                file_name=f"CarePlan_Patient_{selected_idx}.pdf",
                                mime="application/pdf"
                            )
            else:
                st.write("No High Risk patients identified in this batch.")

        # Visualizations
        st.write("---")
        st.subheader("🔍 Global Feature Importance")
        importances = st.session_state['feature_importances']
        importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance')
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#4CAF50')
        st.pyplot(fig)

else:
    st.info("👈 Please upload the appointment CSV file in the sidebar to get started.")