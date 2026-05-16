import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import firebase_admin
import extra_streamlit_components as argostick
import os
import io
import re
import html

from firebase_admin import credentials, auth, firestore
from sklearn.metrics import roc_curve, roc_auc_score
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
from dotenv import load_dotenv


# =========================
# FIREBASE INITIALIZATION
# =========================

# Load variables from .env
load_dotenv()

if not firebase_admin._apps:
    firebase_config = {
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
    }
    
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# =========================
# SESSION STATE MANAGEMENT
# =========================
if 'user_auth' not in st.session_state:
    st.session_state['user_auth'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'Login'


# Initialize the Cookie Manager
cookie_manager = argostick.CookieManager()

def logout():
    # 1. Clear the session state completely
    st.session_state.clear()
    
    # 2. SAFELY delete cookies only if they exist to avoid KeyError
    try:
        # Check if cookie manager is initialized and has the key
        if cookie_manager.get("user_auth_cookie"):
            cookie_manager.delete("user_auth_cookie")
        if cookie_manager.get("username_cookie"):
            cookie_manager.delete("username_cookie")
    except Exception:
        # Ignore errors if cookies are already gone
        pass
    
    # 3. Re-initialize minimal state for the login page
    st.session_state['auth_mode'] = 'Login'
    st.session_state['user_auth'] = None
    
    # 4. Final refresh
    st.rerun()

    
# =========================
# AUTHENTICATION UI
# =========================
def is_valid_email(email):
    # RFC 5322 Compliant Email Regex Validation
    email_regex = r"^[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$"
    return re.match(email_regex, email.strip()) is not None

def sanitize_string(text):
    # Strip whitespace and escape HTML tags to prevent XSS payloads in Usernames
    if text:
        cleaned = text.strip()
        return html.escape(cleaned)
    return ""

def auth_ui():
    st.title("Hypertension Predictor - Access")

    if st.session_state['auth_mode'] == 'Login':
        st.subheader("Login")
        email = st.text_input("Email Address").strip()
        password = st.text_input("Password", type='password')

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Login", key="login_btn_main"):
                if not email or not password:
                    st.error("Please enter both email and password.")
                elif not is_valid_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    try:
                        # Use sanitized email string
                        user = auth.get_user_by_email(email)
 
                        st.session_state['username'] = user.display_name if user.display_name else email.split('@')[0]
                        st.session_state['user_auth'] = user.email
                        st.success("Log in successful!")
                        time.sleep(1)
                        st.rerun()
 
                    except Exception as e:
                        # Generic error prevents attackers from discovering registered emails via brute force
                        st.error("Log in failed. Please verify your credentials.")
 
        st.write("Do not have an account?")
        if st.button("Sign Up", key="go_to_signup_btn"):
            st.session_state['auth_mode'] = 'Signup'
            st.rerun()

    else: # Sign-up mode
        st.subheader("Create Account")
        new_user = st.text_input("Username", key="reg_user")
        new_email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type='password', key="reg_pass")
 
        if st.button("Create Account", key="signup_main_btn"):
            # Sanitize inputs instantly before checking logic rules
            clean_user = sanitize_string(new_user)
            clean_email = new_email.strip()
            
            if not clean_user or not clean_email or not new_password:
                st.error("All credential fields are required.")
            elif len(clean_user) < 3 or len(clean_user) > 20:
                st.error("Username must be between 3 and 20 characters long.")
            elif not is_valid_email(clean_email):
                st.error("Please enter a valid email address.")
            elif len(new_password) < 8:
                st.error("For security, passwords must be at least 8 characters long.")
            else:
                try:
                    user = auth.create_user(
                        email=clean_email,
                        password=new_password,
                        display_name=clean_user
                    )
                    st.success("✅ Account created successfully! Redirecting to login...")
                    time.sleep(2)
                    st.session_state['auth_mode'] = 'Login'
                    st.rerun()
                except Exception as e:
                    error_msg = str(e)
                    # Abstract infrastructure details so system errors aren't leaked
                    if "EMAIL_EXISTS" in error_msg or "already in use" in error_msg:
                        st.error("This email address is already registered.")
                    else:
                        st.error("Registration failed. Please contact support if this persists.")

        if st.button("Back to Login", key="back_to_login"):
            st.session_state['auth_mode'] = 'Login'
            st.rerun()
            
# =========================
# MAIN APP CONTENT
# =========================
if st.session_state['user_auth'] is None:
    auth_ui()
else:
    # Sidebar Logout
    st.sidebar.write(f"Logged in as: **{st.session_state['user_auth']}**")
    if st.sidebar.button("Logout"):
        logout()
        
    # =========================
    # SIDEBAR - ABOUT SECTION
    # =========================
    with st.sidebar:
        st.divider() # Adds a clean visual line
        with st.expander("About The Prediction Logic"):
            st.markdown("""
            This system uses a **Cascaded ML Model Architecture** to ensure high accuracy and reliability:
            
            1. **Screening: (Logistic Regression)**  
            The app first analyzes your data using a linear model. If the probability of hypertension is very low or very high, it provides an immediate result.
            
            2. **Deep Analysis: (Random Forest)**  
            If the initial screening finds the case "uncertain" (near the 50% threshold), it automatically triggers a Random Forest model. This model looks for complex patterns and non-linear relationships in your health data.
            
            **Goal:** It uses two rounds of checking to catch tricky cases that might otherwise be missed — similar to how a doctor might seek a second opinion when a diagnosis isn't clear-cut..
            """)
        
        st.caption("v1.0.2 | Secure Model")


    # --- YOUR ORIGINAL APP LOGIC STARTS HERE ---
    
    # Load models (Inside the 'else' so they don't load unnecessarily on auth page)
    @st.cache_resource
    def load_resources():
        p = pickle.load(open("pipeline.pkl", "rb"))
        l = pickle.load(open("log_model.pkl", "rb"))
        r = pickle.load(open("rf_model.pkl", "rb"))
        m = pickle.load(open("model_metrics.pkl", "rb"))
        return p, l, r, m

    pipeline, log_model, rf_model, metrics = load_resources()

    st.set_page_config(page_title="Hypertension Predictor", layout="wide") 
    st.title("Hypertension Prediction System")
    

    selected = option_menu (
        menu_title=None,
        options=["Home", "Patient Data", "Data Visualization", "Recommendations"],
        icons=["house-fill", "calendar2-range", "bar-chart-fill", "heart-pulse-fill"],
        orientation="horizontal",
    )
    
    
    if selected == "Home":
        # Get the username from session state, default to "User" if not found
        current_user = st.session_state.get('username', 'User')
      
        # 1. CSS for Stylish Cards with Black Text
        st.markdown("""
        <style>
        .main-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #FF4B4B;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            color: #000000 !important; /* Forces body text to black */
        }
        .feature-header {
            color: #000000 !important; /* Forces header text to black */
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 1.2rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # 2. Hero Section
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            # This line now displays your custom username
            st.markdown(f"# Welcome, {current_user}! 👋")
        
            st.markdown("""
            ### Empowering Healthcare Through Machine Learning and AI.
            Welcome to the **Hypertension Prediction System**. This platform uses advanced machine learning to help identify high blood pressure risks early. Navigate through the tabs to input patient data, visualize risk factors, and receive AI-driven health recommendations.
            """)
        
        
        st.divider()

        # 3. Info Cards
        st.markdown("### Platform Capabilities")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""
            <div class="main-card">
                <div class="feature-header">🔍 Accurate Prediction</div>
                Cascaded ML model logic (Logistic Regression + Random Forest) for high-precision diagnosis.
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="main-card">
                <div class="feature-header">📊 Visual Analytics</div>
                Interactive Radar and Gauge charts to understand the 'Why' behind every risk score.
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="main-card">
                <div class="feature-header">📄 Professional Reports</div>
                Instant PDF generation for medical records, including patient metrics and clinical notes.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 4. Global Statistics Banner
        st.markdown("### Why it Matters")
        m1, m2, m3 = st.metrics_adc = st.columns(3)
        m1.metric("Global Prevalence", "1.28 Billion", "+10%")
        m2.metric("Awareness Gap", "46%", "High Risk")
        m3.metric("System Accuracy", "94.2%", "Model v1.0.2")
        
        
        st.success("To Get Started, Go To The Patient Data Tab")
                
        st.markdown("""
        ---
        ⚠️ **Medical Disclaimer**

        This system is an AI-assisted tool and should not replace professional medical diagnosis.
        Always consult a qualified healthcare provider.
        """)
            
    
    if selected == "Patient Data":
        st.markdown("### Enter Your Details ")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 1, 120)
            salt = st.slider("Salt Intake", 0, 10)
            stress = st.slider("Stress Score", 0, 10)
            sleep = st.slider("Sleep Duration", 0, 12)
            bmi = st.slider("BMI", 10.0, 50.0)

        with col2:
            bp = st.selectbox("BP History", ["Normal", "Hypertensive", "Prehypertensive"])
            med = st.selectbox("Medication", ["None", "ACE Inhibitor", "BETA Blocker", "Diuretic", "Other"])
            fam = st.selectbox("Family History", ["Yes", "No"])
            ex = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
            smoke = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])

        # --- PREDICTION LOGIC ---
        if st.button("Prediction"): 
            input_data = pd.DataFrame([{
                'Age': age, 'Salt_Intake': salt, 'Stress_Score': stress,
                'BP_History': bp, 'Sleep_Duration': sleep, 'BMI': bmi,
                'Medication': med, 'Family_History': fam,
                'Exercise_Level': ex, 'Smoking_Status': smoke
            }])
            
            input_data_processed = pipeline.transform(input_data)
            prob = log_model.predict_proba(input_data_processed)[0][1]

            if prob >= 0.7 or prob <= 0.3:
                prediction = log_model.predict(input_data_processed)[0]
                model_used = "Logistic Regression"
                current_auc = metrics["log_auc"]
            else:
                prediction = rf_model.predict(input_data_processed)[0]
                model_used = "Random Forest"
                current_auc = metrics["rf_auc"]
                
            # SAVE EVERYTHING TO SESSION STATE
            st.session_state["input_data"] = input_data
            st.session_state["prediction"] = prediction
            st.session_state["probability"] = prob  
            st.session_state["model_used"] = model_used
            st.session_state["current_auc"] = current_auc

        # --- DISPLAY RESULTS (Persist even after PDF generation) ---
        if "prediction" in st.session_state:
            st.divider()
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.subheader("Diagnosis Result")
                if st.session_state["prediction"] == 1:
                    st.error("⚠️ HIGH RISK: Hypertension Detected")
                else:
                    st.success("✅ LOW RISK: No Hypertension")
                
                st.info(f"MODEL USED: {st.session_state['model_used']}")
                st.metric("Model Confidence", f"{st.session_state['probability']:.2%}")
                st.metric("Model Reliability (AUC)", f"{st.session_state['current_auc']:.2f}")

            with col_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = st.session_state["probability"] * 100,
                    title = {'text': "Risk Probability %"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

        # --- PDF EXPORT FUNCTION ---
        def export_pdf_report(input_data, prediction, probability):
            # 1. Create an in-memory buffer
            buffer = io.BytesIO()
            
            # 2. Build the PDF inside the buffer instead of a file
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>Hypertension Medical Report</b>", styles['Title']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 20))

            data = [
                ["Parameter", "Value"],
                ["Age", input_data['Age'].values[0]],
                ["Salt Intake", input_data['Salt_Intake'].values[0]],
                ["Stress Score", input_data['Stress_Score'].values[0]],
                ["BP History", input_data['BP_History'].values[0]],
                ["Sleep Duration", input_data['Sleep_Duration'].values[0]],
                ["BMI", input_data['BMI'].values[0]],
                ["Medication", input_data['Medication'].values[0]],
                ["Family History", input_data['Family_History'].values[0]],
                ["Exercise Level", input_data['Exercise_Level'].values[0]],
                ["Smoking Status", input_data['Smoking_Status'].values[0]],
            ]

            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 20))

            diag_text = "<font color='red'><b>HIGH RISK: Hypertension Detected</b></font>" if prediction == 1 else "<font color='green'><b>LOW RISK: No Hypertension</b></font>"
            elements.append(Paragraph(f"Diagnosis Result: {diag_text}", styles['Heading2']))
            elements.append(Paragraph(f"Prediction Confidence: {probability:.2f}", styles['Normal']))
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("This report is generated by an AI-based hypertension prediction system. Please consult a medical professional for clinical diagnosis.", styles['Italic']))

            # 3. Build the document
            doc.build(elements)
            
            # 4. Get the value from the buffer and return it
            pdf_value = buffer.getvalue()
            buffer.close()
            return pdf_value

        # --- EXPORT SECTION ---
        if "prediction" in st.session_state:
        # 1. Prepare the PDF data in the background
            pdf_data = export_pdf_report(
                st.session_state["input_data"],
                st.session_state["prediction"],
                st.session_state["probability"]
            )

            # 2. Show the real download button
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"Hypertension_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
            
            # 3. Show the success message
            st.info("PDF report is ready for download!")

        else:
            # If no prediction exists, show a regular button that flags the error
            if st.button("Download PDF Report"):
                st.error("⚠️ Please run the prediction first to generate your report!")
            
                    
            
    if selected == "Data Visualization":
        if "input_data" in st.session_state:
            # 1. EXTRACT DATA
            data = st.session_state["input_data"]
            v_age = data['Age'].iloc[0]
            v_salt = data['Salt_Intake'].iloc[0]
            v_stress = data['Stress_Score'].iloc[0]
            v_bmi = data['BMI'].iloc[0]
            v_sleep = data['Sleep_Duration'].iloc[0]
            
            # ROW 1: Bar Chart and Pie Chart
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.markdown("#### Patient Metrics vs. High-Risk Thresholds")
                metrics_df = pd.DataFrame({
                    'Metric': ['Salt Intake', 'Stress Score', 'BMI (Scaled)', 'Sleep (Inverse)'],
                    'Patient Value': [v_salt, v_stress, v_bmi/5, 12-v_sleep],
                    'Risk Threshold': [7, 7, 6, 5] 
                })
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(name='Patient Value', x=metrics_df['Metric'], y=metrics_df['Patient Value'], marker_color='#FF4B4B'))
                fig_bar.add_trace(go.Bar(name='Risk Threshold', x=metrics_df['Metric'], y=metrics_df['Risk Threshold'], marker_color='#E6E9EF'))
                fig_bar.update_layout(barmode='group', height=350, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_v2:
                st.markdown("#### Risk Factor Contribution")
                labels = ['Salt', 'Stress', 'BMI', 'Sleep Lack']
                values = [v_salt, v_stress, v_bmi/5, (12-v_sleep)]
                fig_pie = px.pie(values=values, names=labels, hole=0.4, 
                                color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            # ROW 2: Scatter Plot and Radar Chart (Side-by-Side)
            st.divider() 
            col_v3, col_v4 = st.columns(2)

            with col_v3:
                st.markdown("#### Risk Positioning (Age vs. BMI)")
                scatter_data = pd.DataFrame({
                    'Age': [20, 40, 60, 80, v_age],
                    'BMI': [20, 25, 30, 35, v_bmi],
                    'Type': ['Reference', 'Reference', 'Reference', 'Reference', 'YOU']
                })
                fig_scatter = px.scatter(scatter_data, x="Age", y="BMI", color="Type",
                                        size=[10,10,10,10,20], 
                                        color_discrete_map={'Reference': 'lightgrey', 'YOU': '#FF4B4B'})
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_v4:
                st.markdown("#### Patient Risk Factor Breakdown")
                # Normalizing inputs for consistent Radar scaling
                risk_factors = {
                    'Salt Intake': v_salt,
                    'Stress': v_stress,
                    'BMI': (v_bmi / 50) * 10,
                    'Sleep (Lack of)': 12 - v_sleep
                }
                
                fig_radar = px.line_polar(
                    r=list(risk_factors.values()),
                    theta=list(risk_factors.keys()),
                    line_close=True,
                    range_r=[0, 10],
                    color_discrete_sequence=['#FF4B4B']
                )
                fig_radar.update_traces(fill='toself')
                fig_radar.update_layout(height=400)
                st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.warning("⚠️ Please go to the **Patient Data** tab and click 'Prediction' to generate your personalized charts.")
        
            
    if selected == "Recommendations":
        st.markdown("""
        <style>
        .recommendation-card {
            background-color: #f8f9fa;
            padding: 40px;
            border-radius: 15px;
            border-left: 10px solid #FF4B4B; /* Thicker red border */
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            color: #000000 !important; /* Black text */
            min-height: 200px;
        }
        
        .recommendation-title {
            color: #000000 !important;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            font-size: 2rem;
        }

        .slide-text {
            text-align: justify;
            font-size: 1.15rem;
            line-height: 1.7;
            color: #000000 !important;
        }
        
        /* Navigation buttons styling */
        div.stButton > button {
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        div.stButton > button:hover {
            border-color: #FF4B4B;
            color: #FF4B4B;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 1. Setup the content
        slides = [
            {"title": "Breakdown", "content": "This is a detailed breakdown of the most critical health tips for hypertension, categorized by preventive and corrective measures."},
            {"title": "Dietary Management", "content": "The most effective dietary intervention is the DASH - Dietary Approaches to Stop Hypertension diet. It focuses on nutrient-dense foods that naturally lower BP. Limit daily sodium intake. Avoid salty foods. Potassium helps the kidneys excrete sodium and eases tension in blood vessel walls. Incorporate bananas, sweet potatoes, spinach, and beans. Ensure adequate intake of low-fat dairy and nuts, rich in magnesium and calcium which support vascular health."},
            {"title": "Regular Exercise", "content": "Weight and blood pressure have a linear relationship; as weight increases, BP typically follows. Aim for at least 150 minutes of moderate-intensity aerobic activity per week e.g., brisk walking, swimming, or cycling. Incorporate physical strength training atleast 2–3 times a week."},
            {"title": "Substance Moderation", "content": "Chemical stimulants and depressants have a profound impact on arterial pressure. Every cigarette causes a temporary spike in BP and in some cases caffeine too. Long-term smoking destroys the lining of the artery walls, leading to atherosclerosis. Excessive alcohol can raise BP and reduce the effectiveness of many medications."},
            {"title": "Stress Mitigation", "content": "Stress is a Silent Trigger. Chronic stress keeps the body in a fight or flight state, producing hormones like cortisol and adrenaline that constrict blood vessels. Observe sleep hygiene. Aim for 7–9 hours of quality sleep."},
            {"title": "Clinical Adherence", "content": "When lifestyle changes aren't enough, medical intervention becomes the corrective necessity. Take prescribed antihypertensives like ACE inhibitors, Beta-blockers, Diuretics, etc at the same time every day. Never skip doses even if you feel fine. Keep a BP log. Regular screenings for the target organs — the heart, brain, kidneys, and eyes are essential to catch early signs of organ strain."},
            {"title": "Medical Note", "content": "Always consult with your primary care physician before starting a new exercise regimen or making drastic changes to your diet, especially if you are already on prescribed medication."}
        ]

        # 2. Initialize Session State
        if 'slide_idx' not in st.session_state:
            st.session_state.slide_idx = 0
        if 'auto_play' not in st.session_state:
            st.session_state.auto_play = True
        
        # 3. Create the Slide "Card"
        @st.fragment(run_every=10.0 if st.session_state.get('auto_play') else None)
        def slide_viewer():
            # Current Slide Content wrapped in the new styled div
            curr = slides[st.session_state.slide_idx]
            
            st.markdown(f"""
                <div class="recommendation-card">
                    <div class="recommendation-title">{curr['title']}</div>
                    <div class="slide-text">{curr['content']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation Row (Outside the gray card for better visual separation)
            col1, col2, col3 = st.columns([1, 2, 1])
            
            if col1.button("Previous", use_container_width=True):
                st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(slides)
                st.rerun()
                
            with col2:
                st.markdown(f"<p style='text-align: center; font-weight: bold;'>Slide {st.session_state.slide_idx + 1} of {len(slides)}</p>", unsafe_allow_html=True)
                
            if col3.button("Next", use_container_width=True):
                st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(slides)
                st.rerun()

            # Logic to advance slide automatically
            if st.session_state.auto_play:
                st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(slides)

        # Run the app
        slide_viewer()

        # Toggle Auto-play
        st.toggle("Auto-play", key="auto_play")
        
    
