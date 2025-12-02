# complete_medfusion_system.py
"""
MEDFUSION AI - Advanced Multi-Modal Medical Diagnostic System
Professional-grade integrated platform for ECG, echocardiography, and multisensor fusion.
Disclaimer: Research prototype only — not for clinical or diagnostic use.
"""
# -------------------------
# Imports & Dependencies
# -------------------------
import math
import time
import uuid
import json
import base64
import io
import threading
import tempfile
import cv2
import librosa
import wave
import pydub
from collections import deque, defaultdict
from datetime import datetime, timedelta
import random
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal, stats, fft
from scipy.interpolate import interp1d
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
from PIL import Image, ImageOps
# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, db, storage
    from firebase_admin.exceptions import FirebaseError
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    st.warning("Firebase Admin SDK not available. Install with: pip install firebase-admin")

# Mock UploadedFile for loading sample files
class MockUploadedFile:
    def __init__(self, bytes_data, name, type_):
        self._bytes = bytes_data
        self.name = name
        self.type = type_

    def getvalue(self):
        return self._bytes

# -------------------------
# Global Configuration
# -------------------------
APP_TITLE = "MEDFUSION AI — Integrated Medical Diagnostic Platform"
VERSION = "v2.0-integrated"
# Color scheme
PRIMARY = "#0b6fab"
ACCENT = "#1fb6ff"
DANGER = "#e74c3c"
WARNING = "#f39c12"
SUCCESS = "#2ecc71"
BACKGROUND = "#f4fbff"
CARD = "#ffffff"
TEXT = "#1f2d3d"
SECONDARY = "#6b7280"
# Sampling rates
ECG_SR = 250
PPG_SR = 100
PCG_SR = 4000
DOP_SR = 100
RESP_SR = 25
# Buffer settings
BUFFER_SECONDS = 120
TREND_WINDOW = 300

# Sample files paths (create a 'samples' directory in the same folder as this script)
SAMPLE_DIR = 'samples'
SAMPLE_ECHO = os.path.join(SAMPLE_DIR, r'C:/Users/prave/Downloads/auscsync_package/Apical_4_chamber_view.png')  # Sample echo image
SAMPLE_ECG = os.path.join(SAMPLE_DIR, r'C:/Users/prave/Downloads/auscsync_package/12_Lead_EKG_ST_Elevation_tracing_color_coded.jpg')    # Sample ECG image
SAMPLE_AUDIO = os.path.join(SAMPLE_DIR, r'C:/Users/prave/Downloads/auscsync_package/Phonocardiograms_from_normal_and_abnormal_heart_sounds.png')  # Sample heart sound

# -------------------------
# Utility Functions
# -------------------------
def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return v
def safe_mean(a):
    try:
        return float(np.mean(a))
    except Exception:
        return 0.0
def formatf(x, fmt="{:.2f}"):
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return fmt.format(x)
    except Exception:
        return str(x)
def generate_synthetic_dataset(n_samples=10000, seed=42):
    """Generate synthetic dataset for ML training"""
    rng = np.random.RandomState(seed)
    rows = []
    labels = []
    feature_names = ['age', 'hr', 'spo2', 'perf', 'flow', 'ecg_irreg', 'hrv_rmssd', 'resp_rate']
   
    for _ in range(n_samples):
        age = int(rng.randint(0, 90))
        if age <= 1:
            hr_base = rng.normal(120, 10)
        elif age <=5:
            hr_base = rng.normal(110, 10)
        elif age <=12:
            hr_base = rng.normal(90, 8)
        elif age <=17:
            hr_base = rng.normal(75, 8)
        elif age <=50:
            hr_base = rng.normal(75, 10)
        elif age <=65:
            hr_base = rng.normal(75, 12)
        else:
            hr_base = rng.normal(70, 12)
        hr = float(clamp(hr_base + rng.randn()*6, 30, 220))
        spo2 = float(clamp(96 + rng.randn()*2, 60, 100))
        perf = float(clamp(rng.normal(0.08, 0.04), 0.001, 1.0))
        flow = float(clamp(rng.normal(0.9, 0.25), 0.01, 1.5))
        ecg_irreg = float(clamp(rng.beta(1, 12), 0.0, 1.0))
        hrv_rmssd = float(clamp(rng.normal(40, 20), 5, 200))
        resp_rate = float(clamp(rng.normal(16, 4), 8, 40))
        label = 0
        if rng.rand() < 0.08:
            spo2 -= rng.uniform(2, 8)
            hr += rng.uniform(10, 40)
            perf *= rng.uniform(0.2, 0.8)
            hrv_rmssd *= rng.uniform(0.5, 0.8)
            resp_rate += rng.uniform(5, 15)
            label = 1
        if rng.rand() < 0.03:
            spo2 -= rng.uniform(6, 25)
            flow *= rng.uniform(0.01, 0.5)
            ecg_irreg = max(ecg_irreg, rng.uniform(0.6, 1.0))
            hrv_rmssd *= rng.uniform(0.3, 0.6)
            resp_rate += rng.uniform(10, 25)
            label = 2
           
        rows.append([age, hr, spo2, perf, flow, ecg_irreg, hrv_rmssd, resp_rate])
        labels.append(label)
       
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32), feature_names
@st.cache_resource(show_spinner=False)
def build_and_train_rf(n_samples=10000):
    """Build and train Random Forest model"""
    X, y, feature_names = generate_synthetic_dataset(n_samples=n_samples)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
   
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
   
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
   
    y_proba = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 2], pos_label=2)
    roc_auc = auc(fpr, tpr)
   
    return {
        'model': model,
        'scaler': scaler,
        'train_score': train_score,
        'test_score': test_score,
        'feature_names': feature_names,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
RF_BUNDLE = build_and_train_rf(10000)
# -------------------------
# Signal Processing Functions
# -------------------------
def butter_bandpass(x, lowcut, highcut, fs, order=3):
    """Butterworth bandpass filter"""
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, x)
    except Exception:
        return x
def estimate_hr_ecg(ecg_array, fs=ECG_SR):
    """Estimate heart rate from ECG signal"""
    if len(ecg_array) < int(0.6 * fs):
        return 0.0, [], {}
   
    try:
        filt = butter_bandpass(np.array(ecg_array), 5.0, 40.0, fs, order=2)
        thr = np.mean(filt) + 0.4 * np.std(filt)
        peaks, properties = signal.find_peaks(filt, distance=int(0.3 * fs), height=thr, prominence=0.5)
       
        if len(peaks) < 2:
            return 0.0, peaks.tolist(), {}
       
        rr = np.diff(peaks) / float(fs)
        hr = 60.0 / float(np.mean(rr))
       
        hrv_features = {
            'mean_rr': float(np.mean(rr)),
            'std_rr': float(np.std(rr)),
            'rmssd': float(np.sqrt(np.mean(np.diff(rr) ** 2))),
            'nn50': int(np.sum(np.abs(np.diff(rr)) > 0.05)),
            'pnn50': float(np.sum(np.abs(np.diff(rr)) > 0.05) / len(rr) * 100) if len(rr) > 0 else 0.0
        }
       
        return float(hr), peaks.tolist(), hrv_features
    except Exception:
        return 0.0, [], {}
# -------------------------
# Clinical Decision Support
# -------------------------
AGE_GROUPS = [
    {'name': 'Neonate', 'min':0, 'max':0.1, 'hr_min':100, 'hr_max':160, 'spo2_min':88, 'resp_min':30, 'resp_max':60},
    {'name': 'Infant', 'min':0.1, 'max':1, 'hr_min':100, 'hr_max':160, 'spo2_min':95, 'resp_min':30, 'resp_max':60},
    {'name': 'Toddler', 'min':1, 'max':3, 'hr_min':90, 'hr_max':150, 'spo2_min':95, 'resp_min':24, 'resp_max':40},
    {'name': 'Preschool', 'min':3, 'max':6, 'hr_min':80, 'hr_max':140, 'spo2_min':95, 'resp_min':22, 'resp_max':34},
    {'name': 'School', 'min':6, 'max':12, 'hr_min':70, 'hr_max':120, 'spo2_min':95, 'resp_min':18, 'resp_max':30},
    {'name': 'Adolescent', 'min':12, 'max':18, 'hr_min':60, 'hr_max':100, 'spo2_min':95, 'resp_min':12, 'resp_max':20},
    {'name': 'Adult', 'min':18, 'max':50, 'hr_min':60, 'hr_max':100, 'spo2_min':95, 'resp_min':12, 'resp_max':20},
    {'name': 'Middle', 'min':50, 'max':65, 'hr_min':60, 'hr_max':100, 'spo2_min':95, 'resp_min':12, 'resp_max':20},
    {'name': 'Elderly', 'min':65, 'max':200, 'hr_min':50, 'hr_max':95, 'spo2_min':92, 'resp_min':12, 'resp_max':24},
]
def get_age_group(age):
    """Get age group parameters"""
    for g in AGE_GROUPS:
        if g['min'] <= age <= g['max']:
            return g
    return AGE_GROUPS[-1]
def param_status(value, key, age):
    """Get parameter status (normal/warning/critical)"""
    g = get_age_group(age)
    if key == 'hr':
        if value <= 0 or np.isnan(value):
            return 'critical'
        if value < g['hr_min']*0.7 or value > g['hr_max']*1.5:
            return 'critical'
        if value < g['hr_min'] or value > g['hr_max']:
            return 'warning'
        return 'normal'
    if key == 'spo2':
        if value < 85:
            return 'critical'
        if value < 90:
            return 'warning'
        if value < g['spo2_min']:
            return 'warning'
        return 'normal'
    if key == 'flow':
        if value < 0.15:
            return 'critical'
        if value < 0.30:
            return 'warning'
        return 'normal'
    if key == 'perf':
        if value < 0.015:
            return 'critical'
        if value < 0.03:
            return 'warning'
        return 'normal'
    if key == 'ecg_irreg':
        if value > 0.7:
            return 'critical'
        if value > 0.4:
            return 'warning'
        return 'normal'
    if key == 'resp_rate':
        if value <= 0 or np.isnan(value):
            return 'critical'
        if value < g['resp_min']*0.7 or value > g['resp_max']*1.5:
            return 'critical'
        if value < g['resp_min'] or value > g['resp_max']:
            return 'warning'
        return 'normal'
    if key == 'hrv_rmssd':
        if value < 10:
            return 'warning'
        if value < 5:
            return 'critical'
        return 'normal'
    return 'normal'
def fuse_rule_ml(age, hr, spo2, perf, flow, ecg_irreg, hrv_rmssd, resp_rate):
    """Fuse rule-based and ML-based risk assessment"""
    weights = {
        'hr': 0.15,
        'spo2': 0.25,
        'flow': 0.15,
        'perf': 0.10,
        'ecg_irreg': 0.10,
        'hrv_rmssd': 0.10,
        'resp_rate': 0.15
    }
   
    status = {
        k: param_status(v, k, age) for k, v in [
            ('hr', hr),
            ('spo2', spo2),
            ('flow', flow),
            ('perf', perf),
            ('ecg_irreg', ecg_irreg),
            ('hrv_rmssd', hrv_rmssd),
            ('resp_rate', resp_rate)
        ]
    }
   
    score_map = {'normal': 0.0, 'warning': 0.6, 'critical': 1.0}
    rule_score = sum([weights[k] * score_map[status[k]] for k in weights])
   
    X = np.array([[age, hr, spo2, perf, flow, ecg_irreg, hrv_rmssd, resp_rate]], dtype=np.float32)
    Xs = RF_BUNDLE['scaler'].transform(X)
    probs = RF_BUNDLE['model'].predict_proba(Xs)[0]
    ml_label_idx = int(np.argmax(probs))
    ml_conf = float(probs[ml_label_idx])
   
    inv_map = {0: 'Normal', 1: 'At Risk', 2: 'Critical'}
    ml_label = inv_map[ml_label_idx]
   
    final_score = 0.4 * rule_score + 0.6 * (ml_conf if ml_label != 'Normal' else (1 - ml_conf))
   
    if ml_label == 'Critical' or rule_score > 0.7 or final_score > 0.65:
        tier = 'Critical'
    elif ml_label == 'At Risk' or rule_score > 0.35 or final_score > 0.35:
        tier = 'At Risk'
    else:
        tier = 'Normal'
       
    return {
        'rule_score': rule_score,
        'ml_label': ml_label,
        'ml_conf': ml_conf,
        'final_score': final_score,
        'tier': tier,
        'param_statuses': status
    }
# -------------------------
# Patient Management
# -------------------------
def make_patient(pid=None):
    """Create a simulated patient"""
    pid = pid or str(uuid.uuid4())[:8]
    age = int(np.random.randint(1, 85))
    sex = np.random.choice(['Male', 'Female', 'Other'])
    name = f"Patient_{pid}"
    height = clamp(np.random.normal(170, 15), 50, 200)
    weight = clamp(np.random.normal(70, 15), 3, 150)
    bmi = weight / ((height/100) ** 2)
   
    conditions = ['Hypertension', 'Diabetes', 'Asthma', 'None']
    probabilities = [0.3, 0.2, 0.1, 0.4]
    medical_history = np.random.choice(conditions, p=probabilities)
   
    return {
        'id': pid,
        'name': name,
        'age': age,
        'sex': sex,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'medical_history': medical_history,
        'created': now_iso()
    }
# -------------------------
# Alert System
# -------------------------
class AlertSystem:
    """Real-time alert system for vital signs monitoring"""
   
    def __init__(self, firebase_manager=None):
        self.alerts = deque(maxlen=100)
        self.firebase_manager = firebase_manager
        self.alert_rules = {
            'critical_spo2': {'threshold': 90, 'duration': 10, 'priority': 'high'},
            'critical_hr': {'threshold_low': 40, 'threshold_high': 160, 'duration': 15, 'priority': 'high'},
            'low_perfusion': {'threshold': 0.02, 'duration': 20, 'priority': 'medium'},
            'low_flow': {'threshold': 0.3, 'duration': 15, 'priority': 'medium'},
            'high_resp_rate': {'threshold': 30, 'duration': 30, 'priority': 'medium'},
            'low_resp_rate': {'threshold': 8, 'duration': 30, 'priority': 'high'}
        }
        self.alert_states = {key: {'count': 0, 'triggered': False} for key in self.alert_rules}
       
    def check_alerts(self, vital_signs, patient_id=None):
        """Check for alert conditions"""
        current_alerts = []
        ts = now_iso()
       
        # Check SpO2
        if vital_signs.get('spo2', 100) < self.alert_rules['critical_spo2']['threshold']:
            self.alert_states['critical_spo2']['count'] += 1
            if self.alert_states['critical_spo2']['count'] >= self.alert_rules['critical_spo2']['duration']:
                if not self.alert_states['critical_spo2']['triggered']:
                    alert = {
                        'id': str(uuid.uuid4()),
                        'type': 'critical_spo2',
                        'message': f"Critical SpO2: {vital_signs.get('spo2', 0)}%",
                        'priority': 'high',
                        'timestamp': ts,
                        'acknowledged': False,
                        'patient_id': patient_id
                    }
                    self.alerts.append(alert)
                    current_alerts.append(alert)
                    self.alert_states['critical_spo2']['triggered'] = True
                   
                    if self.firebase_manager and patient_id:
                        self.firebase_manager.push_alert(patient_id, alert)
        else:
            self.alert_states['critical_spo2']['count'] = 0
            self.alert_states['critical_spo2']['triggered'] = False
           
        # Check heart rate
        hr = vital_signs.get('hr', 0)
        if hr < self.alert_rules['critical_hr']['threshold_low'] or hr > self.alert_rules['critical_hr']['threshold_high']:
            self.alert_states['critical_hr']['count'] += 1
            if self.alert_states['critical_hr']['count'] >= self.alert_rules['critical_hr']['duration']:
                if not self.alert_states['critical_hr']['triggered']:
                    alert = {
                        'id': str(uuid.uuid4()),
                        'type': 'critical_hr',
                        'message': f"Critical HR: {hr} bpm",
                        'priority': 'high',
                        'timestamp': ts,
                        'acknowledged': False,
                        'patient_id': patient_id
                    }
                    self.alerts.append(alert)
                    current_alerts.append(alert)
                    self.alert_states['critical_hr']['triggered'] = True
                   
                    if self.firebase_manager and patient_id:
                        self.firebase_manager.push_alert(patient_id, alert)
        else:
            self.alert_states['critical_hr']['count'] = 0
            self.alert_states['critical_hr']['triggered'] = False
           
        return current_alerts
   
    def get_active_alerts(self):
        """Get all active (unacknowledged) alerts"""
        return [alert for alert in self.alerts if not alert['acknowledged']]
   
    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
# -------------------------
# EchoCardiology Modules
# -------------------------
class EchoPreprocessor:
    """Preprocess echocardiography images and videos"""
   
    @staticmethod
    def preprocess_echo(uploaded_file):
        """Preprocess ultrasound image or video"""
        try:
            file_type = uploaded_file.type
            file_bytes = uploaded_file.getvalue()
           
            results = {
                'original_type': file_type,
                'filename': uploaded_file.name,
                'processed_image': None,
                'quality_score': 0.0,
                'dimensions': (0, 0),
                'aspect_ratio': 0.0,
                'brightness_mean': 0.0,
                'contrast_score': 0.0,
                'sharpness_score': 0.0
            }
           
            # Handle images
            if 'image' in file_type:
                image = Image.open(io.BytesIO(file_bytes))
               
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
               
                # Resize for consistency
                max_size = (512, 512)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
               
                # Calculate image quality metrics
                img_array = np.array(image)
                brightness = np.mean(img_array) / 255.0
                contrast = np.std(img_array) / 128.0
               
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
               
                quality_score = (brightness * 0.3 +
                               min(contrast, 1.0) * 0.4 +
                               min(sharpness, 1.0) * 0.3)
               
                results.update({
                    'processed_image': image,
                    'dimensions': image.size,
                    'aspect_ratio': image.size[0] / image.size[1] if image.size[1] > 0 else 0,
                    'brightness_mean': brightness,
                    'contrast_score': contrast,
                    'sharpness_score': sharpness,
                    'quality_score': min(max(quality_score, 0), 1)
                })
               
            # Handle videos
            elif 'video' in file_type:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
               
                cap = cv2.VideoCapture(tmp_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        max_size = (512, 512)
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                       
                        img_array = np.array(image)
                        brightness = np.mean(img_array) / 255.0
                        contrast = np.std(img_array) / 128.0
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
                        quality_score = (brightness * 0.3 +
                                       min(contrast, 1.0) * 0.4 +
                                       min(sharpness, 1.0) * 0.3)
                       
                        results.update({
                            'processed_image': image,
                            'dimensions': image.size,
                            'aspect_ratio': image.size[0] / image.size[1] if image.size[1] > 0 else 0,
                            'brightness_mean': brightness,
                            'contrast_score': contrast,
                            'sharpness_score': sharpness,
                            'quality_score': min(max(quality_score, 0), 1),
                            'video_frame_extracted': True,
                            'video_duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                        })
                    cap.release()
               
                os.unlink(tmp_path)
           
            return results
           
        except Exception as e:
            st.error(f"Error preprocessing echo file: {e}")
            return {
                'error': str(e),
                'quality_score': 0.0,
                'processed_image': None
            }
class ECGImagePreprocessor:
    """Preprocess ECG images"""
   
    @staticmethod
    def preprocess_ecg(uploaded_file):
        """Preprocess ECG image"""
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
           
            if image.mode != 'RGB':
                image = image.convert('RGB')
           
            max_size = (400, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
           
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
           
            height, width = gray.shape
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
           
            simulated_qrs_count = int(edge_density * width / 10)
           
            paper_speed = 25
            mm_per_pixel = 0.1
            pixels_per_second = paper_speed / mm_per_pixel
           
            if simulated_qrs_count > 1:
                avg_rr_pixels = width / simulated_qrs_count
                hr_bpm = 60 * pixels_per_second / avg_rr_pixels
            else:
                hr_bpm = random.uniform(60, 100)
           
            hr_bpm += random.uniform(-5, 5)
            hr_bpm = max(40, min(120, hr_bpm))
           
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 128.0
            quality_score = 0.5 * brightness + 0.5 * min(contrast, 1.0)
           
            return {
                'processed_image': image,
                'estimated_hr': hr_bpm,
                'quality_score': quality_score,
                'edge_density': edge_density,
                'simulated_qrs_count': simulated_qrs_count,
                'dimensions': image.size
            }
           
        except Exception as e:
            st.error(f"Error preprocessing ECG: {e}")
            return {
                'processed_image': None,
                'estimated_hr': random.uniform(60, 100),
                'quality_score': 0.0,
                'error': str(e)
            }
class AudioPreprocessor:
    """Preprocess heart sound audio"""
   
    @staticmethod
    def preprocess_audio(uploaded_file):
        """Preprocess heart sound audio file"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
           
            if uploaded_file.type in ['audio/wav', 'audio/x-wav']:
                y, sr = librosa.load(tmp_path, sr=None)
            else:
                try:
                    audio = pydub.AudioSegment.from_file(tmp_path)
                    y = np.array(audio.get_array_of_samples()).astype(np.float32)
                    if audio.channels == 2:
                        y = y.reshape((-1, 2)).mean(axis=1)
                    sr = audio.frame_rate
                except:
                    y, sr = librosa.load(tmp_path, sr=22050)
           
            duration = len(y) / sr
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
           
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
           
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
           
            hr_from_audio = random.uniform(50, 120)
            snr_estimate = 20 * np.log10(np.std(y) / (np.std(y[:1000]) + 1e-10))
            quality_score = min(max(snr_estimate / 40, 0), 1)
           
            os.unlink(tmp_path)
           
            return {
                'duration': duration,
                'sample_rate': sr,
                'audio_data': y,
                'mfcc_features': mfcc_mean.tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'zcr_mean': float(np.mean(zcr)),
                'estimated_hr': hr_from_audio,
                'quality_score': quality_score,
                'snr_estimate': snr_estimate,
                'peak_amplitude': float(np.max(np.abs(y)))
            }
           
        except Exception as e:
            st.error(f"Error preprocessing audio: {e}")
            return {
                'duration': 0,
                'sample_rate': 0,
                'audio_data': None,
                'mfcc_features': [0] * 13,
                'estimated_hr': random.uniform(50, 120),
                'quality_score': 0.0,
                'error': str(e)
            }
class EchoInferenceEngine:
    """Simulated AI inference engine for echocardiography"""
   
    @staticmethod
    def run_inference(echo_data, ecg_data=None, audio_data=None):
        """Run simulated AI inference on echocardiography data"""
        try:
            view_options = ['AP4', 'PLAX', 'PSAX', 'A4C', 'A2C', 'Unknown']
           
            if echo_data and 'aspect_ratio' in echo_data:
                aspect_ratio = echo_data['aspect_ratio']
                if aspect_ratio > 1.2:
                    view = 'PLAX'
                elif aspect_ratio > 0.9:
                    view = 'AP4'
                elif aspect_ratio > 0.7:
                    view = 'PSAX'
                else:
                    view = random.choice(view_options[:-1])
            else:
                view = random.choice(view_options)
           
            base_ef = random.uniform(40, 70)
            if echo_data and 'quality_score' in echo_data:
                quality = echo_data['quality_score']
                ef_variation = 15 * (1 - quality)
                ef = base_ef + random.uniform(-ef_variation, ef_variation)
            else:
                ef = base_ef
           
            ef = max(20, min(80, ef))
           
            flags = []
            flag_options = [
                'Normal',
                'Possible EF reduction',
                'Possible wall motion abnormality',
                'Possible valvular abnormality',
                'Possible pericardial effusion',
                'Possible chamber enlargement'
            ]
           
            if ef < 50:
                flags.append('Possible EF reduction')
            if random.random() < 0.3:
                flags.append(random.choice(flag_options[2:]))
            if not flags:
                flags.append('Normal')
           
            triage_scores = {'Green': 0, 'Yellow': 0, 'Red': 0}
           
            if ef < 35:
                triage_scores['Red'] += 2
            elif ef < 50:
                triage_scores['Yellow'] += 1
            else:
                triage_scores['Green'] += 1
           
            if ecg_data and 'estimated_hr' in ecg_data:
                hr = ecg_data['estimated_hr']
                if hr < 50 or hr > 120:
                    triage_scores['Yellow'] += 1
                if hr < 40 or hr > 140:
                    triage_scores['Red'] += 1
           
            if audio_data:
                if audio_data['quality_score'] < 0.3:
                    triage_scores['Yellow'] += 0.5
           
            if triage_scores['Red'] > 0:
                triage = 'Red'
            elif triage_scores['Yellow'] > 1:
                triage = 'Yellow'
            else:
                triage = 'Green'
           
            view_confidence = random.uniform(0.7, 0.95)
            ef_confidence = min(0.9, 0.5 + echo_data.get('quality_score', 0) * 0.4)
           
            findings = []
            findings.append(f"View: {view} (confidence: {view_confidence:.1%})")
            findings.append(f"Estimated EF: {ef:.1f}% (confidence: {ef_confidence:.1%})")
            findings.append(f"Flags: {', '.join(flags)}")
           
            if ecg_data:
                findings.append(f"Estimated HR from ECG: {ecg_data.get('estimated_hr', 'N/A'):.1f} bpm")
           
            if audio_data:
                findings.append(f"Audio quality: {audio_data.get('quality_score', 0):.2f}")
           
            return {
                'view_classification': view,
                'view_confidence': view_confidence,
                'ejection_fraction': ef,
                'ef_confidence': ef_confidence,
                'pathology_flags': flags,
                'triage_level': triage,
                'triage_scores': triage_scores,
                'findings_summary': findings,
                'quality_score': echo_data.get('quality_score', 0) if echo_data else 0,
                'timestamp': datetime.now().isoformat()
            }
           
        except Exception as e:
            st.error(f"Error in inference: {e}")
            return {
                'view_classification': 'Unknown',
                'view_confidence': 0.0,
                'ejection_fraction': 55.0,
                'ef_confidence': 0.0,
                'pathology_flags': ['Error in analysis'],
                'triage_level': 'Yellow',
                'findings_summary': [f"Error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
# -------------------------
# Firebase Integration
# -------------------------
class FirebaseManager:
    """Advanced Firebase Realtime Database manager with schema support"""
   
    def __init__(self):
        self.initialized = False
        self.connection_status = "Disconnected"
        self.last_sync_time = None
        self.error_count = 0
        self.max_errors = 5
        self.patient_cache = {}
       
    def initialize_firebase(self, firebase_config, database_url):
        """Initialize Firebase with service account credentials"""
        try:
            if not FIREBASE_AVAILABLE:
                st.error("Firebase Admin SDK not installed")
                return False
               
            if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
                self.initialized = True
                self.connection_status = "Connected"
                return True
           
            if isinstance(firebase_config, dict):
                cred = credentials.Certificate(firebase_config)
            else:
                cred = credentials.Certificate(firebase_config)
               
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })
           
            self.initialized = True
            self.connection_status = "Connected"
            self.last_sync_time = datetime.now()
            self._load_patient_profiles()
            return True
           
        except Exception as e:
            st.error(f"❌ Firebase initialization failed: {str(e)}")
            self.connection_status = f"Error: {str(e)}"
            return False
   
    def _load_patient_profiles(self):
        """Load patient profiles from Firebase"""
        try:
            ref = db.reference('/patients')
            snapshot = ref.get()
           
            if snapshot:
                for patient_id, patient_data in snapshot.items():
                    if 'profile' in patient_data:
                        self.patient_cache[patient_id] = patient_data['profile']
            st.success(f"✅ Loaded {len(self.patient_cache)} patient profiles from Firebase")
        except Exception as e:
            st.warning(f"Could not load patient profiles: {e}")
   
    def fetch_patient_data(self, patient_id, data_type="vitals", limit=100):
        """Fetch patient data from Firebase"""
        if not self.initialized:
            return None
           
        try:
            ref = db.reference(f'/patients/{patient_id}/{data_type}')
            snapshot = ref.order_by_child('timestamp').limit_to_last(limit).get()
           
            if snapshot:
                self.error_count = 0
                self.last_sync_time = datetime.now()
                return snapshot
            else:
                return None
               
        except Exception as e:
            self.error_count += 1
            return None
   
    def fetch_all_patients_vitals(self, limit_per_patient=50):
        """Fetch vitals data for all patients"""
        if not self.initialized:
            return {}
           
        try:
            ref = db.reference('/patients')
            snapshot = ref.get()
           
            all_patients_data = {}
            if snapshot:
                for patient_id in snapshot.keys():
                    vitals = self.fetch_patient_data(patient_id, "vitals", limit_per_patient)
                    if vitals:
                        all_patients_data[patient_id] = vitals
           
            return all_patients_data
        except Exception as e:
            st.error(f"Error fetching all patients data: {e}")
            return {}
   
    def push_patient_data(self, patient_id, data_type, data):
        """Push data to Firebase according to schema"""
        if not self.initialized:
            return False
           
        try:
            ref = db.reference(f'/patients/{patient_id}/{data_type}')
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
           
            new_ref = ref.push(data)
            self.last_sync_time = datetime.now()
            return True
           
        except Exception as e:
            return False
   
    def push_patient_profile(self, patient_id, profile_data):
        """Push patient profile to Firebase"""
        if not self.initialized:
            return False
           
        try:
            ref = db.reference(f'/patients/{patient_id}/profile')
            ref.set(profile_data)
            self.patient_cache[patient_id] = profile_data
            return True
        except Exception as e:
            return False
   
    def push_echo_study(self, patient_id, study_data):
        """Push echocardiography study to Firebase"""
        if not self.initialized:
            return False
           
        try:
            ref = db.reference(f'/patients/{patient_id}/echo_studies')
            new_ref = ref.push(study_data)
            return True
        except Exception as e:
            return False
   
    def push_alert(self, patient_id, alert_data):
        """Push alert to Firebase"""
        return self.push_patient_data(patient_id, "alerts", alert_data)
   
    def get_patient_profile(self, patient_id):
        """Get patient profile from cache or Firebase"""
        if patient_id in self.patient_cache:
            return self.patient_cache[patient_id]
       
        try:
            ref = db.reference(f'/patients/{patient_id}/profile')
            profile = ref.get()
            if profile:
                self.patient_cache[patient_id] = profile
            return profile
        except:
            return None
   
    def get_available_patients(self):
        """Get list of available patients from Firebase"""
        if not self.initialized:
            return []
           
        try:
            ref = db.reference('/patients')
            snapshot = ref.get()
            return list(snapshot.keys()) if snapshot else []
        except:
            return []
   
    def get_connection_status(self):
        """Get current connection status"""
        if self.last_sync_time:
            time_since_sync = (datetime.now() - self.last_sync_time).total_seconds()
            if time_since_sync > 60:
                self.connection_status = "Stale Connection"
        return self.connection_status
class FirebaseDataProcessor:
    """Processes Firebase data according to the provided schema"""
   
    def __init__(self, firebase_manager):
        self.firebase_manager = firebase_manager
   
    def process_firebase_vitals_record(self, firebase_record, patient_id):
        """Convert Firebase vitals record to app format"""
        try:
            profile = self.firebase_manager.get_patient_profile(patient_id)
            record = {
                'timestamp': firebase_record.get('timestamp', now_iso()),
                'patient_id': patient_id,
                'patient_name': profile.get('name', f'Patient_{patient_id}') if profile else f'Patient_{patient_id}',
                'age': profile.get('age', 45) if profile else 45,
                'hr': float(firebase_record.get('hr', 75)),
                'spo2': float(firebase_record.get('spo2', 97)),
                'perf': float(firebase_record.get('perf', 0.08)),
                'flow': float(firebase_record.get('flow', 0.95)),
                'resp_rate': float(firebase_record.get('resp_rate', 16)),
                'ecg_irreg': float(firebase_record.get('ecg_irreg', 0.1)),
                'hrv_rmssd': float(firebase_record.get('hrv_rmssd', 40)),
                'pcg_murmur_index': float(firebase_record.get('pcg_murmur_index', 0.05)),
                'condition': firebase_record.get('condition', 'normal'),
                'data_source': 'Firebase'
            }
           
            fuse_result = fuse_rule_ml(
                record['age'], record['hr'], record['spo2'], record['perf'],
                record['flow'], record['ecg_irreg'], record['hrv_rmssd'], record['resp_rate']
            )
           
            record['fusion_tier'] = fuse_result['tier']
            record['fusion_score'] = fuse_result['final_score']
            record['ml_confidence'] = fuse_result['ml_conf']
            record['rule_score'] = fuse_result['rule_score']
           
            return record
        except Exception as e:
            st.error(f"Error processing Firebase record: {e}")
            return None
   
    def fetch_and_process_realtime_data(self, patient_id, limit=10):
        """Fetch and process real-time data for a patient"""
        firebase_data = self.firebase_manager.fetch_patient_data(patient_id, "vitals", limit)
        processed_records = []
       
        if firebase_data:
            for record_id, record_data in firebase_data.items():
                processed_record = self.process_firebase_vitals_record(record_data, patient_id)
                if processed_record:
                    processed_records.append(processed_record)
       
        return processed_records
   
    def sync_all_patients_data(self, limit_per_patient=20):
        """Sync data for all patients from Firebase"""
        all_patients_data = self.firebase_manager.fetch_all_patients_vitals(limit_per_patient)
        processed_data = {}
       
        for patient_id, vitals_data in all_patients_data.items():
            processed_records = []
            for record_id, record_data in vitals_data.items():
                processed_record = self.process_firebase_vitals_record(record_data, patient_id)
                if processed_record:
                    processed_records.append(processed_record)
           
            if processed_records:
                processed_data[patient_id] = processed_records
       
        return processed_data
class FirebasePatientManager:
    """Manages patient data with Firebase integration"""
   
    def __init__(self, firebase_manager):
        self.firebase_manager = firebase_manager
        self.data_processor = FirebaseDataProcessor(firebase_manager)
   
    def get_available_patients(self):
        """Get patients from Firebase or fallback to simulated patients"""
        if self.firebase_manager.initialized:
            firebase_patients = self.firebase_manager.get_available_patients()
            if firebase_patients:
                patients = []
                for patient_id in firebase_patients:
                    profile = self.firebase_manager.get_patient_profile(patient_id)
                    if profile:
                        patients.append({
                            'id': patient_id,
                            'name': profile.get('name', f'Patient_{patient_id}'),
                            'age': profile.get('age', 45),
                            'sex': profile.get('sex', 'Unknown'),
                            'height': profile.get('height', 170),
                            'weight': profile.get('weight', 70),
                            'bmi': profile.get('bmi', 24.2),
                            'medical_history': profile.get('medical_history', 'None'),
                            'created': profile.get('created', now_iso())
                        })
                return patients
       
        return [make_patient(f"P{1000+i}") for i in range(20)]
   
    def create_firebase_patient(self, patient_data):
        """Create a new patient in Firebase"""
        if not self.firebase_manager.initialized:
            return False
           
        try:
            profile_data = {
                'name': patient_data['name'],
                'age': patient_data['age'],
                'sex': patient_data['sex'],
                'height': patient_data['height'],
                'weight': patient_data['weight'],
                'bmi': patient_data['bmi'],
                'medical_history': patient_data['medical_history'],
                'created': now_iso()
            }
           
            return self.firebase_manager.push_patient_profile(patient_data['id'], profile_data)
        except Exception as e:
            st.error(f"Error creating patient in Firebase: {e}")
            return False
# -------------------------
# Enhanced Report Generator
# -------------------------
class ComprehensiveReportGenerator:
    """Generates comprehensive clinical reports with multi-modal analysis"""
   
    def __init__(self):
        pass
   
    def generate_multi_modal_report(self, patient_data, session_data, echo_data=None,
                                  ecg_image_data=None, audio_data=None, echo_inference=None):
        """Generate comprehensive multi-modal report"""
        try:
            report = {
                "metadata": {
                    "report_id": str(uuid.uuid4()),
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "multi_modal_clinical_analysis",
                    "software_version": VERSION
                },
                "patient_information": self._extract_patient_info(patient_data),
                "ecg_multisensor_analysis": self._analyze_ecg_multisensor(session_data),
                "echocardiography_analysis": self._analyze_echocardiography(echo_data, echo_inference),
                "integrated_risk_assessment": self._integrated_risk_assessment(session_data, echo_inference),
                "clinical_recommendations": self._generate_multi_modal_recommendations(session_data, echo_inference),
                "technical_summary": self._technical_summary(session_data, echo_data, ecg_image_data, audio_data)
            }
            return json.dumps(report, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
   
    def _extract_patient_info(self, patient_data):
        """Extract patient information"""
        return {
            "patient_id": patient_data.get('id', 'N/A'),
            "name": patient_data.get('name', 'N/A'),
            "age": patient_data.get('age', 'N/A'),
            "sex": patient_data.get('sex', 'N/A'),
            "height_cm": patient_data.get('height', 'N/A'),
            "weight_kg": patient_data.get('weight', 'N/A'),
            "bmi": patient_data.get('bmi', 'N/A'),
            "medical_history": patient_data.get('medical_history', 'N/A'),
            "data_source": patient_data.get('data_source', 'Local')
        }
   
    def _analyze_ecg_multisensor(self, session_data):
        """Analyze ECG multisensor data"""
        if not session_data:
            return {"status": "No data available"}
       
        df = pd.DataFrame(session_data)
        analysis = {}
       
        vital_params = ['hr', 'spo2', 'resp_rate', 'perf', 'flow', 'ecg_irreg', 'hrv_rmssd']
       
        for param in vital_params:
            if param in df.columns:
                analysis[param] = {
                    'mean': float(df[param].mean()),
                    'std': float(df[param].std()),
                    'min': float(df[param].min()),
                    'max': float(df[param].max()),
                    'trend': 'increasing' if len(df) > 1 and df[param].iloc[-1] > df[param].iloc[0] else 'decreasing'
                }
       
        if 'fusion_score' in df.columns:
            analysis['risk_assessment'] = {
                'mean_risk_score': float(df['fusion_score'].mean()),
                'max_risk_score': float(df['fusion_score'].max()),
                'risk_tier_distribution': dict(df['fusion_tier'].value_counts())
            }
       
        return analysis
   
    def _analyze_echocardiography(self, echo_data, echo_inference):
        """Analyze echocardiography data"""
        if not echo_inference:
            return {"status": "No echo data available"}
       
        analysis = {
            "view_classification": echo_inference.get('view_classification', 'Unknown'),
            "view_confidence": echo_inference.get('view_confidence', 0),
            "ejection_fraction": echo_inference.get('ejection_fraction', 0),
            "ef_confidence": echo_inference.get('ef_confidence', 0),
            "pathology_flags": echo_inference.get('pathology_flags', []),
            "triage_level": echo_inference.get('triage_level', 'Unknown'),
            "quality_score": echo_inference.get('quality_score', 0)
        }
       
        if echo_data:
            analysis.update({
                "image_quality": {
                    "brightness": echo_data.get('brightness_mean', 0),
                    "contrast": echo_data.get('contrast_score', 0),
                    "sharpness": echo_data.get('sharpness_score', 0),
                    "overall_score": echo_data.get('quality_score', 0)
                }
            })
       
        return analysis
   
    def _integrated_risk_assessment(self, session_data, echo_inference):
        """Generate integrated risk assessment from both modalities"""
        integrated_risk = {
            "ecg_risk": "Unknown",
            "echo_risk": "Unknown",
            "combined_risk": "Unknown",
            "confidence": 0.0
        }
       
        if session_data and len(session_data) > 0:
            latest_ecg = session_data[-1]
            integrated_risk['ecg_risk'] = latest_ecg.get('fusion_tier', 'Unknown')
            integrated_risk['ecg_score'] = latest_ecg.get('fusion_score', 0)
       
        if echo_inference:
            integrated_risk['echo_risk'] = echo_inference.get('triage_level', 'Unknown')
            integrated_risk['echo_score'] = 1.0 if echo_inference.get('triage_level') == 'Red' else \
                                          0.5 if echo_inference.get('triage_level') == 'Yellow' else 0.1
       
        # Combine risks
        if integrated_risk['ecg_risk'] == 'Critical' or integrated_risk['echo_risk'] == 'Red':
            integrated_risk['combined_risk'] = 'Critical'
            integrated_risk['confidence'] = 0.85
        elif integrated_risk['ecg_risk'] == 'At Risk' or integrated_risk['echo_risk'] == 'Yellow':
            integrated_risk['combined_risk'] = 'At Risk'
            integrated_risk['confidence'] = 0.70
        else:
            integrated_risk['combined_risk'] = 'Normal'
            integrated_risk['confidence'] = 0.90
       
        return integrated_risk
   
    def _generate_multi_modal_recommendations(self, session_data, echo_inference):
        """Generate clinical recommendations based on multi-modal data"""
        recommendations = []
       
        # ECG-based recommendations
        if session_data and len(session_data) > 0:
            latest = session_data[-1]
            if latest.get('fusion_tier') == 'Critical':
                recommendations.append("Immediate intervention required based on ECG multisensor data")
            elif latest.get('fusion_tier') == 'At Risk':
                recommendations.append("Increased monitoring frequency recommended based on ECG trends")
       
        # Echo-based recommendations
        if echo_inference:
            if echo_inference.get('triage_level') == 'Red':
                recommendations.append("Urgent cardiology consultation required based on echocardiography")
            elif echo_inference.get('triage_level') == 'Yellow':
                recommendations.append("Follow-up echocardiogram recommended in 1-3 months")
       
        # Combined recommendations
        if len(recommendations) == 0:
            recommendations.append("Continue routine monitoring as per standard protocol")
       
        return recommendations
   
    def _technical_summary(self, session_data, echo_data, ecg_image_data, audio_data):
        """Generate technical summary"""
        summary = {
            "data_modalities_available": [],
            "total_records": len(session_data) if session_data else 0,
            "echo_data_quality": echo_data.get('quality_score', 0) if echo_data else 0,
            "ecg_image_processed": ecg_image_data is not None,
            "audio_data_processed": audio_data is not None
        }
       
        if session_data:
            summary["data_modalities_available"].append("ECG Multisensor")
        if echo_data:
            summary["data_modalities_available"].append("Echocardiography")
        if ecg_image_data:
            summary["data_modalities_available"].append("ECG Image")
        if audio_data:
            summary["data_modalities_available"].append("Heart Sound Audio")
       
        return summary
   
    def generate_pdf_report(self, patient_data, session_data, echo_data=None,
                          echo_inference=None, output_path=None):
        """Generate PDF report"""
        if output_path is None:
            output_path = f"report_{patient_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
       
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
       
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor(PRIMARY)
        )
        story.append(Paragraph("MEDFUSION AI - Clinical Diagnostic Report", title_style))
        story.append(Spacer(1, 12))
       
        # Patient Information
        story.append(Paragraph("Patient Information", styles['Heading2']))
        patient_info = [
            ["Patient ID:", patient_data.get('id', 'N/A')],
            ["Name:", patient_data.get('name', 'N/A')],
            ["Age/Sex:", f"{patient_data.get('age', 'N/A')}/{patient_data.get('sex', 'N/A')}"],
            ["Height/Weight:", f"{patient_data.get('height', 'N/A')} cm / {patient_data.get('weight', 'N/A')} kg"],
            ["BMI:", f"{patient_data.get('bmi', 'N/A'):.1f}"],
            ["Medical History:", patient_data.get('medical_history', 'N/A')]
        ]
        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f8ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
       
        # ECG Multisensor Analysis
        if session_data and len(session_data) > 0:
            story.append(Paragraph("ECG Multisensor Analysis", styles['Heading2']))
            latest = session_data[-1]
           
            ecg_data = [
                ["Parameter", "Value", "Status"],
                ["Heart Rate", f"{latest.get('hr', 0):.1f} bpm", param_status(latest.get('hr', 0), 'hr', latest.get('age', 45))],
                ["SpO₂", f"{latest.get('spo2', 0):.1f}%", param_status(latest.get('spo2', 0), 'spo2', latest.get('age', 45))],
                ["Respiratory Rate", f"{latest.get('resp_rate', 0):.1f} bpm", param_status(latest.get('resp_rate', 0), 'resp_rate', latest.get('age', 45))],
                ["Fusion Risk Score", f"{latest.get('fusion_score', 0):.3f}", latest.get('fusion_tier', 'Unknown')]
            ]
           
            ecg_table = Table(ecg_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            ecg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(ecg_table)
            story.append(Spacer(1, 20))
       
        # Echocardiography Analysis
        if echo_inference:
            story.append(Paragraph("Echocardiography Analysis", styles['Heading2']))
           
            echo_data = [
                ["Parameter", "Value", "Confidence"],
                ["View Classification", echo_inference.get('view_classification', 'Unknown'), f"{echo_inference.get('view_confidence', 0):.1%}"],
                ["Ejection Fraction", f"{echo_inference.get('ejection_fraction', 0):.1f}%", f"{echo_inference.get('ef_confidence', 0):.1%}"],
                ["Triage Level", echo_inference.get('triage_level', 'Unknown'), "N/A"],
                ["Quality Score", f"{echo_inference.get('quality_score', 0):.2f}", "N/A"]
            ]
           
            echo_table = Table(echo_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            echo_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(ACCENT)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(echo_table)
            story.append(Spacer(1, 20))
           
            # Pathology Flags
            flags = echo_inference.get('pathology_flags', [])
            if flags:
                story.append(Paragraph("Pathology Flags:", styles['Heading3']))
                for flag in flags:
                    story.append(Paragraph(f"• {flag}", styles['Normal']))
                story.append(Spacer(1, 10))
       
        # Clinical Recommendations
        story.append(Paragraph("Clinical Recommendations", styles['Heading2']))
        recommendations = self._generate_multi_modal_recommendations(session_data, echo_inference)
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 20))
       
        # Footer
        story.append(Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"MEDFUSION AI v{VERSION} - For Research Use Only", styles['Italic']))
       
        doc.build(story)
        return output_path
   
    def generate_csv_report(self, session_data):
        """Generate CSV report from session data"""
        if not session_data:
            return ""
       
        df = pd.DataFrame(session_data)
       
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
       
        column_order = []
        for col in ['timestamp', 'patient_id', 'patient_name', 'age', 'data_source',
                   'hr', 'spo2', 'resp_rate', 'perf', 'flow',
                   'ecg_irreg', 'hrv_rmssd', 'pcg_murmur_index',
                   'fusion_tier', 'fusion_score', 'condition']:
            if col in df.columns:
                column_order.append(col)
       
        if column_order:
            df = df[column_order]
       
        return df.to_csv(index=False)
# -------------------------
# Main Streamlit Application
# -------------------------
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🏥"
    )
   
    # Custom CSS
    st.markdown(f"""
    <style>
    .main {{
        background-color: {BACKGROUND};
    }}
    .stAlert {{
        border-radius: 10px;
        border-left: 5px solid;
    }}
    .metric-card {{
        background-color: {CARD};
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    .firebase-status {{
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }}
    .status-connected {{
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }}
    .status-disconnected {{
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }}
    .triage-green {{
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }}
    .triage-yellow {{
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }}
    .triage-red {{
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }}
    .upload-section {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }}
    .tab-content {{
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        margin-top: 10px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f7ff;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)
   
    # App Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🏥 MEDFUSION AI")
        st.markdown(f"### Integrated Multi-Modal Medical Diagnostic Platform")
    with col2:
        st.metric("Version", VERSION)
    with col3:
        st.metric("Status", "🟢 Online")
   
    st.markdown("---")
   
    # Initialize session state
    if 'firebase_manager' not in st.session_state:
        st.session_state.firebase_manager = FirebaseManager()
   
    if 'patient_manager' not in st.session_state:
        st.session_state.patient_manager = FirebasePatientManager(st.session_state.firebase_manager)
   
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ComprehensiveReportGenerator()
   
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = AlertSystem(st.session_state.firebase_manager)
   
    # Initialize ECG multisensor buffers
    if 'buffers' not in st.session_state:
        st.session_state.buffers = {
            'time': deque(maxlen=BUFFER_SECONDS * 50),
            'ecg': deque(maxlen=BUFFER_SECONDS * 50),
            'ppg': deque(maxlen=BUFFER_SECONDS * 50),
            'pcg': deque(maxlen=BUFFER_SECONDS * 50),
            'dop': deque(maxlen=BUFFER_SECONDS * 50),
            'resp': deque(maxlen=BUFFER_SECONDS * 50),
            'spo2': deque(maxlen=BUFFER_SECONDS * 50),
            'perf': deque(maxlen=BUFFER_SECONDS * 50),
            'hr': deque(maxlen=BUFFER_SECONDS * 2),
            'resp_rate': deque(maxlen=BUFFER_SECONDS * 2),
        }
   
    if 'session_log' not in st.session_state:
        st.session_state.session_log = []
   
    if 'running' not in st.session_state:
        st.session_state.running = False
   
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
   
    if 'trend_data' not in st.session_state:
        st.session_state.trend_data = {
            'timestamps': deque(maxlen=TREND_WINDOW),
            'hr': deque(maxlen=TREND_WINDOW),
            'spo2': deque(maxlen=TREND_WINDOW),
            'perf': deque(maxlen=TREND_WINDOW),
            'flow': deque(maxlen=TREND_WINDOW),
            'resp_rate': deque(maxlen=TREND_WINDOW),
            'fusion_score': deque(maxlen=TREND_WINDOW),
        }
   
    # Initialize EchoCardiology session state
    if 'echo_data' not in st.session_state:
        st.session_state.echo_data = None
    if 'ecg_image_data' not in st.session_state:
        st.session_state.ecg_image_data = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'echo_inference' not in st.session_state:
        st.session_state.echo_inference = None
   
    # Initialize processors
    echo_preprocessor = EchoPreprocessor()
    ecg_image_preprocessor = ECGImagePreprocessor()
    audio_preprocessor = AudioPreprocessor()
    echo_inference_engine = EchoInferenceEngine()
   
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
       
        # Data Source Selection
        data_source = st.radio(
            "Data Source Mode",
            ["Simulation Only", "Firebase + Simulation", "Firebase Only"],
            help="Choose data source configuration"
        )
       
        use_firebase = data_source != "Simulation Only"
        firebase_only = data_source == "Firebase Only"
       
        # Firebase Configuration
        if use_firebase:
            st.subheader("🔗 Firebase Configuration")
           
            firebase_url = st.text_input(
                "Database URL",
                placeholder="https://your-project.firebaseio.com"
            )
           
            service_account_type = st.radio(
                "Credentials",
                ["Upload JSON", "Paste JSON"],
                horizontal=True
            )
           
            firebase_config = None
            if service_account_type == "Upload JSON":
                service_account_file = st.file_uploader(
                    "Service Account JSON",
                    type=['json'],
                    label_visibility="collapsed"
                )
                if service_account_file:
                    firebase_config = json.load(service_account_file)
            else:
                service_account_json = st.text_area(
                    "Paste JSON Credentials",
                    height=150,
                    placeholder='{"type": "service_account", ...}'
                )
                if service_account_json:
                    try:
                        firebase_config = json.loads(service_account_json)
                    except:
                        st.error("Invalid JSON format")
           
            if st.button("🚀 Connect to Firebase", type="primary", use_container_width=True):
                if firebase_url and firebase_config:
                    with st.spinner("Initializing Firebase connection..."):
                        success = st.session_state.firebase_manager.initialize_firebase(
                            firebase_config, firebase_url
                        )
                        if success:
                            st.success("✅ Firebase connected!")
                            st.rerun()
                        else:
                            st.error("❌ Connection failed")
                else:
                    st.error("Please provide Firebase URL and credentials")
       
        # Display Firebase Status
        if use_firebase:
            status = st.session_state.firebase_manager.get_connection_status()
            status_class = "status-connected" if status == "Connected" else "status-disconnected"
            st.markdown(f"""
            <div class="firebase-status {status_class}">
            🔗 Firebase: {status}
            </div>
            """, unsafe_allow_html=True)
           
            if st.session_state.firebase_manager.last_sync_time:
                st.caption(f"Last sync: {st.session_state.firebase_manager.last_sync_time.strftime('%H:%M:%S')}")
       
        st.markdown("---")
       
        # Patient Management
        st.header("👨‍⚕️ Patient Management")
       
        # Get available patients
        available_patients = st.session_state.patient_manager.get_available_patients()
       
        if available_patients:
            patient_options = [f"{p['id']} - {p['name']} ({p['age']}y)" for p in available_patients]
            selected_patient = st.selectbox("Select Patient", patient_options, index=0)
           
            if selected_patient:
                patient_id = selected_patient.split(" - ")[0]
                current_patient = next((p for p in available_patients if p['id'] == patient_id), None)
               
                if current_patient:
                    st.session_state.current_patient = current_patient
                    st.success(f"Selected: {current_patient['name']}")
        else:
            st.session_state.current_patient = make_patient("P1000")
            st.warning("No patients found. Using demo patient.")
       
        # Add new patient
        with st.expander("➕ Add New Patient"):
            new_patient_id = st.text_input("Patient ID", value=f"P{random.randint(1000, 9999)}")
            new_patient_name = st.text_input("Name", value="New Patient")
            col_age, col_sex = st.columns(2)
            with col_age:
                new_patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
            with col_sex:
                new_patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
           
            if st.button("Create Patient", type="secondary"):
                new_patient = {
                    'id': new_patient_id,
                    'name': new_patient_name,
                    'age': new_patient_age,
                    'sex': new_patient_sex,
                    'height': 170,
                    'weight': 70,
                    'bmi': 24.2,
                    'medical_history': 'None'
                }
               
                if use_firebase and st.session_state.firebase_manager.initialized:
                    if st.session_state.patient_manager.create_firebase_patient(new_patient):
                        st.success(f"Patient {new_patient_name} created!")
                        st.rerun()
                    else:
                        st.error("Failed to create patient in Firebase")
                else:
                    st.session_state.current_patient = new_patient
                    st.success("Patient created locally!")
       
        st.markdown("---")
       
        # Monitoring Controls
        st.header("🎛️ Monitoring Controls")
       
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start Monitoring", type="primary", use_container_width=True):
                st.session_state.running = True
                st.success("Monitoring started")
        with col_stop:
            if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                st.session_state.running = False
                st.info("Monitoring stopped")
       
        update_frequency = st.slider("Update Frequency (Hz)", 0.1, 5.0, 1.0, 0.1)
       
        st.markdown("---")
       
        # Display Options
        st.header("👁️ Display Options")
       
        show_waveforms = st.checkbox("Show Waveforms", value=True)
        show_trends = st.checkbox("Show Trends", value=True)
        show_alerts = st.checkbox("Show Alerts Panel", value=True)
        auto_refresh = st.checkbox("Auto-refresh", value=True)
       
        # Sample Data Option
        st.markdown("---")
        st.header("🧪 Sample Data")
        load_samples = st.checkbox("Load Sample Files for Testing", value=False)
        if load_samples:
            st.info("Sample files will be loaded automatically in the EchoCardiology tab for testing. Ensure 'samples/' directory exists with: echo_sample.jpg, ecg_sample.jpg, heart_sound_sample.wav")
   
    # Main Content Area
    tabs = st.tabs([
        "🏠 Dashboard",
        "📈 ECG Multisensor",
        "🫀 EchoCardiology",
        "🔄 Fusion Analysis",
        "📊 Trends & Analytics",
        "📋 Reports",
        "🌐 Data Management"
    ])
   
    # Tab 1: Dashboard
    with tabs[0]:
        st.header("🏠 Integrated Dashboard")
       
        # Patient Summary
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Patient", st.session_state.current_patient.get('name', 'Unknown'))
            st.caption(f"ID: {st.session_state.current_patient.get('id', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Age/Sex", f"{st.session_state.current_patient.get('age', 'N/A')}/{st.session_state.current_patient.get('sex', 'N/A')}")
            st.caption("Demographics")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if st.session_state.session_log:
                latest = st.session_state.session_log[-1]
                st.metric("ECG Risk", latest.get('fusion_tier', 'Unknown'))
                st.caption(f"Score: {latest.get('fusion_score', 0):.3f}")
            else:
                st.metric("ECG Risk", "No Data")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if st.session_state.echo_inference:
                triage = st.session_state.echo_inference.get('triage_level', 'Unknown')
                triage_color_class = f"triage-{triage.lower()}"
                st.markdown(f'<div class="{triage_color_class}">', unsafe_allow_html=True)
                st.metric("Echo Triage", triage)
                st.caption(f"EF: {st.session_state.echo_inference.get('ejection_fraction', 0):.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.metric("Echo Triage", "No Data")
            st.markdown('</div>', unsafe_allow_html=True)
       
        st.markdown("---")
       
        # Real-time Metrics
        st.subheader("📊 Real-time Metrics")
       
        if st.session_state.session_log:
            latest = st.session_state.session_log[-1]
           
            # Vital Signs Grid
            col1, col2, col3, col4, col5 = st.columns(5)
           
            with col1:
                status = param_status(latest['hr'], 'hr', latest['age'])
                color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: {SECONDARY};">Heart Rate</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{latest['hr']:.0f}</div>
                    <div style="font-size: 0.8em; color: {color};">bpm • {status}</div>
                </div>
                """, unsafe_allow_html=True)
           
            with col2:
                status = param_status(latest['spo2'], 'spo2', latest['age'])
                color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: {SECONDARY};">SpO₂</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{latest['spo2']:.0f}</div>
                    <div style="font-size: 0.8em; color: {color};">% • {status}</div>
                </div>
                """, unsafe_allow_html=True)
           
            with col3:
                status = param_status(latest['resp_rate'], 'resp_rate', latest['age'])
                color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: {SECONDARY};">Resp Rate</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{latest['resp_rate']:.0f}</div>
                    <div style="font-size: 0.8em; color: {color};">bpm • {status}</div>
                </div>
                """, unsafe_allow_html=True)
           
            with col4:
                status = param_status(latest['perf'], 'perf', latest['age'])
                color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: {SECONDARY};">Perfusion</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{latest['perf']:.3f}</div>
                    <div style="font-size: 0.8em; color: {color};">index • {status}</div>
                </div>
                """, unsafe_allow_html=True)
           
            with col5:
                status = param_status(latest['flow'], 'flow', latest['age'])
                color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: {SECONDARY};">Flow</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{latest['flow']:.3f}</div>
                    <div style="font-size: 0.8em; color: {color};">index • {status}</div>
                </div>
                """, unsafe_allow_html=True)
       
        st.markdown("---")
       
        # Combined Risk Assessment
        st.subheader("🎯 Integrated Risk Assessment")
       
        col1, col2 = st.columns(2)
       
        with col1:
            # ECG Risk Gauge
            if st.session_state.session_log:
                latest = st.session_state.session_log[-1]
                risk_score = latest.get('fusion_score', 0)
               
                fig_ecg_risk = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "ECG Multisensor Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score * 100
                        }
                    }
                ))
                fig_ecg_risk.update_layout(height=250)
                st.plotly_chart(fig_ecg_risk, use_container_width=True)
       
        with col2:
            # Echo Risk Gauge
            if st.session_state.echo_inference:
                ef = st.session_state.echo_inference.get('ejection_fraction', 0)
               
                fig_echo_risk = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ef,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Ejection Fraction"},
                    gauge={
                        'axis': {'range': [0, 80]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 35], 'color': "red"},
                            {'range': [35, 50], 'color': "orange"},
                            {'range': [50, 80], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': ef
                        }
                    }
                ))
                fig_echo_risk.update_layout(height=250)
                st.plotly_chart(fig_echo_risk, use_container_width=True)
       
        # Alerts Panel
        if show_alerts:
            st.markdown("---")
            st.subheader("🚨 Active Alerts")
           
            active_alerts = st.session_state.alert_system.get_active_alerts()
            if active_alerts:
                for alert in active_alerts:
                    if alert['priority'] == 'high':
                        st.error(f"**{alert['message']}** - {alert['timestamp']}")
                    else:
                        st.warning(f"**{alert['message']}** - {alert['timestamp']}")
                   
                    col_ack, _ = st.columns([1, 5])
                    with col_ack:
                        if st.button(f"Acknowledge", key=f"ack_{alert['id']}"):
                            st.session_state.alert_system.acknowledge_alert(alert['id'])
                            st.rerun()
            else:
                st.success("✅ No active alerts")
   
    # Tab 2: ECG Multisensor
    with tabs[1]:
        st.header("📈 ECG Multisensor Monitoring")
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            st.subheader("Real-time Waveforms")
           
            if show_waveforms and st.session_state.buffers['ecg']:
                fig_waveforms = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('ECG', 'PPG', 'PCG', 'Doppler', 'Respiratory', 'Fusion Score'),
                    vertical_spacing=0.1
                )
               
                # Add traces
                ecg_plot = list(st.session_state.buffers['ecg'])[-200:]
                if ecg_plot:
                    fig_waveforms.add_trace(
                        go.Scatter(y=ecg_plot, name='ECG', line=dict(color='#003f5c')),
                        row=1, col=1
                    )
               
                ppg_plot = list(st.session_state.buffers['ppg'])[-200:]
                if ppg_plot:
                    fig_waveforms.add_trace(
                        go.Scatter(y=ppg_plot, name='PPG', line=dict(color='#ff6f61')),
                        row=1, col=2
                    )
               
                pcg_plot = list(st.session_state.buffers['pcg'])[-200:]
                if pcg_plot:
                    fig_waveforms.add_trace(
                        go.Scatter(y=pcg_plot, name='PCG', line=dict(color='#2ca02c')),
                        row=2, col=1
                    )
               
                dop_plot = list(st.session_state.buffers['dop'])[-200:]
                if dop_plot:
                    fig_waveforms.add_trace(
                        go.Scatter(y=dop_plot, name='Doppler', line=dict(color='#b56576')),
                        row=2, col=2
                    )
               
                resp_plot = list(st.session_state.buffers['resp'])[-200:]
                if resp_plot:
                    fig_waveforms.add_trace(
                        go.Scatter(y=resp_plot, name='Respiratory', line=dict(color='#6a0572')),
                        row=3, col=1
                    )
               
                fusion_plot = list(st.session_state.trend_data['fusion_score'])[-200:]
                if fusion_plot:
                    fig_waveforms.add_trace(
                        go.Scatter(y=fusion_plot, name='Fusion Score', line=dict(color=DANGER)),
                        row=3, col=2
                    )
               
                fig_waveforms.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_waveforms, use_container_width=True)
            else:
                st.info("No waveform data available. Start monitoring to see data.")
       
        with col2:
            st.subheader("Controls")
           
            # Data source selection
            data_gen_mode = st.radio(
                "Data Generation",
                ["Simulation", "Firebase", "Mixed"],
                index=0
            )
           
            # Advanced parameters
            with st.expander("⚙️ Advanced Parameters"):
                hr_range = st.slider("HR Range", 40, 180, (60, 100))
                spo2_range = st.slider("SpO₂ Range", 70, 100, (92, 98))
                noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
           
            # Data recording
            with st.expander("💾 Data Recording"):
                if st.button("Record Session", type="secondary"):
                    st.info(f"Recording {len(st.session_state.session_log)} data points")
               
                if st.session_state.session_log:
                    record_count = len(st.session_state.session_log)
                    st.metric("Records", record_count)
                    st.metric("Duration", f"{record_count / update_frequency:.1f}s")
                   
                    if st.button("Clear Session Data", type="secondary"):
                        st.session_state.session_log = []
                        st.success("Session data cleared")
                        st.rerun()
           
            # Signal quality
            with st.expander("📶 Signal Quality"):
                if st.session_state.session_log:
                    latest = st.session_state.session_log[-1]
                   
                    col_q1, col_q2 = st.columns(2)
                    with col_q1:
                        st.metric("ECG SNR", "42 dB")
                        st.metric("PPG SNR", "38 dB")
                    with col_q2:
                        st.metric("Motion Art.", "Low")
                        st.metric("Baseline", "Stable")
       
        # Detailed Parameters
        st.markdown("---")
        st.subheader("📊 Detailed Parameters")
       
        if st.session_state.session_log:
            latest = st.session_state.session_log[-1]
           
            col1, col2, col3, col4 = st.columns(4)
           
            with col1:
                st.metric("HRV RMSSD", f"{latest.get('hrv_rmssd', 0):.1f} ms")
                st.caption("Heart Rate Variability")
           
            with col2:
                st.metric("ECG Irregularity", f"{latest.get('ecg_irreg', 0):.3f}")
                st.caption("Rhythm Stability")
           
            with col3:
                st.metric("PCG Murmur Index", f"{latest.get('pcg_murmur_index', 0):.3f}")
                st.caption("Heart Sound Analysis")
           
            with col4:
                st.metric("Condition", latest.get('condition', 'normal').replace('_', ' ').title())
                st.caption("AI Classification")
   
    # Tab 3: EchoCardiology
    with tabs[2]:
        st.header("🫀 EchoCardiology Analysis")
       
        # File Upload Section
        st.subheader("📁 Upload Diagnostic Files")
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.subheader("🎥 Echocardiography")
            echo_file = st.file_uploader(
                "Upload Ultrasound Image/Video",
                type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'],
                key="echo_upload"
            )
           
            # Load sample if checkbox enabled and no file uploaded
            if load_samples and not echo_file and os.path.exists(SAMPLE_ECHO):
                try:
                    with open(SAMPLE_ECHO, 'rb') as f:
                        bytes_data = f.read()
                    echo_file = MockUploadedFile(bytes_data, 'echo_sample.jpg', 'image/jpeg')
                    st.info("✅ Loaded sample echo image for testing")
                except Exception as e:
                    st.warning(f"Sample echo file not found: {e}. Please add 'samples/echo_sample.jpg'")
           
            if echo_file:
                with st.spinner("Processing echo image..."):
                    st.session_state.echo_data = echo_preprocessor.preprocess_echo(echo_file)
               
                if st.session_state.echo_data and 'processed_image' in st.session_state.echo_data:
                    st.image(st.session_state.echo_data['processed_image'], caption="Processed Echo Image")
                   
                    quality = st.session_state.echo_data.get('quality_score', 0)
                    st.metric("Image Quality Score", f"{quality:.2%}")
                   
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        st.metric("Brightness", f"{st.session_state.echo_data.get('brightness_mean', 0):.2f}")
                    with col_q2:
                        st.metric("Contrast", f"{st.session_state.echo_data.get('contrast_score', 0):.2f}")
                    with col_q3:
                        st.metric("Sharpness", f"{st.session_state.echo_data.get('sharpness_score', 0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col2:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.subheader("📈 ECG Image (Optional)")
            ecg_image_file = st.file_uploader(
                "Upload ECG Image",
                type=['png', 'jpg', 'jpeg'],
                key="ecg_image_upload"
            )
           
            # Load sample if checkbox enabled and no file uploaded
            if load_samples and not ecg_image_file and os.path.exists(SAMPLE_ECG):
                try:
                    with open(SAMPLE_ECG, 'rb') as f:
                        bytes_data = f.read()
                    ecg_image_file = MockUploadedFile(bytes_data, 'ecg_sample.jpg', 'image/jpeg')
                    st.info("✅ Loaded sample ECG image for testing")
                except Exception as e:
                    st.warning(f"Sample ECG file not found: {e}. Please add 'samples/ecg_sample.jpg'")
           
            if ecg_image_file:
                with st.spinner("Processing ECG image..."):
                    st.session_state.ecg_image_data = ecg_image_preprocessor.preprocess_ecg(ecg_image_file)
               
                if st.session_state.ecg_image_data and 'processed_image' in st.session_state.ecg_image_data:
                    st.image(st.session_state.ecg_image_data['processed_image'], caption="Processed ECG", width=300)
                    st.metric("Estimated Heart Rate", f"{st.session_state.ecg_image_data.get('estimated_hr', 0):.1f} bpm")
                    st.metric("ECG Quality", f"{st.session_state.ecg_image_data.get('quality_score', 0):.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col3:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.subheader("🎵 Heart Sounds (Optional)")
            audio_file = st.file_uploader(
                "Upload Heart Sound Audio",
                type=['wav', 'mp3', 'm4a'],
                key="audio_upload"
            )
           
            # Load sample if checkbox enabled and no file uploaded
            if load_samples and not audio_file and os.path.exists(SAMPLE_AUDIO):
                try:
                    with open(SAMPLE_AUDIO, 'rb') as f:
                        bytes_data = f.read()
                    audio_file = MockUploadedFile(bytes_data, 'heart_sound_sample.wav', 'audio/wav')
                    st.info("✅ Loaded sample heart sound audio for testing")
                except Exception as e:
                    st.warning(f"Sample audio file not found: {e}. Please add 'samples/heart_sound_sample.wav'")
           
            if audio_file:
                with st.spinner("Processing audio..."):
                    st.session_state.audio_data = audio_preprocessor.preprocess_audio(audio_file)
               
                if st.session_state.audio_data:
                    st.metric("Audio Duration", f"{st.session_state.audio_data.get('duration', 0):.1f} s")
                    st.metric("Estimated HR", f"{st.session_state.audio_data.get('estimated_hr', 0):.1f} bpm")
                    st.metric("Audio Quality", f"{st.session_state.audio_data.get('quality_score', 0):.2%}")
                   
                    if st.session_state.audio_data.get('audio_data') is not None:
                        audio_wave = st.session_state.audio_data['audio_data']
                        if len(audio_wave) > 0:
                            fig_audio = go.Figure()
                            time_axis = np.linspace(0, st.session_state.audio_data['duration'], len(audio_wave))
                            fig_audio.add_trace(go.Scatter(
                                x=time_axis[:5000],
                                y=audio_wave[:5000],
                                mode='lines',
                                line=dict(color='#1f77b4', width=1)
                            ))
                            fig_audio.update_layout(
                                title="Audio Waveform",
                                xaxis_title="Time (s)",
                                yaxis_title="Amplitude",
                                height=200
                            )
                            st.plotly_chart(fig_audio, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
       
        # Analysis Button
        st.markdown("---")
        col_analyze1, col_analyze2 = st.columns([1, 3])
       
        with col_analyze1:
            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                if st.session_state.echo_data:
                    with st.spinner("Running AI analysis..."):
                        st.session_state.echo_inference = echo_inference_engine.run_inference(
                            st.session_state.echo_data,
                            st.session_state.ecg_image_data,
                            st.session_state.audio_data
                        )
                    st.success("Analysis complete!")
                else:
                    st.error("Please upload an echocardiography image first.")
       
        with col_analyze2:
            if st.session_state.echo_data:
                st.info(f"✅ Echo data loaded: {st.session_state.echo_data.get('filename', 'Unknown')}")
            if st.session_state.ecg_image_data:
                st.info(f"✅ ECG image loaded")
            if st.session_state.audio_data:
                st.info(f"✅ Audio data loaded")
       
        # Results Display
        if st.session_state.echo_inference:
            st.markdown("---")
            st.subheader("🔬 AI Analysis Results")
           
            results = st.session_state.echo_inference
           
            # Header with triage
            triage = results.get('triage_level', 'Yellow')
            triage_color = {'Green': '🟢', 'Yellow': '🟡', 'Red': '🔴'}.get(triage, '⚪')
           
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        margin-bottom: 20px;">
                <h2 style="margin: 0;">{triage_color} Triage Level: {triage}</h2>
                <p style="margin: 0; opacity: 0.9;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
           
            # Results in columns
            col_r1, col_r2, col_r3 = st.columns(3)
           
            with col_r1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("📐 View Classification")
                view = results.get('view_classification', 'Unknown')
                confidence = results.get('view_confidence', 0)
               
                st.progress(float(confidence))
                st.metric("View", view)
                st.metric("Confidence", f"{confidence:.1%}")
               
                view_explanations = {
                    'AP4': "Apical 4-Chamber view showing all four chambers",
                    'PLAX': "Parasternal Long Axis view for LV assessment",
                    'PSAX': "Parasternal Short Axis view at papillary level",
                    'A4C': "Apical 4-Chamber view alternative",
                    'A2C': "Apical 2-Chamber view"
                }
                st.info(view_explanations.get(view, "Standard echocardiography view"))
                st.markdown('</div>', unsafe_allow_html=True)
           
            with col_r2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("💓 Ejection Fraction")
                ef = results.get('ejection_fraction', 0)
                ef_confidence = results.get('ef_confidence', 0)
               
                fig_ef = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ef,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "LVEF (%)"},
                    gauge={
                        'axis': {'range': [0, 80]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 35], 'color': "red"},
                            {'range': [35, 50], 'color': "orange"},
                            {'range': [50, 80], 'color': "green"}
                        ]
                    }
                ))
                fig_ef.update_layout(height=250)
                st.plotly_chart(fig_ef, use_container_width=True)
               
                st.metric("EF Classification",
                         "Normal" if ef >= 50 else "Mild Reduction" if ef >= 40 else "Severe Reduction")
                st.metric("Confidence", f"{ef_confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
           
            with col_r3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("⚠️ Pathology Flags")
                flags = results.get('pathology_flags', [])
               
                if flags:
                    for flag in flags:
                        if 'Normal' in flag:
                            st.success(f"✅ {flag}")
                        elif 'Possible' in flag:
                            st.warning(f"⚠️ {flag}")
                        else:
                            st.error(f"❌ {flag}")
                else:
                    st.success("✅ No significant abnormalities detected")
               
                st.markdown("---")
                st.subheader("📊 Triage Breakdown")
                triage_scores = results.get('triage_scores', {})
                for level, score in triage_scores.items():
                    if score > 0:
                        st.write(f"{level}: {score}")
                st.markdown('</div>', unsafe_allow_html=True)
           
            # Detailed Findings
            st.markdown("---")
            st.subheader("📋 Detailed Findings")
           
            findings = results.get('findings_summary', [])
            for finding in findings:
                st.write(f"• {finding}")
   
    # Tab 4: Fusion Analysis
    with tabs[3]:
        st.header("🔄 Multi-Modal Fusion Analysis")
       
        if not st.session_state.session_log and not st.session_state.echo_inference:
            st.info("Please collect data in both ECG Multisensor and EchoCardiology tabs first.")
        else:
            # Integrated Risk Assessment
            st.subheader("🎯 Integrated Risk Assessment")
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                # ECG Risk
                if st.session_state.session_log:
                    latest_ecg = st.session_state.session_log[-1]
                    ecg_risk = latest_ecg.get('fusion_tier', 'Unknown')
                    ecg_score = latest_ecg.get('fusion_score', 0)
                   
                    risk_color = SUCCESS if ecg_risk == 'Normal' else (WARNING if ecg_risk == 'At Risk' else DANGER)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 1.2em; color: {SECONDARY};">ECG Multisensor</div>
                            <div style="font-size: 2em; font-weight: bold; color: {risk_color}; margin: 10px 0;">{ecg_risk}</div>
                            <div style="font-size: 1em; color: {risk_color};">Score: {ecg_score:.3f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
           
            with col2:
                # Echo Risk
                if st.session_state.echo_inference:
                    echo_triage = st.session_state.echo_inference.get('triage_level', 'Unknown')
                    echo_ef = st.session_state.echo_inference.get('ejection_fraction', 0)
                   
                    triage_color = SUCCESS if echo_triage == 'Green' else (WARNING if echo_triage == 'Yellow' else DANGER)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 1.2em; color: {SECONDARY};">Echocardiography</div>
                            <div style="font-size: 2em; font-weight: bold; color: {triage_color}; margin: 10px 0;">{echo_triage}</div>
                            <div style="font-size: 1em; color: {triage_color};">EF: {echo_ef:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
           
            with col3:
                # Combined Risk
                combined_risk = "Unknown"
                combined_color = SECONDARY
               
                if st.session_state.session_log and st.session_state.echo_inference:
                    ecg_risk = latest_ecg.get('fusion_tier', 'Normal')
                    echo_triage = st.session_state.echo_inference.get('triage_level', 'Green')
                   
                    if ecg_risk == 'Critical' or echo_triage == 'Red':
                        combined_risk = "Critical"
                        combined_color = DANGER
                    elif ecg_risk == 'At Risk' or echo_triage == 'Yellow':
                        combined_risk = "At Risk"
                        combined_color = WARNING
                    else:
                        combined_risk = "Normal"
                        combined_color = SUCCESS
               
                st.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 1.2em; color: {SECONDARY};">Combined Assessment</div>
                        <div style="font-size: 2.5em; font-weight: bold; color: {combined_color}; margin: 10px 0;">{combined_risk}</div>
                        <div style="font-size: 1em; color: {combined_color};">Multi-Modal Fusion</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
           
            st.markdown("---")
           
            # Correlation Analysis
            st.subheader("📈 Parameter Correlation")
           
            if st.session_state.session_log:
                df_ecg = pd.DataFrame(st.session_state.session_log)
               
                # Select parameters for correlation
                param_options = ['hr', 'spo2', 'resp_rate', 'perf', 'flow', 'ecg_irreg', 'hrv_rmssd', 'fusion_score']
                selected_params = st.multiselect(
                    "Select parameters for correlation analysis",
                    param_options,
                    default=['hr', 'spo2', 'resp_rate']
                )
               
                if len(selected_params) >= 2:
                    corr_data = df_ecg[selected_params].corr()
                   
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_data.values,
                        x=corr_data.columns,
                        y=corr_data.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_data.round(2).values,
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                   
                    fig_corr.update_layout(
                        title="Parameter Correlation Matrix",
                        height=400
                    )
                   
                    st.plotly_chart(fig_corr, use_container_width=True)
           
            st.markdown("---")
           
            # Decision Support
            st.subheader("💡 Clinical Decision Support")
           
            recommendations = []
           
            # ECG-based recommendations
            if st.session_state.session_log:
                latest_ecg = st.session_state.session_log[-1]
                ecg_risk = latest_ecg.get('fusion_tier', 'Normal')
               
                if ecg_risk == 'Critical':
                    recommendations.append("🔴 **Immediate intervention required** based on ECG multisensor data")
                    recommendations.append("• Consider continuous ECG monitoring")
                    recommendations.append("• Review anti-arrhythmic medications")
                elif ecg_risk == 'At Risk':
                    recommendations.append("🟡 **Increased monitoring frequency** recommended")
                    recommendations.append("• Consider 12-lead ECG")
                    recommendations.append("• Review patient medications")
           
            # Echo-based recommendations
            if st.session_state.echo_inference:
                echo_triage = st.session_state.echo_inference.get('triage_level', 'Green')
               
                if echo_triage == 'Red':
                    recommendations.append("🔴 **Urgent cardiology consultation** required")
                    recommendations.append("• Consider echocardiogram repeat within 24 hours")
                    recommendations.append("• Evaluate for inpatient admission")
                elif echo_triage == 'Yellow':
                    recommendations.append("🟡 **Cardiology follow-up** recommended")
                    recommendations.append("• Schedule follow-up in 1-2 weeks")
                    recommendations.append("• Consider cardiac MRI if uncertain")
           
            # Combined recommendations
            if not recommendations:
                recommendations.append("✅ **Continue routine monitoring**")
                recommendations.append("• Schedule routine follow-up")
                recommendations.append("• Maintain current treatment plan")
           
            for rec in recommendations:
                if "🔴" in rec:
                    st.error(rec)
                elif "🟡" in rec:
                    st.warning(rec)
                elif "✅" in rec:
                    st.success(rec)
                else:
                    st.info(rec)
   
    # Tab 5: Trends & Analytics
    with tabs[4]:
        st.header("📊 Trends & Analytics")
       
        if show_trends and st.session_state.trend_data['timestamps']:
            # Create trend DataFrame
            trend_df = pd.DataFrame({
                'timestamp': list(st.session_state.trend_data['timestamps']),
                'hr': list(st.session_state.trend_data['hr']),
                'spo2': list(st.session_state.trend_data['spo2']),
                'perf': list(st.session_state.trend_data['perf']),
                'flow': list(st.session_state.trend_data['flow']),
                'resp_rate': list(st.session_state.trend_data['resp_rate']),
                'fusion_score': list(st.session_state.trend_data['fusion_score'])
            })
           
            # Convert timestamp
            trend_df['datetime'] = pd.to_datetime(trend_df['timestamp'])
           
            # Plot trends
            fig_trends = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Heart Rate', 'SpO₂', 'Perfusion Index', 'Doppler Flow', 'Respiratory Rate', 'Fusion Score'),
                vertical_spacing=0.1
            )
           
            # Add traces
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['hr'], name='HR', line=dict(color=PRIMARY)),
                row=1, col=1
            )
           
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['spo2'], name='SpO2', line=dict(color=SUCCESS)),
                row=1, col=2
            )
           
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['perf'], name='Perfusion', line=dict(color=ACCENT)),
                row=2, col=1
            )
           
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['flow'], name='Flow', line=dict(color=WARNING)),
                row=2, col=2
            )
           
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['resp_rate'], name='Resp Rate', line=dict(color=SECONDARY)),
                row=3, col=1
            )
           
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['fusion_score'], name='Fusion Score', line=dict(color=DANGER)),
                row=3, col=2
            )
           
            fig_trends.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_trends, use_container_width=True)
           
            # Statistical Analysis
            st.markdown("---")
            st.subheader("📈 Statistical Analysis")
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                st.metric("HR Mean ± Std", f"{trend_df['hr'].mean():.1f} ± {trend_df['hr'].std():.1f}")
                st.metric("SpO₂ Mean ± Std", f"{trend_df['spo2'].mean():.1f} ± {trend_df['spo2'].std():.1f}")
           
            with col2:
                st.metric("Perf Mean ± Std", f"{trend_df['perf'].mean():.3f} ± {trend_df['perf'].std():.3f}")
                st.metric("Flow Mean ± Std", f"{trend_df['flow'].mean():.3f} ± {trend_df['flow'].std():.3f}")
           
            with col3:
                st.metric("Resp Rate Mean ± Std", f"{trend_df['resp_rate'].mean():.1f} ± {trend_df['resp_rate'].std():.1f}")
                st.metric("Fusion Score Mean ± Std", f"{trend_df['fusion_score'].mean():.3f} ± {trend_df['fusion_score'].std():.3f}")
           
            # Advanced Analytics
            st.markdown("---")
            st.subheader("🔬 Advanced Analytics")
           
            if st.session_state.session_log:
                df_log = pd.DataFrame(st.session_state.session_log)
               
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': RF_BUNDLE['feature_names'],
                    'importance': RF_BUNDLE['model'].feature_importances_
                }).sort_values('importance', ascending=False)
               
                fig_fi = px.bar(feature_importance, x='importance', y='feature',
                               orientation='h', title='Random Forest Feature Importance',
                               color='importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("No trend data available. Start monitoring to collect data.")
   
    # Tab 6: Reports
    with tabs[5]:
        st.header("📋 Comprehensive Reports")
       
        if not st.session_state.session_log and not st.session_state.echo_inference:
            st.info("No data available for report generation. Please collect data first.")
        else:
            # Report Generation Options
            col1, col2, col3 = st.columns(3)
           
            with col1:
                st.subheader("📄 JSON Report")
                if st.button("Generate JSON Report", use_container_width=True):
                    json_report = st.session_state.report_generator.generate_multi_modal_report(
                        st.session_state.current_patient,
                        st.session_state.session_log,
                        st.session_state.echo_data,
                        st.session_state.ecg_image_data,
                        st.session_state.audio_data,
                        st.session_state.echo_inference
                    )
                   
                    st.download_button(
                        label="Download JSON",
                        data=json_report,
                        file_name=f"medfusion_report_{st.session_state.current_patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
           
            with col2:
                st.subheader("📊 CSV Export")
                if st.button("Generate CSV Report", use_container_width=True):
                    csv_report = st.session_state.report_generator.generate_csv_report(
                        st.session_state.session_log
                    )
                   
                    st.download_button(
                        label="Download CSV",
                        data=csv_report,
                        file_name=f"ecg_data_{st.session_state.current_patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
           
            with col3:
                st.subheader("📋 PDF Report")
                if st.button("Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        pdf_path = st.session_state.report_generator.generate_pdf_report(
                            st.session_state.current_patient,
                            st.session_state.session_log,
                            st.session_state.echo_data,
                            st.session_state.echo_inference
                        )
                       
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                       
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=f"clinical_report_{st.session_state.current_patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
           
            st.markdown("---")
           
            # Report Preview
            st.subheader("🔍 Report Preview")
           
            with st.expander("📋 Preview Multi-Modal Report"):
                if st.session_state.session_log or st.session_state.echo_inference:
                    # Generate preview
                    preview_data = {
                        "patient": st.session_state.current_patient,
                        "ecg_records": len(st.session_state.session_log),
                        "echo_available": st.session_state.echo_inference is not None,
                        "latest_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                   
                    if st.session_state.session_log:
                        latest_ecg = st.session_state.session_log[-1]
                        preview_data.update({
                            "ecg_risk": latest_ecg.get('fusion_tier', 'Unknown'),
                            "ecg_score": latest_ecg.get('fusion_score', 0),
                            "vitals": {
                                "hr": latest_ecg.get('hr', 0),
                                "spo2": latest_ecg.get('spo2', 0),
                                "resp_rate": latest_ecg.get('resp_rate', 0)
                            }
                        })
                   
                    if st.session_state.echo_inference:
                        preview_data.update({
                            "echo_triage": st.session_state.echo_inference.get('triage_level', 'Unknown'),
                            "ejection_fraction": st.session_state.echo_inference.get('ejection_fraction', 0),
                            "view_classification": st.session_state.echo_inference.get('view_classification', 'Unknown')
                        })
                   
                    st.json(preview_data)
                else:
                    st.info("No data available for preview")
           
            # Data Export Options
            st.markdown("---")
            st.subheader("📤 Export Options")
           
            export_col1, export_col2, export_col3 = st.columns(3)
           
            with export_col1:
                if st.button("📧 Send to EMR (Simulated)", use_container_width=True):
                    st.success("✅ Report sent to Electronic Medical Record system (simulated)")
           
            with export_col2:
                if st.button("📱 Share with Team", use_container_width=True):
                    st.success("✅ Report shared with care team (simulated)")
           
            with export_col3:
                if st.button("🖨️ Print Summary", use_container_width=True):
                    st.info("Print functionality would be implemented in production")
   
    # Tab 7: Data Management
    with tabs[6]:
        st.header("🌐 Data Management")
       
        col1, col2 = st.columns(2)
       
        with col1:
            st.subheader("Firebase Operations")
           
            if use_firebase and st.session_state.firebase_manager.initialized:
                if st.button("🔄 Sync All Patient Data", use_container_width=True):
                    with st.spinner("Syncing with Firebase..."):
                        all_data = st.session_state.patient_manager.data_processor.sync_all_patients_data(limit_per_patient=50)
                        if all_data:
                            total_records = sum(len(records) for records in all_data.values())
                            st.success(f"✅ Synced {total_records} records from {len(all_data)} patients")
                           
                            for patient_id, records in all_data.items():
                                st.write(f"**{patient_id}**: {len(records)} records")
                        else:
                            st.warning("No data found in Firebase")
               
                if st.button("📊 Load Historical Data", use_container_width=True):
                    with st.spinner("Loading historical data..."):
                        historical_data = st.session_state.patient_manager.data_processor.fetch_and_process_realtime_data(
                            st.session_state.current_patient['id'], limit=100
                        )
                        if historical_data:
                            st.session_state.session_log.extend(historical_data)
                            st.success(f"✅ Loaded {len(historical_data)} historical records")
                        else:
                            st.warning("No historical data found for this patient")
               
                if st.button("💾 Save Echo Study", use_container_width=True):
                    if st.session_state.echo_inference:
                        study_data = {
                            'patient_id': st.session_state.current_patient['id'],
                            'study_date': datetime.now().isoformat(),
                            'results': st.session_state.echo_inference,
                            'echo_data': {
                                'filename': st.session_state.echo_data.get('filename', '') if st.session_state.echo_data else '',
                                'quality_score': st.session_state.echo_data.get('quality_score', 0) if st.session_state.echo_data else 0
                            } if st.session_state.echo_data else None
                        }
                       
                        success = st.session_state.firebase_manager.push_echo_study(
                            st.session_state.current_patient['id'],
                            study_data
                        )
                       
                        if success:
                            st.success("✅ Study saved to Firebase")
                        else:
                            st.error("❌ Failed to save to Firebase")
                    else:
                        st.warning("No echo study to save")
           
            else:
                st.info("Firebase not configured or not connected")
       
        with col2:
            st.subheader("Database Statistics")
           
            if use_firebase and st.session_state.firebase_manager.initialized:
                st.metric("Connection Status", st.session_state.firebase_manager.connection_status)
               
                if st.session_state.firebase_manager.last_sync_time:
                    st.metric("Last Sync", st.session_state.firebase_manager.last_sync_time.strftime("%H:%M:%S"))
               
                st.metric("Error Count", st.session_state.firebase_manager.error_count)
                st.metric("Cached Patients", len(st.session_state.firebase_manager.patient_cache))
               
                available_patients = st.session_state.firebase_manager.get_available_patients()
                st.metric("Patients in DB", len(available_patients))
           
            st.markdown("---")
            st.subheader("Local Data")
           
            st.metric("Session Records", len(st.session_state.session_log))
            st.metric("Trend Points", len(st.session_state.trend_data['timestamps']))
            st.metric("Echo Studies", 1 if st.session_state.echo_inference else 0)
           
            if st.button("🗑️ Clear All Local Data", type="secondary"):
                st.session_state.session_log = []
                st.session_state.trend_data = {k: deque(maxlen=TREND_WINDOW) for k in st.session_state.trend_data.keys()}
                st.session_state.echo_data = None
                st.session_state.echo_inference = None
                st.session_state.ecg_image_data = None
                st.session_state.audio_data = None
                st.success("All local data cleared")
                st.rerun()
   
    # Footer
    st.markdown("---")
    footer_cols = st.columns(4)
   
    with footer_cols[0]:
        st.caption(f"Version {VERSION}")
   
    with footer_cols[1]:
        st.caption("© 2024 MEDFUSION AI Research")
   
    with footer_cols[2]:
        st.caption("For Research & Educational Use Only")
   
    with footer_cols[3]:
        if use_firebase and st.session_state.firebase_manager.initialized:
            status = st.session_state.firebase_manager.get_connection_status()
            status_emoji = "🟢" if status == "Connected" else "🟡" if "Stale" in status else "🔴"
            st.caption(f"{status_emoji} {status}")
   
    # Data generation thread (simulated real-time data)
    if st.session_state.running and auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.last_update > (1.0 / update_frequency):
            # Generate simulated data
            base_time = time.time()
           
            hr = 70 + 10 * math.sin(base_time * 0.1) + random.uniform(-5, 5)
            spo2 = 97 + random.uniform(-1, 1)
            perf = 0.08 + 0.02 * math.sin(base_time * 0.05) + random.uniform(-0.01, 0.01)
            flow = 0.95 + 0.1 * math.sin(base_time * 0.08) + random.uniform(-0.05, 0.05)
            resp_rate = 16 + 2 * math.sin(base_time * 0.07) + random.uniform(-2, 2)
           
            record = {
                'timestamp': now_iso(),
                'patient_id': st.session_state.current_patient['id'],
                'patient_name': st.session_state.current_patient['name'],
                'age': st.session_state.current_patient['age'],
                'hr': float(hr),
                'spo2': float(spo2),
                'perf': float(perf),
                'flow': float(flow),
                'resp_rate': float(resp_rate),
                'ecg_irreg': random.uniform(0.05, 0.15),
                'hrv_rmssd': random.uniform(30, 50),
                'pcg_murmur_index': random.uniform(0.02, 0.08),
                'condition': 'normal',
                'data_source': 'Simulation'
            }
           
            # Calculate fusion score
            fuse_result = fuse_rule_ml(
                record['age'], record['hr'], record['spo2'], record['perf'],
                record['flow'], record['ecg_irreg'], record['hrv_rmssd'], record['resp_rate']
            )
           
            record['fusion_tier'] = fuse_result['tier']
            record['fusion_score'] = fuse_result['final_score']
            record['ml_confidence'] = fuse_result['ml_conf']
            record['rule_score'] = fuse_result['rule_score']
           
            # Update buffers
            st.session_state.buffers['time'].append(record['timestamp'])
            st.session_state.buffers['ecg'].append(record['hr'] / 100)
            st.session_state.buffers['ppg'].append(record['perf'] * 10)
            st.session_state.buffers['pcg'].append(record['pcg_murmur_index'])
            st.session_state.buffers['dop'].append(record['flow'])
            st.session_state.buffers['resp'].append(record['resp_rate'] / 20)
            st.session_state.buffers['spo2'].append(record['spo2'])
            st.session_state.buffers['perf'].append(record['perf'])
            st.session_state.buffers['hr'].append(record['hr'])
            st.session_state.buffers['resp_rate'].append(record['resp_rate'])
           
            # Add to session log
            st.session_state.session_log.append(record)
           
            # Update trend data
            st.session_state.trend_data['timestamps'].append(record['timestamp'])
            st.session_state.trend_data['hr'].append(record['hr'])
            st.session_state.trend_data['spo2'].append(record['spo2'])
            st.session_state.trend_data['perf'].append(record['perf'])
            st.session_state.trend_data['flow'].append(record['flow'])
            st.session_state.trend_data['resp_rate'].append(record['resp_rate'])
            st.session_state.trend_data['fusion_score'].append(record['fusion_score'])
           
            # Check for alerts
            vital_signs = {
                'hr': record['hr'],
                'spo2': record['spo2'],
                'perf': record['perf'],
                'flow': record['flow'],
                'resp_rate': record['resp_rate']
            }
           
            alerts = st.session_state.alert_system.check_alerts(
                vital_signs, st.session_state.current_patient['id']
            )
           
            st.session_state.last_update = current_time
           
            # Auto-refresh if needed
            if auto_refresh:
                time.sleep(0.1) # Small delay to prevent UI freezing
                st.rerun()
if __name__ == "__main__":
 main()

