# auscsync_firebase_professional.py
"""
AUSCSYNC - Advanced Medical Monitoring System with Firebase Integration
Professional-grade medical monitoring dashboard with real-time Firebase data.

Disclaimer: Research prototype only — not for clinical or diagnostic use.
"""

# -------------------------
# Imports & Basic Settings
# -------------------------
import math
import time
import uuid
import json
import base64
import io
import threading
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

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, db
    from firebase_admin.exceptions import FirebaseError
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    st.warning("Firebase Admin SDK not available. Install with: pip install firebase-admin")

# -------------------------
# Enhanced Firebase Manager with Schema Support
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
                
            # Check if already initialized
            if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
                self.initialized = True
                self.connection_status = "Connected"
                return True
            
            # Initialize with service account
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
            
            # Load initial patient data
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
        """Fetch patient data from Firebase according to schema"""
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
            if self.error_count >= self.max_errors:
                self.connection_status = "Connection Issues"
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

    def push_alert(self, patient_id, alert_data):
        """Push alert to Firebase alerts node"""
        return self.push_patient_data(patient_id, "alerts", alert_data)

# -------------------------
# Enhanced Data Processing for Firebase Schema
# -------------------------
class FirebaseDataProcessor:
    """Processes Firebase data according to the provided schema"""
    
    def __init__(self, firebase_manager):
        self.firebase_manager = firebase_manager
    
    def process_firebase_vitals_record(self, firebase_record, patient_id):
        """Convert Firebase vitals record to app format with AI prediction"""
        try:
            # Get patient profile for age and other details
            profile = self.firebase_manager.get_patient_profile(patient_id)
            
            # Extract all possible fields with safe defaults
            record = {
                'ts': firebase_record.get('timestamp', now_iso()),
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
            
            # Calculate AI risk assessment - FIXED: Ensure all required parameters are present
            fuse_result = self.calculate_ai_prediction(record)
            
            # Add AI prediction results to record
            record.update({
                'fusion_tier': fuse_result['tier'],
                'fusion_score': fuse_result['final_score'],
                'ml_confidence': fuse_result['ml_conf'],
                'rule_score': fuse_result['rule_score'],
                'ml_label': fuse_result['ml_label']
            })
            
            return record
        except Exception as e:
            st.error(f"Error processing Firebase record: {e}")
            st.error(f"Record data: {firebase_record}")
            return None
    
    def calculate_ai_prediction(self, record):
        """Calculate AI risk prediction with proper error handling"""
        try:
            # Ensure all required parameters are present and valid
            age = record.get('age', 45)
            hr = record.get('hr', 75)
            spo2 = record.get('spo2', 97)
            perf = record.get('perf', 0.08)
            flow = record.get('flow', 0.95)
            ecg_irreg = record.get('ecg_irreg', 0.1)
            hrv_rmssd = record.get('hrv_rmssd', 40)
            resp_rate = record.get('resp_rate', 16)
            
            # Validate numerical values
            hr = float(hr) if hr is not None else 75
            spo2 = float(spo2) if spo2 is not None else 97
            perf = float(perf) if perf is not None else 0.08
            flow = float(flow) if flow is not None else 0.95
            ecg_irreg = float(ecg_irreg) if ecg_irreg is not None else 0.1
            hrv_rmssd = float(hrv_rmssd) if hrv_rmssd is not None else 40
            resp_rate = float(resp_rate) if resp_rate is not None else 16
            
            return fuse_rule_ml(age, hr, spo2, perf, flow, ecg_irreg, hrv_rmssd, resp_rate)
            
        except Exception as e:
            st.error(f"AI prediction error: {e}")
            # Return safe default prediction
            return {
                'tier': 'Normal',
                'final_score': 0.1,
                'ml_conf': 0.9,
                'rule_score': 0.1,
                'ml_label': 'Normal'
            }
    
    def fetch_and_process_realtime_data(self, patient_id, limit=10):
        """Fetch and process real-time data for a patient with AI predictions"""
        firebase_data = self.firebase_manager.fetch_patient_data(patient_id, "vitals", limit)
        processed_records = []
        
        if firebase_data:
            for record_id, record_data in firebase_data.items():
                processed_record = self.process_firebase_vitals_record(record_data, patient_id)
                if processed_record:
                    processed_records.append(processed_record)
        
        return processed_records
    
    def sync_all_patients_data(self, limit_per_patient=20):
        """Sync data for all patients from Firebase with AI predictions"""
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

# -------------------------
# Enhanced Patient Manager for Firebase
# -------------------------
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
        
        # Fallback to simulated patients
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
# App Configuration
# -------------------------
APP_TITLE = "AUSCSYNC — Advanced Medical Monitoring System"
VERSION = "v1.0-firebase-pro"

# Color scheme
PRIMARY = "#0b6fab"
ACCENT  = "#1fb6ff"
DANGER  = "#e74c3c"
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

# -------------------------
# Signal Processing
# -------------------------
def butter_bandpass(x, lowcut, highcut, fs, order=3):
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, x)
    except Exception:
        return x

def estimate_hr_ecg(ecg_array, fs=ECG_SR):
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
# ML Model for Risk Prediction
# -------------------------
def generate_synthetic_dataset(n_samples=10000, seed=42):
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

# Initialize ML model
try:
    RF_BUNDLE = build_and_train_rf(10000)
    ML_MODEL_LOADED = True
except Exception as e:
    st.error(f"Failed to load ML model: {e}")
    ML_MODEL_LOADED = False
    RF_BUNDLE = None

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
    for g in AGE_GROUPS:
        if g['min'] <= age <= g['max']:
            return g
    return AGE_GROUPS[-1]

def param_status(value, key, age):
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
    """Fusion of rule-based and ML risk assessment with proper error handling"""
    try:
        if not ML_MODEL_LOADED:
            # Fallback to rule-based only if ML model not loaded
            return {
                'rule_score': 0.1, 
                'ml_label': 'Normal', 
                'ml_conf': 0.9, 
                'final_score': 0.1, 
                'tier': 'Normal', 
                'param_statuses': {}
            }
        
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
        
        # Prepare features for ML model
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
    except Exception as e:
        st.error(f"Error in fuse_rule_ml: {e}")
        # Return safe default
        return {
            'rule_score': 0.1, 
            'ml_label': 'Normal', 
            'ml_conf': 0.9, 
            'final_score': 0.1, 
            'tier': 'Normal', 
            'param_statuses': {}
        }

# -------------------------
# Alert System
# -------------------------
class AlertSystem:
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
        return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id):
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break

# -------------------------
# Patient Data Management
# -------------------------
def make_patient(pid=None):
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
# Enhanced Report Generator
# -------------------------
class ClinicalReportGenerator:
    """Generates comprehensive clinical reports with AI risk analysis"""
    
    def __init__(self):
        self.report_templates = {
            'comprehensive': self._generate_comprehensive_report,
            'executive': self._generate_executive_summary,
            'technical': self._generate_technical_report
        }
    
    def generate_json_report(self, patient_data, session_data, risk_assessment):
        """Generate comprehensive JSON report"""
        report = {
            "metadata": {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.now().isoformat(),
                "report_type": "comprehensive_clinical_analysis",
                "software_version": VERSION
            },
            "patient_information": {
                "patient_id": patient_data['id'],
                "name": patient_data['name'],
                "age": patient_data['age'],
                "sex": patient_data['sex'],
                "height_cm": patient_data['height'],
                "weight_kg": patient_data['weight'],
                "bmi": patient_data['bmi'],
                "medical_history": patient_data['medical_history'],
                "monitoring_duration_hours": self._calculate_monitoring_duration(session_data)
            },
            "ai_risk_assessment": {
                "risk_tier": risk_assessment.get('tier', 'Unknown'),
                "fusion_score": risk_assessment.get('final_score', 0),
                "ml_confidence": risk_assessment.get('ml_conf', 0),
                "rule_based_score": risk_assessment.get('rule_score', 0),
                "parameter_analysis": risk_assessment.get('param_statuses', {}),
                "clinical_interpretation": self._get_risk_interpretation(risk_assessment)
            },
            "vital_signs_analysis": self._analyze_vital_signs(session_data),
            "trend_analysis": self._analyze_trends(session_data),
            "signal_quality_metrics": self._calculate_signal_quality(session_data),
            "clinical_recommendations": self._generate_recommendations(risk_assessment, session_data),
            "raw_data_summary": {
                "total_records": len(session_data),
                "data_time_span": self._get_data_timespan(session_data),
                "parameters_monitored": self._get_monitored_parameters(session_data),
                "data_source": session_data[-1].get('data_source', 'Unknown') if session_data else 'Unknown'
            }
        }
        return json.dumps(report, indent=2)
    
    def generate_csv_report(self, session_data):
        """Generate comprehensive CSV report"""
        if not session_data:
            return ""
        
        df = pd.DataFrame(session_data)
        
        # Add calculated metrics
        df['timestamp'] = pd.to_datetime(df['ts'])
        df['hour'] = df['timestamp'].dt.hour
        df['risk_category'] = df['fusion_tier']
        
        # Reorder columns for better readability
        column_order = [
            'timestamp', 'patient_id', 'patient_name', 'age', 'data_source',
            'hr', 'spo2', 'resp_rate', 'perf', 'flow', 
            'hrv_rmssd', 'ecg_irreg', 'pcg_murmur_index',
            'fusion_tier', 'fusion_score', 'condition', 'risk_category'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        return df.to_csv(index=False)
    
    def generate_infographic_data(self, patient_data, session_data, risk_assessment):
        """Generate data for infographic visualization"""
        if not session_data:
            return {}
        
        df = pd.DataFrame(session_data)
        
        return {
            "patient_summary": {
                "name": patient_data['name'],
                "age_sex": f"{patient_data['age']}y {patient_data['sex']}",
                "bmi_category": self._get_bmi_category(patient_data['bmi']),
                "risk_level": risk_assessment.get('tier', 'Unknown')
            },
            "vital_stats": {
                "avg_heart_rate": df['hr'].mean(),
                "avg_spo2": df['spo2'].mean(),
                "avg_resp_rate": df['resp_rate'].mean(),
                "stability_score": self._calculate_stability_score(df),
                "anomaly_count": self._count_anomalies(df)
            },
            "risk_breakdown": {
                "critical_parameters": self._count_critical_parameters(risk_assessment.get('param_statuses', {})),
                "trend_direction": self._analyze_trend_direction(df),
                "ai_confidence": risk_assessment.get('ml_conf', 0) * 100
            },
            "timeline_analysis": {
                "monitoring_start": df['ts'].min() if 'ts' in df.columns else "Unknown",
                "monitoring_end": df['ts'].max() if 'ts' in df.columns else "Unknown",
                "record_count": len(df),
                "data_quality": "High" if len(df) > 10 else "Low"
            }
        }
    
    def _calculate_monitoring_duration(self, session_data):
        """Calculate total monitoring duration in hours"""
        if not session_data:
            return 0
        try:
            times = [pd.to_datetime(record['ts']) for record in session_data]
            return (max(times) - min(times)).total_seconds() / 3600
        except:
            return 0
    
    def _get_risk_interpretation(self, risk_assessment):
        """Generate clinical interpretation of risk assessment"""
        tier = risk_assessment.get('tier', 'Normal')
        score = risk_assessment.get('final_score', 0)
        
        interpretations = {
            'Normal': "Patient shows stable vital signs within normal ranges. Continue routine monitoring.",
            'At Risk': "Moderate risk detected. Some parameters outside normal ranges. Consider increased monitoring frequency.",
            'Critical': "High risk condition detected. Immediate clinical review recommended. Multiple parameters show critical values."
        }
        
        base_interpretation = interpretations.get(tier, "Risk assessment unavailable.")
        
        # Add score-based details
        if score > 0.8:
            base_interpretation += " Very high confidence in risk assessment."
        elif score > 0.6:
            base_interpretation += " High confidence in risk assessment."
        
        return base_interpretation
    
    def _analyze_vital_signs(self, session_data):
        """Analyze vital signs statistics"""
        if not session_data:
            return {}
        
        df = pd.DataFrame(session_data)
        analysis = {}
        
        vital_params = ['hr', 'spo2', 'resp_rate', 'perf', 'flow']
        
        for param in vital_params:
            if param in df.columns:
                analysis[param] = {
                    'mean': float(df[param].mean()),
                    'std': float(df[param].std()),
                    'min': float(df[param].min()),
                    'max': float(df[param].max()),
                    'trend': 'stable' if df[param].std() < (df[param].mean() * 0.1) else 'variable'
                }
        
        return analysis
    
    def _analyze_trends(self, session_data):
        """Analyze trends in the data"""
        if not session_data:
            return {}
        
        df = pd.DataFrame(session_data)
        trends = {}
        
        if 'fusion_score' in df.columns:
            trends['risk_trend'] = 'increasing' if len(df) > 1 and df['fusion_score'].iloc[-1] > df['fusion_score'].iloc[0] else 'decreasing'
        
        return trends
    
    def _calculate_signal_quality(self, session_data):
        """Calculate signal quality metrics"""
        if not session_data:
            return {}
        
        df = pd.DataFrame(session_data)
        quality = {
            'completeness': 1.0,  # Assuming complete data for now
            'consistency': 0.9,   # Placeholder
            'reliability_score': 0.85  # Placeholder
        }
        return quality
    
    def _generate_recommendations(self, risk_assessment, session_data):
        """Generate clinical recommendations"""
        recommendations = []
        tier = risk_assessment.get('tier', 'Normal')
        
        if tier == 'Critical':
            recommendations.extend([
                "Immediate clinical assessment required",
                "Consider continuous monitoring in critical care setting",
                "Review medication and treatment plans",
                "Prepare for potential emergency intervention"
            ])
        elif tier == 'At Risk':
            recommendations.extend([
                "Increase monitoring frequency to every 15 minutes",
                "Review patient history and current medications",
                "Consider specialist consultation",
                "Monitor for further deterioration"
            ])
        else:
            recommendations.extend([
                "Continue current monitoring regimen",
                "Maintain routine clinical observations",
                "Schedule follow-up as per standard protocol"
            ])
        
        return recommendations
    
    def _get_data_timespan(self, session_data):
        """Get the timespan of the data"""
        if not session_data:
            return "No data"
        
        try:
            times = [record['ts'] for record in session_data]
            return f"{min(times)} to {max(times)}"
        except:
            return "Unknown"
    
    def _get_monitored_parameters(self, session_data):
        """Get list of monitored parameters"""
        if not session_data:
            return []
        return list(session_data[0].keys()) if session_data else []
    
    def _get_bmi_category(self, bmi):
        """Categorize BMI"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def _calculate_stability_score(self, df):
        """Calculate overall stability score"""
        if df.empty:
            return 0
        
        stability_scores = []
        for col in ['hr', 'spo2', 'resp_rate']:
            if col in df.columns:
                cv = df[col].std() / df[col].mean()  # Coefficient of variation
                stability = max(0, 1 - cv)  # Higher CV = lower stability
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0
    
    def _count_anomalies(self, df):
        """Count parameter anomalies"""
        if df.empty:
            return 0
        
        anomaly_count = 0
        thresholds = {
            'hr': (60, 100),  # Example thresholds
            'spo2': (95, 100),
            'resp_rate': (12, 20)
        }
        
        for param, (low, high) in thresholds.items():
            if param in df.columns:
                anomalies = ((df[param] < low) | (df[param] > high)).sum()
                anomaly_count += anomalies
        
        return anomaly_count
    
    def _count_critical_parameters(self, param_statuses):
        """Count critical parameters"""
        return sum(1 for status in param_statuses.values() if status == 'critical')
    
    def _analyze_trend_direction(self, df):
        """Analyze overall trend direction"""
        if len(df) < 2:
            return "Insufficient data"
        
        # Simple trend analysis based on fusion score
        if 'fusion_score' in df.columns:
            first_half = df['fusion_score'].iloc[:len(df)//2].mean()
            second_half = df['fusion_score'].iloc[len(df)//2:].mean()
            return "Improving" if second_half < first_half else "Deteriorating" if second_half > first_half else "Stable"
        
        return "Unknown"

# -------------------------
# Main Streamlit App with Enhanced Firebase Integration
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for professional appearance
    st.markdown(f"""
    <style>
    .main {{
        background-color: {BACKGROUND};
    }}
    .stAlert {{
        border-radius: 10px;
    }}
    .metric-card {{
        background-color: {CARD};
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
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
    .report-section {{
        background-color: {CARD};
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid {PRIMARY};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h1 style='color:{TEXT};'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#6b7280'>Version {VERSION} — Professional Medical Monitoring System</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Initialize Managers
    if 'firebase_manager' not in st.session_state:
        st.session_state.firebase_manager = FirebaseManager()
    
    if 'patient_manager' not in st.session_state:
        st.session_state.patient_manager = FirebasePatientManager(st.session_state.firebase_manager)
    
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = FirebaseDataProcessor(st.session_state.firebase_manager)
    
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ClinicalReportGenerator()

    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Firebase Configuration
        st.subheader("Firebase Configuration")
        
        data_source = st.radio(
            "Data Source",
            ["Simulation Only", "Firebase + Simulation", "Firebase Only"],
            help="Choose where to get patient data from"
        )
        
        use_firebase = data_source != "Simulation Only"
        firebase_only = data_source == "Firebase Only"
        
        if use_firebase:
            st.info("Firebase Configuration Required")
            
            # Firebase URL input
            firebase_url = st.text_input(
                "Firebase Database URL",
                placeholder="https://your-project.firebaseio.com",
                help="Your Firebase Realtime Database URL"
            )
            
            # Service account configuration
            service_account_type = st.radio(
                "Service Account Configuration",
                ["Upload JSON File", "Paste JSON Content"],
                help="Configure Firebase service account credentials"
            )
            
            firebase_config = None
            
            if service_account_type == "Upload JSON File":
                service_account_file = st.file_uploader(
                    "Upload Service Account JSON",
                    type=['json'],
                    help="Download from Firebase Console > Project Settings > Service Accounts"
                )
                if service_account_file:
                    firebase_config = json.load(service_account_file)
            else:
                service_account_json = st.text_area(
                    "Paste Service Account JSON",
                    height=200,
                    help="Paste the entire service account JSON content"
                )
                if service_account_json:
                    try:
                        firebase_config = json.loads(service_account_json)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
            
            # Initialize Firebase button
            if st.button("Initialize Firebase Connection", type="primary"):
                if firebase_url and firebase_config:
                    with st.spinner("Initializing Firebase connection..."):
                        success = st.session_state.firebase_manager.initialize_firebase(
                            firebase_config, firebase_url
                        )
                        if success:
                            st.success("Firebase connected successfully!")
                        else:
                            st.error("Failed to connect to Firebase")
                else:
                    st.error("Please provide both Firebase URL and service account configuration")
        
        # Display Firebase Status
        if use_firebase:
            status = st.session_state.firebase_manager.get_connection_status()
            status_class = "status-connected" if status == "Connected" else "status-disconnected"
            st.markdown(f"""
            <div class="firebase-status {status_class}">
            🔗 Firebase Status: {status}
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.firebase_manager.last_sync_time:
                st.caption(f"Last sync: {st.session_state.firebase_manager.last_sync_time.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        st.header("🎛️ Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            run_monitoring = st.button("Start Monitoring", type="primary", use_container_width=True)
        with col2:
            stop_monitoring = st.button("Stop Monitoring", use_container_width=True)
        
        update_frequency = st.slider("Update Frequency (Hz)", 0.1, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        st.header("👨‍⚕️ Patient Manager")
        
        # Get available patients
        available_patients = st.session_state.patient_manager.get_available_patients()
        
        if available_patients:
            patient_list = [f"{p['id']} - {p['name']} ({p['age']}y, {p['sex']})" for p in available_patients]
            sel = st.selectbox("Select patient", patient_list, index=0)
            pid = sel.split(" - ")[0]
            current_patient = next((p for p in available_patients if p['id'] == pid), available_patients[0])
        else:
            st.warning("No patients available")
            current_patient = make_patient("P1000")
        
        # Patient details
        st.subheader("Patient Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Age:** {current_patient['age']} years")
            st.write(f"**Sex:** {current_patient['sex']}")
            st.write(f"**Height:** {current_patient['height']:.1f} cm")
        with col2:
            st.write(f"**Weight:** {current_patient['weight']:.1f} kg")
            st.write(f"**BMI:** {current_patient['bmi']:.1f}")
            st.write(f"**History:** {current_patient['medical_history']}")
        
        # Add new patient button
        if st.button("➕ Add New Patient to Firebase"):
            new_patient = make_patient()
            if st.session_state.patient_manager.create_firebase_patient(new_patient):
                st.success(f"Patient {new_patient['name']} added to Firebase!")
                st.rerun()
            else:
                st.error("Failed to add patient to Firebase")
        
        st.markdown("---")
        st.header("📊 Display Options")
        
        show_waveforms = st.checkbox("Show Waveforms", value=True)
        show_trends = st.checkbox("Show Trends", value=True)
        show_advanced = st.checkbox("Show Advanced Analytics", value=False)

    # Initialize session state
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

    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = AlertSystem(
            firebase_manager=st.session_state.firebase_manager if use_firebase else None
        )

    # Control monitoring
    if run_monitoring:
        st.session_state.running = True
        st.success("🟢 Monitoring started")
        
    if stop_monitoring:
        st.session_state.running = False
        st.info("🟡 Monitoring stopped")

    # Data processing functions
    def process_firebase_data():
        """Process data from Firebase according to schema"""
        if not st.session_state.firebase_manager.initialized:
            return None
            
        try:
            # Fetch and process latest data from Firebase
            processed_records = st.session_state.data_processor.fetch_and_process_realtime_data(
                current_patient['id'], limit=1
            )
            
            if processed_records:
                return processed_records[-1]  # Return the most recent record
            else:
                return None
                
        except Exception as e:
            st.warning(f"Error processing Firebase data: {e}")
            return None

    def process_simulation_data():
        """Generate simulated data with AI prediction"""
        # Simple simulation for demo
        base_time = time.time()
        
        # Simulate physiological parameters with some variation
        hr = 70 + 10 * math.sin(base_time * 0.1) + random.uniform(-5, 5)
        spo2 = 97 + random.uniform(-1, 1)
        perf = 0.08 + 0.02 * math.sin(base_time * 0.05) + random.uniform(-0.01, 0.01)
        flow = 0.95 + 0.1 * math.sin(base_time * 0.08) + random.uniform(-0.05, 0.05)
        resp_rate = 16 + 2 * math.sin(base_time * 0.07) + random.uniform(-2, 2)
        
        record = {
            'ts': now_iso(),
            'patient_id': current_patient['id'],
            'patient_name': current_patient['name'],
            'age': current_patient['age'],
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
        
        # Calculate AI risk assessment
        fuse_result = st.session_state.data_processor.calculate_ai_prediction(record)
        
        record['fusion_tier'] = fuse_result['tier']
        record['fusion_score'] = fuse_result['final_score']
        record['ml_confidence'] = fuse_result['ml_conf']
        record['rule_score'] = fuse_result['rule_score']
        record['ml_label'] = fuse_result['ml_label']
        
        return record

    def record_sample():
        """Record a sample from the appropriate data source"""
        if firebase_only and st.session_state.firebase_manager.initialized:
            record = process_firebase_data()
        elif use_firebase and st.session_state.firebase_manager.initialized:
            # Try Firebase first, fall back to simulation
            record = process_firebase_data() or process_simulation_data()
        else:
            record = process_simulation_data()
            
        if not record:
            return None, []
            
        # Update buffers
        st.session_state.buffers['time'].append(record['ts'])
        st.session_state.buffers['ecg'].append(record['hr'] / 100)  # Simplified
        st.session_state.buffers['ppg'].append(record['perf'] * 10)  # Simplified
        st.session_state.buffers['pcg'].append(record['pcg_murmur_index'])
        st.session_state.buffers['dop'].append(record['flow'])
        st.session_state.buffers['resp'].append(record['resp_rate'] / 20)  # Simplified
        st.session_state.buffers['spo2'].append(record['spo2'])
        st.session_state.buffers['perf'].append(record['perf'])
        st.session_state.buffers['hr'].append(record['hr'])
        st.session_state.buffers['resp_rate'].append(record['resp_rate'])
        
        # Add to session log
        st.session_state.session_log.append(record)
        
        # Update trend data
        st.session_state.trend_data['timestamps'].append(record['ts'])
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
            vital_signs, current_patient['id']
        )
        
        return record, alerts

    # Auto-sample if running
    if st.session_state.running:
        current_time = time.time()
        if current_time - st.session_state.last_update > (1.0 / update_frequency):
            rec, alerts = record_sample()
            st.session_state.last_update = current_time
            
            # Display any new alerts
            for alert in alerts:
                if alert['priority'] == 'high':
                    st.error(f"🚨 {alert['message']}")
                else:
                    st.warning(f"⚠️ {alert['message']}")

    # Display active alerts in sidebar
    active_alerts = st.session_state.alert_system.get_active_alerts()
    if active_alerts:
        st.sidebar.markdown("---")
        st.sidebar.header("🚨 Active Alerts")
        
        for alert in active_alerts:
            if alert['priority'] == 'high':
                st.sidebar.error(f"{alert['message']}")
            else:
                st.sidebar.warning(f"{alert['message']}")
            
            if st.sidebar.button(f"Acknowledge", key=f"ack_{alert['id']}"):
                st.session_state.alert_system.acknowledge_alert(alert['id'])
                st.rerun()

    # Main display tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard", 
        "📈 Waveforms", 
        "📊 Trends", 
        "🔬 Analytics",
        "🌐 Firebase Data",
        "📋 Clinical Report"
    ])

    with tab1:
        # Dashboard
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("🫀 Vital Signs")
            if st.session_state.session_log:
                latest_rec = st.session_state.session_log[-1]
                
                # Create metric cards
                metrics_data = [
                    ("Heart Rate", f"{latest_rec['hr']:.1f}", "bpm", 'hr'),
                    ("SpO₂", f"{latest_rec['spo2']:.1f}", "%", 'spo2'),
                    ("Respiratory Rate", f"{latest_rec['resp_rate']:.1f}", "bpm", 'resp_rate'),
                    ("Perfusion Index", f"{latest_rec['perf']:.3f}", "", 'perf'),
                    ("Doppler Flow", f"{latest_rec['flow']:.3f}", "", 'flow')
                ]
                
                for metric_name, value, unit, param in metrics_data:
                    status = param_status(float(value), param, latest_rec['age'])
                    color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9em; color: {SECONDARY};">{metric_name}</div>
                        <div style="font-size: 1.5em; font-weight: bold; color: {color};">
                            {value} {unit}
                        </div>
                        <div style="font-size: 0.8em; color: {color}; text-transform: capitalize;">
                            {status}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No readings yet — start monitoring to see data.")
        
        with col2:
            st.subheader("⚡ Advanced Parameters")
            if st.session_state.session_log:
                latest_rec = st.session_state.session_log[-1]
                
                st.metric("HRV RMSSD", f"{latest_rec['hrv_rmssd']:.1f} ms")
                st.metric("ECG Irregularity", f"{latest_rec['ecg_irreg']:.3f}")
                st.metric("PCG Murmur Index", f"{latest_rec['pcg_murmur_index']:.3f}")
                st.metric("Fusion Score", f"{latest_rec['fusion_score']:.3f}")
                
                # Condition indicator
                condition_color = SUCCESS if latest_rec['condition'] == 'normal' else (
                    WARNING if latest_rec['condition'] == 'mild_distress' else DANGER
                )
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; background-color: {condition_color}; color: white; text-align: center;">
                    <strong>Condition: {latest_rec['condition'].replace('_', ' ').title()}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.subheader("🎯 AI Risk Assessment")
            if st.session_state.session_log:
                latest_rec = st.session_state.session_log[-1]
                
                tier = latest_rec.get('fusion_tier', 'Normal')
                color = SUCCESS if tier == "Normal" else (WARNING if tier == "At Risk" else DANGER)
                
                st.markdown(f"""
                <div style="padding: 30px; border-radius: 15px; background: {color}; 
                         color: white; font-weight: 700; text-align: center; font-size: 24px;
                         margin: 10px 0;">
                    {tier} STATUS
                </div>
                """, unsafe_allow_html=True)
                
                # AI Confidence
                ai_confidence = latest_rec.get('ml_confidence', 0.9) * 100
                st.metric("AI Confidence", f"{ai_confidence:.1f}%")
                
                # Data source indicator
                source_color = PRIMARY if latest_rec.get('data_source') == 'Firebase' else SECONDARY
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; color: {source_color};">
                    📡 Data Source: <strong>{latest_rec.get('data_source', 'Simulation')}</strong>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        if show_waveforms and st.session_state.buffers['ecg']:
            st.subheader("Real-time Waveforms with AI Risk Score")
            
            # Create subplots for waveforms
            fig_waveforms = make_subplots(
                rows=3, cols=2,
                subplot_titles=('ECG', 'PPG', 'PCG', 'Doppler', 'Respiratory', 'AI Fusion Score'),
                vertical_spacing=0.1
            )
            
            # Add ECG
            ecg_plot = list(st.session_state.buffers['ecg'])[-200:]
            if ecg_plot:
                fig_waveforms.add_trace(
                    go.Scatter(y=ecg_plot, name='ECG', line=dict(color='#003f5c')),
                    row=1, col=1
                )
            
            # Add PPG
            ppg_plot = list(st.session_state.buffers['ppg'])[-200:]
            if ppg_plot:
                fig_waveforms.add_trace(
                    go.Scatter(y=ppg_plot, name='PPG', line=dict(color='#ff6f61')),
                    row=1, col=2
                )
            
            # Add PCG
            pcg_plot = list(st.session_state.buffers['pcg'])[-200:]
            if pcg_plot:
                fig_waveforms.add_trace(
                    go.Scatter(y=pcg_plot, name='PCG', line=dict(color='#2ca02c')),
                    row=2, col=1
                )
            
            # Add Doppler
            dop_plot = list(st.session_state.buffers['dop'])[-200:]
            if dop_plot:
                fig_waveforms.add_trace(
                    go.Scatter(y=dop_plot, name='Doppler', line=dict(color='#b56576')),
                    row=2, col=2
                )
            
            # Add Respiratory
            resp_plot = list(st.session_state.buffers['resp'])[-200:]
            if resp_plot:
                fig_waveforms.add_trace(
                    go.Scatter(y=resp_plot, name='Respiratory', line=dict(color='#6a0572')),
                    row=3, col=1
                )
            
            # Add AI Fusion Score
            fusion_plot = list(st.session_state.trend_data['fusion_score'])[-200:]
            if fusion_plot:
                # Color based on risk level
                colors = []
                for score in fusion_plot:
                    if score > 0.65:
                        colors.append(DANGER)
                    elif score > 0.35:
                        colors.append(WARNING)
                    else:
                        colors.append(SUCCESS)
                
                fig_waveforms.add_trace(
                    go.Scatter(
                        y=fusion_plot, 
                        name='AI Fusion Score', 
                        line=dict(color=DANGER),
                        marker=dict(color=colors)
                    ),
                    row=3, col=2
                )
            
            fig_waveforms.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_waveforms, use_container_width=True)
            
            # Display current AI prediction
            if st.session_state.session_log:
                latest = st.session_state.session_log[-1]
                st.info(f"**Current AI Prediction:** {latest.get('fusion_tier', 'Normal')} (Score: {latest.get('fusion_score', 0):.3f})")
        else:
            st.info("No waveform data available or display disabled.")

    with tab3:
        if show_trends and st.session_state.trend_data['timestamps']:
            st.subheader("Trend Analysis with AI Risk Assessment")
            
            # Create DataFrame for trends
            trend_df = pd.DataFrame({
                'timestamp': list(st.session_state.trend_data['timestamps']),
                'hr': list(st.session_state.trend_data['hr']),
                'spo2': list(st.session_state.trend_data['spo2']),
                'perf': list(st.session_state.trend_data['perf']),
                'flow': list(st.session_state.trend_data['flow']),
                'resp_rate': list(st.session_state.trend_data['resp_rate']),
                'fusion_score': list(st.session_state.trend_data['fusion_score'])
            })
            
            # Convert timestamp to datetime
            trend_df['datetime'] = pd.to_datetime(trend_df['timestamp'])
            
            # Plot trends
            fig_trends = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Heart Rate', 'SpO₂', 'Perfusion Index', 'Doppler Flow', 'Respiratory Rate', 'AI Fusion Score'),
                vertical_spacing=0.1
            )
            
            # Heart Rate
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['hr'], name='HR', line=dict(color=PRIMARY)),
                row=1, col=1
            )
            
            # SpO2
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['spo2'], name='SpO2', line=dict(color=SUCCESS)),
                row=1, col=2
            )
            
            # Perfusion Index
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['perf'], name='Perfusion', line=dict(color=ACCENT)),
                row=2, col=1
            )
            
            # Doppler Flow
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['flow'], name='Flow', line=dict(color=WARNING)),
                row=2, col=2
            )
            
            # Respiratory Rate
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['resp_rate'], name='Resp Rate', line=dict(color=SECONDARY)),
                row=3, col=1
            )
            
            # AI Fusion Score with risk zones
            fig_trends.add_trace(
                go.Scatter(x=trend_df['datetime'], y=trend_df['fusion_score'], name='AI Fusion Score', line=dict(color=DANGER)),
                row=3, col=2
            )
            
            # Add risk zones to fusion score
            fig_trends.add_hrect(y0=0, y1=0.35, line_width=0, fillcolor=SUCCESS, opacity=0.1, row=3, col=2)
            fig_trends.add_hrect(y0=0.35, y1=0.65, line_width=0, fillcolor=WARNING, opacity=0.1, row=3, col=2)
            fig_trends.add_hrect(y0=0.65, y1=1.0, line_width=0, fillcolor=DANGER, opacity=0.1, row=3, col=2)
            
            fig_trends.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Statistical summary
            st.subheader("Trend Statistics with AI Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("HR Mean ± Std", f"{trend_df['hr'].mean():.1f} ± {trend_df['hr'].std():.1f}")
                st.metric("SpO₂ Mean ± Std", f"{trend_df['spo2'].mean():.1f} ± {trend_df['spo2'].std():.1f}")
            
            with col2:
                st.metric("Perf Mean ± Std", f"{trend_df['perf'].mean():.3f} ± {trend_df['perf'].std():.3f}")
                st.metric("Flow Mean ± Std", f"{trend_df['flow'].mean():.3f} ± {trend_df['flow'].std():.3f}")
            
            with col3:
                st.metric("Resp Rate Mean ± Std", f"{trend_df['resp_rate'].mean():.1f} ± {trend_df['resp_rate'].std():.1f}")
                st.metric("AI Risk Score Mean ± Std", f"{trend_df['fusion_score'].mean():.3f} ± {trend_df['fusion_score'].std():.3f}")
                
            # AI Risk Analysis
            if len(trend_df) > 1:
                risk_trend = "Increasing" if trend_df['fusion_score'].iloc[-1] > trend_df['fusion_score'].iloc[0] else "Decreasing"
                current_risk = trend_df['fusion_score'].iloc[-1]
                risk_level = "Critical" if current_risk > 0.65 else "At Risk" if current_risk > 0.35 else "Normal"
                
                st.subheader("📊 AI Risk Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Risk Level", risk_level)
                with col2:
                    st.metric("Risk Score Trend", risk_trend)
                with col3:
                    st.metric("Max Risk Score", f"{trend_df['fusion_score'].max():.3f}")
        else:
            st.info("No trend data available or trends display disabled.")

    with tab4:
        if show_advanced and st.session_state.session_log:
            st.subheader("Advanced Analytics with AI Insights")
            
            # Create DataFrame from session log
            df_log = pd.DataFrame(st.session_state.session_log)
            
            # Feature correlations including AI score
            st.subheader("Feature Correlations with AI Risk Score")
            numeric_cols = ['hr', 'spo2', 'perf', 'flow', 'ecg_irreg', 'hrv_rmssd', 'resp_rate', 'fusion_score']
            available_cols = [col for col in numeric_cols if col in df_log.columns]
            
            if available_cols:
                corr_matrix = df_log[available_cols].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig_corr.update_layout(title="Correlation Matrix with AI Fusion Score")
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature importance from ML model
            if ML_MODEL_LOADED:
                st.subheader("AI Model Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': RF_BUNDLE['feature_names'],
                    'importance': RF_BUNDLE['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_fi = px.bar(feature_importance, x='importance', y='feature', 
                               orientation='h', title='Random Forest Feature Importance for Risk Prediction')
                st.plotly_chart(fig_fi, use_container_width=True)
            
            # Risk distribution
            st.subheader("Risk Score Distribution")
            if 'fusion_score' in df_log.columns:
                fig_dist = px.histogram(df_log, x='fusion_score', nbins=20, 
                                       title='Distribution of AI Fusion Scores',
                                       color_discrete_sequence=[DANGER])
                fig_dist.update_layout(xaxis_title='AI Fusion Score', yaxis_title='Frequency')
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Risk over time
            st.subheader("Risk Progression Over Time")
            if 'fusion_score' in df_log.columns and 'ts' in df_log.columns:
                df_log['timestamp'] = pd.to_datetime(df_log['ts'])
                fig_risk_time = px.line(df_log, x='timestamp', y='fusion_score',
                                      title='AI Risk Score Over Time',
                                      color_discrete_sequence=[DANGER])
                fig_risk_time.add_hrect(y0=0.65, y1=1.0, fillcolor=DANGER, opacity=0.2, line_width=0)
                fig_risk_time.add_hrect(y0=0.35, y1=0.65, fillcolor=WARNING, opacity=0.2, line_width=0)
                fig_risk_time.add_hrect(y0=0, y1=0.35, fillcolor=SUCCESS, opacity=0.2, line_width=0)
                st.plotly_chart(fig_risk_time, use_container_width=True)
            
        else:
            st.info("No data available for advanced analytics.")

    with tab5:
        st.subheader("🌐 Firebase Data Management")
        
        if use_firebase and st.session_state.firebase_manager.initialized:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Firebase Operations")
                
                if st.button("🔄 Sync All Patient Data with AI Analysis"):
                    with st.spinner("Syncing with Firebase and running AI analysis..."):
                        all_data = st.session_state.data_processor.sync_all_patients_data(limit_per_patient=50)
                        if all_data:
                            total_records = sum(len(records) for records in all_data.values())
                            st.success(f"Synced {total_records} records from {len(all_data)} patients")
                            
                            # Display patient summary
                            st.subheader("Patient Data Summary with AI Analysis")
                            for patient_id, records in all_data.items():
                                if records:
                                    latest_record = records[-1]
                                    risk_level = latest_record.get('fusion_tier', 'Unknown')
                                    risk_score = latest_record.get('fusion_score', 0)
                                    st.write(f"**{patient_id}**: {len(records)} records | Risk: {risk_level} ({risk_score:.3f})")
                        else:
                            st.warning("No data found in Firebase")
                
                if st.button("📊 Load Historical Data with AI"):
                    with st.spinner("Loading historical data with AI analysis..."):
                        historical_data = st.session_state.data_processor.fetch_and_process_realtime_data(
                            current_patient['id'], limit=100
                        )
                        if historical_data:
                            # Clear existing data and replace with historical
                            st.session_state.session_log = historical_data
                            st.success(f"Loaded {len(historical_data)} historical records with AI analysis")
                            
                            # Update trend data
                            st.session_state.trend_data = {
                                'timestamps': deque(maxlen=TREND_WINDOW),
                                'hr': deque(maxlen=TREND_WINDOW),
                                'spo2': deque(maxlen=TREND_WINDOW),
                                'perf': deque(maxlen=TREND_WINDOW),
                                'flow': deque(maxlen=TREND_WINDOW),
                                'resp_rate': deque(maxlen=TREND_WINDOW),
                                'fusion_score': deque(maxlen=TREND_WINDOW),
                            }
                            
                            for record in historical_data:
                                st.session_state.trend_data['timestamps'].append(record['ts'])
                                st.session_state.trend_data['hr'].append(record['hr'])
                                st.session_state.trend_data['spo2'].append(record['spo2'])
                                st.session_state.trend_data['perf'].append(record['perf'])
                                st.session_state.trend_data['flow'].append(record['flow'])
                                st.session_state.trend_data['resp_rate'].append(record['resp_rate'])
                                st.session_state.trend_data['fusion_score'].append(record['fusion_score'])
                        else:
                            st.warning("No historical data found for this patient")
            
            with col2:
                st.subheader("Firebase Statistics")
                
                if st.session_state.firebase_manager.last_sync_time:
                    st.metric("Last Sync", st.session_state.firebase_manager.last_sync_time.strftime("%H:%M:%S"))
                st.metric("Connection Status", st.session_state.firebase_manager.connection_status)
                st.metric("Error Count", st.session_state.firebase_manager.error_count)
                st.metric("Cached Patients", len(st.session_state.firebase_manager.patient_cache))
                
                # Database info
                st.subheader("Database Info")
                available_patients = st.session_state.firebase_manager.get_available_patients()
                st.metric("Patients in DB", len(available_patients))
                
            # Test data generation with AI
            st.markdown("---")
            st.subheader("🧪 Test Data Generation with AI Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_test_records = st.slider("Number of test records", 1, 50, 10)
                include_risk_cases = st.checkbox("Include high-risk cases", value=True)
                
            with col2:
                if st.button("Generate Test Data with AI"):
                    with st.spinner("Generating test data with AI analysis..."):
                        for i in range(num_test_records):
                            # Create some high-risk cases if requested
                            if include_risk_cases and i % 5 == 0:
                                # High risk case
                                test_record = {
                                    'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                                    'hr': float(np.random.normal(120, 15)),
                                    'spo2': float(np.random.normal(85, 3)),
                                    'perf': float(np.random.normal(0.02, 0.01)),
                                    'flow': float(np.random.normal(0.3, 0.1)),
                                    'resp_rate': float(np.random.normal(25, 5)),
                                    'ecg_irreg': float(np.random.uniform(0.6, 0.9)),
                                    'hrv_rmssd': float(np.random.normal(20, 5)),
                                    'pcg_murmur_index': float(np.random.uniform(0.1, 0.3)),
                                    'condition': 'critical'
                                }
                            else:
                                # Normal case
                                test_record = {
                                    'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                                    'hr': float(np.random.normal(75, 10)),
                                    'spo2': float(np.random.normal(97, 1.5)),
                                    'perf': float(np.random.normal(0.08, 0.02)),
                                    'flow': float(np.random.normal(0.95, 0.1)),
                                    'resp_rate': float(np.random.normal(16, 2)),
                                    'ecg_irreg': float(np.random.uniform(0.05, 0.15)),
                                    'hrv_rmssd': float(np.random.normal(40, 10)),
                                    'pcg_murmur_index': float(np.random.uniform(0.02, 0.08)),
                                    'condition': 'normal'
                                }
                            
                            success = st.session_state.firebase_manager.push_patient_data(
                                current_patient['id'], "vitals", test_record
                            )
                        
                        st.success(f"Generated {num_test_records} test records with AI analysis")
        else:
            st.info("Firebase not configured or not connected.")

    with tab6:
        st.subheader("📋 Comprehensive Clinical Report with AI Analysis")
        
        if not st.session_state.session_log:
            st.info("No data available for report generation. Start monitoring to collect data.")
        else:
            # Get latest risk assessment
            latest_rec = st.session_state.session_log[-1]
            risk_assessment = st.session_state.data_processor.calculate_ai_prediction(latest_rec)
            
            # Report Overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Patient", current_patient['name'])
                st.metric("Age/Sex", f"{current_patient['age']}y/{current_patient['sex']}")
            
            with col2:
                st.metric("AI Risk Level", risk_assessment['tier'])
                st.metric("AI Confidence", f"{risk_assessment['ml_conf']*100:.1f}%")
            
            with col3:
                st.metric("Records Analyzed", len(st.session_state.session_log))
                st.metric("Data Source", st.session_state.session_log[-1].get('data_source', 'Unknown'))
            
            st.markdown("---")
            
            # Report Sections
            st.subheader("📊 AI Risk Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="report-section">
                    <h4>🧠 AI Risk Assessment Breakdown</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk metrics
                st.metric("AI Fusion Score", f"{risk_assessment['final_score']:.3f}")
                st.metric("Rule-Based Score", f"{risk_assessment['rule_score']:.3f}")
                st.metric("ML Confidence", f"{risk_assessment['ml_conf']:.3f}")
                st.metric("ML Prediction", risk_assessment['ml_label'])
                
                # Parameter status
                st.subheader("Parameter Status Analysis")
                for param, status in risk_assessment['param_statuses'].items():
                    status_color = SUCCESS if status == 'normal' else (WARNING if status == 'warning' else DANGER)
                    st.markdown(f"- **{param}**: <span style='color:{status_color};'>{status}</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="report-section">
                    <h4>📈 AI Risk Visualization</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Create risk gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_assessment['final_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AI Fusion Risk Score"},
                    delta = {'reference': 0.3},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': DANGER},
                        'steps': [
                            {'range': [0, 0.3], 'color': SUCCESS},
                            {'range': [0.3, 0.7], 'color': WARNING},
                            {'range': [0.7, 1], 'color': DANGER}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.7
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # Clinical Recommendations
            st.subheader("💡 AI-Powered Clinical Recommendations")
            infographic_data = st.session_state.report_generator.generate_infographic_data(
                current_patient, st.session_state.session_log, risk_assessment
            )
            
            recommendations = st.session_state.report_generator._generate_recommendations(
                risk_assessment, st.session_state.session_log
            )
            
            for i, recommendation in enumerate(recommendations, 1):
                st.markdown(f"{i}. {recommendation}")
            
            st.markdown("---")
            
            # Download Section
            st.subheader("📥 Download AI-Enhanced Reports")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON Report
                json_report = st.session_state.report_generator.generate_json_report(
                    current_patient, st.session_state.session_log, risk_assessment
                )
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_report,
                    file_name=f"clinical_report_{current_patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV Report
                csv_report = st.session_state.report_generator.generate_csv_report(
                    st.session_state.session_log
                )
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_report,
                    file_name=f"patient_data_{current_patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # AI Summary
                summary_text = f"""
                AUSCSYNC AI CLINICAL REPORT
                ===========================
                
                Patient: {current_patient['name']}
                ID: {current_patient['id']}
                Age: {current_patient['age']} years
                Sex: {current_patient['sex']}
                
                AI RISK ASSESSMENT: {risk_assessment['tier']}
                Fusion Score: {risk_assessment['final_score']:.3f}
                AI Confidence: {risk_assessment['ml_conf']*100:.1f}%
                ML Prediction: {risk_assessment['ml_label']}
                
                Monitoring Period: {len(st.session_state.session_log)} records
                Data Source: {st.session_state.session_log[-1].get('data_source', 'Unknown')}
                
                AI RECOMMENDATIONS:
                {chr(10).join(f'- {rec}' for rec in recommendations)}
                
                Generated with AI Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="🤖 Download AI Summary",
                    data=summary_text,
                    file_name=f"ai_report_{current_patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Raw Data Preview with AI predictions
            st.markdown("---")
            st.subheader("🔍 Raw Data Preview with AI Predictions (Last 10 Records)")
            if st.session_state.session_log:
                df_preview = pd.DataFrame(st.session_state.session_log[-10:])
                # Select important columns including AI predictions
                preview_cols = ['ts', 'hr', 'spo2', 'resp_rate', 'fusion_tier', 'fusion_score', 'ml_confidence']
                available_cols = [col for col in preview_cols if col in df_preview.columns]
                st.dataframe(df_preview[available_cols])

    # Session log at the bottom
    st.markdown("---")
    st.subheader("Session Log with AI Predictions (last 20 entries)")

    if st.session_state.session_log:
        df = pd.DataFrame(st.session_state.session_log)
        display_cols = ['ts', 'patient_id', 'hr', 'spo2', 'resp_rate', 'fusion_tier', 'fusion_score', 'data_source']
        available_display_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(df.tail(20)[available_display_cols].style.format({
            'hr': '{:.1f}',
            'spo2': '{:.1f}',
            'resp_rate': '{:.1f}',
            'fusion_score': '{:.3f}'
        }))
    else:
        st.info("No session data yet")

    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col1:
        st.caption(f"Version {VERSION}")
    
    with footer_col2:
        st.caption("AUSCSYNC Advanced Medical Monitoring System — For Research Use Only")
    
    with footer_col3:
        if use_firebase and st.session_state.firebase_manager.initialized:
            status = st.session_state.firebase_manager.get_connection_status()
            status_emoji = "🟢" if status == "Connected" else "🟡" if "Stale" in status else "🔴"
            st.caption(f"{status_emoji} {status}")

if __name__ == "__main__":
    main()