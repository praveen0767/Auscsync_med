#!/usr/bin/env python3
"""
backend.py

Flask backend to manage Arduino diagnosis sessions and upload admin PDFs to Firebase.

Features:
- CORS enabled (dev-friendly; restrict origins in production).
- Serves static files from ./static (so you can drop index.html there).
- GET /admin_details/latest -> returns adminDetails/latest metadata from RTDB.
- POST /init -> optional runtime Firebase initialization.
- CRUD: patients list/create, start/stop sessions, list sessions, upload admin PDF.
- Simple API key protection via X-API-KEY header or apiKey query param.

Run:
  pip install flask firebase-admin pyserial flask-cors python-dotenv
  export SERVICE_ACCOUNT_JSON="/path/to/service-account.json"
  export DATABASE_URL="https://your-db.firebaseio.com/"
  export STORAGE_BUCKET="your-project.appspot.com"
  export BACKEND_API_KEY="your-secret-key"
  python backend.py --host 0.0.0.0 --port 5000
"""

import os
import time
import threading
import uuid
import re
import math
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Firebase admin
import firebase_admin
from firebase_admin import credentials, db, storage as fb_storage

# Serial (optional)
try:
    import serial
except Exception:
    serial = None  # allow simulation mode when pyserial not available

# CORS
from flask_cors import CORS

# Load .env optionally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Configuration via env ----------
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", None)  # optional but recommended for PDF upload
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "changeme")
DEFAULT_BAUD = int(os.environ.get("DEFAULT_BAUD", "115200"))

# ---------- Firebase init helper ----------
firebase_app = None
db_root = None
storage_bucket = None
firebase_lock = threading.Lock()

def initialize_firebase(sa_json=None, database_url=None, storage_bucket_name=None):
    global firebase_app, db_root, storage_bucket
    with firebase_lock:
        sa = sa_json or SERVICE_ACCOUNT_JSON
        dburl = database_url or DATABASE_URL
        sb = storage_bucket_name or STORAGE_BUCKET
        if not sa or not dburl:
            raise RuntimeError("SERVICE_ACCOUNT_JSON and DATABASE_URL required to initialize Firebase.")
        cred = credentials.Certificate(sa)
        if firebase_admin._apps:
            firebase_app = firebase_admin.get_app()
        else:
            firebase_app = firebase_admin.initialize_app(cred, {"databaseURL": dburl, "storageBucket": sb} if sb else {"databaseURL": dburl})
        db_root = db.reference("/")
        if sb:
            storage_bucket = fb_storage.bucket()
        else:
            storage_bucket = None
    return True

# Try auto-initialize if env provided
try:
    if SERVICE_ACCOUNT_JSON and DATABASE_URL:
        initialize_firebase()
        print("Firebase initialized from environment variables.")
except Exception as e:
    print("Firebase auto-init failed:", e)

# ---------- Flask app ----------
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Allow all origins by default (DEV). For production restrict origins.

# ---------- Simple API key decorator ----------
def require_api_key(fn):
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-KEY") or request.args.get("apiKey")
        if not key or key != BACKEND_API_KEY:
            return jsonify({"error":"Unauthorized - invalid API key"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

# ---------- Utility: list patients ----------
def list_patients():
    if not db_root:
        return {}
    pts = db_root.child("patients").get() or {}
    return pts

# ---------- Session manager ----------
sessions = {}  # session_id -> { meta, stop_event, thread }
sessions_lock = threading.Lock()

num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_line_to_values(line):
    toks = num_re.findall(line)
    if len(toks) < 3:
        return None
    try:
        ecg = float(toks[0])
        ppg = float(toks[1])
        spo2 = float(toks[2])
        return {"ecg": ecg, "ppg": ppg, "spo2": spo2}
    except:
        return None

def estimate_hr_from_ecg(ecg_values, fs=100.0):
    try:
        import numpy as np
    except Exception:
        return None
    if not ecg_values or len(ecg_values) < 10:
        return None
    arr = np.array(ecg_values)
    arr = arr - np.median(arr)
    thresh = np.mean(arr) + 0.5 * np.std(arr)
    peaks = []
    for i in range(1, len(arr)-1):
        if arr[i] > thresh and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks.append(i)
    if len(peaks) < 2:
        return None
    rr_samples = np.diff(peaks)
    rr_sec = rr_samples / fs
    mean_rr = rr_sec.mean()
    if mean_rr <= 0:
        return None
    hr = 60.0 / mean_rr
    return float(hr)

def session_worker(session_id):
    """
    Worker thread: reads from serial (or simulates) and pushes readings into:
      patients/{patient_id}/sessions/{session_id}/readings/{pushId}
    Writes session summary to meta at end.
    """
    with sessions_lock:
        sess = sessions.get(session_id)
        if not sess:
            print(f"[{session_id}] Session not found in sessions map.")
            return
    meta = sess.get("meta", {})
    patient_id = meta.get("patient_id")
    duration = int(meta.get("duration_sec", 40))
    sample_interval_ms = int(meta.get("sample_interval_ms", 1000))
    source = meta.get("source", "simulate")
    port = meta.get("port")
    baud = int(meta.get("baud", DEFAULT_BAUD))
    stop_event = sess.get("stop_event")

    start_ts = time.time()
    samples_local = []

    # push helper
    def push_reading_to_db(reading):
        try:
            ref = db_root.child("patients").child(patient_id).child("sessions").child(session_id).child("readings").push()
            payload = {
                "ts": int(time.time() * 1000),
                "ecg": float(reading.get("ecg")) if reading.get("ecg") is not None else None,
                "ppg": float(reading.get("ppg")) if reading.get("ppg") is not None else None,
                "spo2": float(reading.get("spo2")) if reading.get("spo2") is not None else None,
                "raw_line": reading.get("_raw", None)
            }
            ref.set(payload)
            return True
        except Exception as e:
            print(f"[{session_id}] DB push error: {e}")
            return False

    use_serial = (source == "serial" and serial is not None and port)
    ser = None
    if use_serial:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(1.0)
            ser.reset_input_buffer()
            print(f"[{session_id}] Serial opened on {port}@{baud}")
        except Exception as e:
            print(f"[{session_id}] Failed to open serial {port}: {e}")
            use_serial = False

    try:
        next_sample_ts = time.time()
        while (time.time() - start_ts) < duration and not stop_event.is_set():
            now = time.time()
            if now < next_sample_ts:
                time.sleep(min(0.05, next_sample_ts - now))
                continue

            if use_serial and ser:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception:
                    line = ""
                if not line:
                    reading = {"_raw": None}
                else:
                    parsed = parse_line_to_values(line)
                    reading = parsed if parsed else {"_raw": line}
                    reading["_raw"] = line
            else:
                # simulate
                import random
                t = time.time()
                ecg = 500 + 100 * math.sin(t*2*math.pi*1) + (random.random()-0.5)*30
                ppg = 200 + 40 * math.sin(t*2*math.pi*1.2) + (random.random()-0.5)*10
                spo2 = 96 + (random.random()-0.5)*2
                reading = {"ecg": ecg, "ppg": ppg, "spo2": spo2, "_raw": None}

            samples_local.append({"t": time.time(), **{k:v for k,v in reading.items() if not k.startswith("_")}, "_raw": reading.get("_raw")})
            push_reading_to_db(reading)
            next_sample_ts += (sample_interval_ms / 1000.0)
    finally:
        if ser:
            try:
                ser.close()
            except:
                pass

    # compute summary & write meta
    try:
        ecg_vals = [s["ecg"] for s in samples_local if s.get("ecg") is not None]
        ppg_vals = [s["ppg"] for s in samples_local if s.get("ppg") is not None]
        spo2_vals = [s["spo2"] for s in samples_local if s.get("spo2") is not None]

        hr_est = estimate_hr_from_ecg(ecg_vals, fs=100.0) if ecg_vals else None
        spo2_avg = round(sum(spo2_vals)/len(spo2_vals), 2) if spo2_vals else None
        ecg_mean = round(sum(ecg_vals)/len(ecg_vals), 4) if ecg_vals else None
        ppg_mean = round(sum(ppg_vals)/len(ppg_vals), 4) if ppg_vals else None

        condition = "normal"
        if spo2_avg is not None and spo2_avg < 90:
            condition = "critical"
        elif spo2_avg is not None and spo2_avg < 94:
            condition = "warning"
        if hr_est is not None and (hr_est > 120 or hr_est < 40):
            condition = "critical"
        elif hr_est is not None and (hr_est > 100 or hr_est < 50):
            if condition != "critical":
                condition = "warning"

        hr_norm = 0.0
        if hr_est is not None:
            hr_norm = min(max((hr_est - 50)/70.0, 0.0), 1.0)
        spo2_norm = 1.0 - (0.0 if spo2_avg is None else min(max((95.0 - spo2_avg)/20.0, 0.0), 1.0))
        fusion_score = round(0.6*hr_norm + 0.4*(1-spo2_norm), 3)
        fusion_tier = "Normal"
        if fusion_score > 0.7: fusion_tier = "High"
        elif fusion_score > 0.4: fusion_tier = "Moderate"

        ts_iso = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
        summary = {
            "timestamp": ts_iso,
            "hr": round(hr_est,1) if hr_est is not None else None,
            "spo2": spo2_avg,
            "ecg_mean": ecg_mean,
            "ppg_mean": ppg_mean,
            "condition": condition,
            "fusion_tier": fusion_tier,
            "fusion_score": fusion_score,
            "data_source": "serial" if use_serial else "simulate",
            "samples_count": len(samples_local)
        }

        # write meta
        meta_ref = db_root.child("patients").child(patient_id).child("sessions").child(session_id).child("meta")
        meta_ref.update({**meta, **summary, "endedAt": int(time.time()*1000)})
    except Exception as e:
        print(f"[{session_id}] Error computing/writing summary: {e}")

    # push an alert if needed
    try:
        if summary.get("condition") in ("warning", "critical"):
            alerts_ref = db_root.child("patients").child(patient_id).child("alerts")
            alert_obj = {
                "id": f"alert-{int(time.time())}",
                "type": "vitals_alert",
                "message": f"{summary['condition'].upper()} detected: hr={summary.get('hr')}, spo2={summary.get('spo2')}",
                "priority": "high" if summary['condition']=="critical" else "medium",
                "timestamp": summary['timestamp'],
                "acknowledged": False
            }
            alerts_ref.push(alert_obj)
    except Exception as e:
        print(f"[{session_id}] Error creating alert: {e}")

    # mark session finished
    try:
        db_root.child("patients").child(patient_id).child("sessions").child(session_id).child("status").set("finished")
    except Exception:
        pass

    with sessions_lock:
        sessions.pop(session_id, None)
    print(f"[{session_id}] Worker finished.")

# ---------- Endpoints ----------

@app.route('/', methods=['GET'])
def root_index():
    # Serve the static index.html from ./static/index.html
    if app.static_folder:
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"ok": True, "message": "Backend running. No static folder configured."})

@app.route("/init", methods=["POST"])
@require_api_key
def init_endpoint():
    """
    POST /init
    JSON body: { serviceAccountJson: "/path/sa.json", databaseUrl: "...", storageBucket(optional): "project.appspot.com" }
    """
    data = request.get_json() or {}
    sa = data.get("serviceAccountJson")
    dburl = data.get("databaseUrl")
    sb = data.get("storageBucket")
    try:
        initialize_firebase(sa, dburl, sb)
        return jsonify({"ok": True, "message":"Firebase initialized"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/patients", methods=["GET", "POST"])
@require_api_key
def patients_endpoint():
    if not db_root:
        return jsonify({"error": "Firebase not initialized"}), 500
    if request.method == "GET":
        pts = db_root.child("patients").get() or {}
        return jsonify({"patients": pts}), 200
    else:
        body = request.get_json() or {}
        name = body.get("name")
        if not name:
            return jsonify({"error":"name required"}), 400
        pid = body.get("id") or f"P{int(time.time())}{uuid.uuid4().hex[:4]}"
        profile = {
            "name": name,
            "age": body.get("age"),
            "gender": body.get("gender"),
            "height_cm": body.get("height_cm"),
            "weight_kg": body.get("weight_kg"),
            "bmi": body.get("bmi"),
            "createdAt": int(time.time()*1000)
        }
        db_root.child("patients").child(pid).child("profile").set(profile)
        return jsonify({"ok": True, "patient_id": pid, "profile": profile}), 201

@app.route("/start_session", methods=["POST"])
@require_api_key
def start_session():
    if not db_root:
        return jsonify({"error": "Firebase not initialized"}), 500
    data = request.get_json() or {}
    pid = data.get("patient_id")
    if not pid:
        return jsonify({"error":"patient_id required"}), 400

    duration = int(data.get("duration_sec", 40))
    simulate = bool(data.get("simulate", False))
    port = data.get("port", None)
    baud = int(data.get("baud", DEFAULT_BAUD))
    sample_interval_ms = int(data.get("sample_interval_ms", 1000))
    source = "simulate" if simulate or not port else "serial"

    session_id = uuid.uuid4().hex[:12]
    meta = {"patient_id": pid, "startedAt": int(time.time()*1000), "duration_sec": duration,
            "sample_interval_ms": sample_interval_ms, "source": source, "port": port, "baud": baud}
    db_root.child("patients").child(pid).child("sessions").child(session_id).child("meta").set(meta)
    db_root.child("patients").child(pid).child("sessions").child(session_id).child("status").set("running")

    stop_event = threading.Event()
    sess_obj = {"meta": meta, "stop_event": stop_event, "thread": None}
    with sessions_lock:
        sessions[session_id] = sess_obj

    t = threading.Thread(target=session_worker, args=(session_id,), daemon=True)
    sess_obj["thread"] = t
    t.start()

    return jsonify({"ok": True, "session_id": session_id, "meta": meta}), 201

@app.route("/stop_session", methods=["POST"])
@require_api_key
def stop_session():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error":"session_id required"}), 400
    with sessions_lock:
        sess = sessions.get(session_id)
        if not sess:
            return jsonify({"error":"session not found or already finished"}), 404
        sess["stop_event"].set()
    return jsonify({"ok": True, "message":"stop requested"}), 200

@app.route("/sessions/<patient_id>", methods=["GET"])
@require_api_key
def list_sessions(patient_id):
    if not db_root:
        return jsonify({"error": "Firebase not initialized"}), 500
    snap = db_root.child("patients").child(patient_id).child("sessions").get() or {}
    return jsonify({"sessions": snap}), 200

@app.route("/admin_details/latest", methods=["GET"])
@require_api_key
def get_admin_details():
    if not db_root:
        return jsonify({"error": "Firebase not initialized"}), 500
    try:
        meta = db_root.child("adminDetails").child("latest").get()
        if not meta:
            return jsonify({"ok": True, "meta": None, "message": "No admin details found"}), 200
        return jsonify({"ok": True, "meta": meta}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/upload_admin_pdf", methods=["POST"])
@require_api_key
def upload_admin_pdf():
    if not db_root:
        return jsonify({"error":"Firebase not initialized"}), 500
    if 'pdf' not in request.files:
        return jsonify({"error":"pdf file field required"}), 400
    pdf = request.files['pdf']
    if pdf.filename == '':
        return jsonify({"error":"no file chosen"}), 400
    filename = secure_filename(pdf.filename)
    timestamp = int(time.time()*1000)
    path = f"admin-details/{timestamp}-{filename}"
    if storage_bucket is None:
        return jsonify({"error":"Storage bucket not configured on Firebase app"}), 500
    try:
        blob = storage_bucket.blob(path)
        blob.upload_from_file(pdf.stream, content_type='application/pdf')
        # signed URL (7 days)
        try:
            download_url = blob.generate_signed_url(version="v4", expiration=3600*24*7)
        except TypeError:
            # For some firebase-admin versions generate_signed_url signature may differ
            download_url = blob.generate_signed_url(expiration=3600*24*7)
        meta = {"name": filename, "path": path, "url": download_url, "uploadedAt": timestamp}
        db_root.child("adminDetails").child("latest").set(meta)
        return jsonify({"ok": True, "meta": meta}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "firebase_initialized": bool(db_root)}), 200

# ---------- Run ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not db_root:
        print("Firebase not initialized yet. Call POST /init or set SERVICE_ACCOUNT_JSON and DATABASE_URL env vars.")
    print("Starting backend on http://%s:%d" % (args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug)
