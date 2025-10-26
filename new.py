# arduino_to_firebase_full.py
# Copy-paste and run in your virtualenv (terminal). Edit SERVICE_ACCOUNT_JSON and DATABASE_URL.

import os, time, threading, re, math
from collections import deque
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, db

# Serial
try:
    import serial
    import serial.tools.list_ports
except Exception as e:
    raise SystemExit("pyserial is required. Install with: pip install pyserial")

# ---------- CONFIG: EDIT THESE ----------
SERVICE_ACCOUNT_JSON = r"C:\Users\prave\Downloads\auscsync_package\auscsync-firebase-adminsdk-fbsvc-07a6684b0f.json"
DATABASE_URL = "https://auscsync-default-rtdb.firebaseio.com/"
PORT = "COM11"        # set your serial port
BAUD = 115200
RECORD_SECONDS = 40
# ---------------------------------------

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
    firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
db_root = db.reference("/")

# Helper: list patients
def list_patients():
    pts = db_root.child("patients").get() or {}
    return pts  # dict keyed by patient_id

def create_patient(profile):
    # profile: dict with name, age, sex, height, weight, bmi, medical_history(optional)
    patients_ref = db_root.child("patients")
    # create a new patient id (use provided id if given, else auto-generate 'P'+random)
    # We'll make an ID like P#### using timestamp
    pid = f"P{int(time.time())}"
    patients_ref.child(pid).child("profile").set(profile)
    print(f"Created patient {pid}")
    return pid

def set_patient_profile(pid, profile):
    db_root.child("patients").child(pid).child("profile").set(profile)

# Serial reader thread
stop_event = threading.Event()
samples_lock = threading.Lock()
samples = []  # list of dicts: {"t": epoch, "ecg":..., "ppg":..., "spo2":...}

def parse_line_to_values(line):
    # robust numeric parser; returns a tuple (ecg, ppg, spo2) or None
    toks = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
    if len(toks) < 3:
        return None
    try:
        ecg = float(toks[0])
        ppg = float(toks[1])
        spo2 = float(toks[2])
        return ecg, ppg, spo2
    except:
        return None

def serial_reader(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print("Failed to open serial port:", e)
        stop_event.set()
        return
    # allow Arduino reset
    time.sleep(1.5)
    ser.reset_input_buffer()
    print("Serial opened, reading...")
    start_ts = time.time()
    while (time.time() - start_ts) < RECORD_SECONDS and not stop_event.is_set():
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            continue
        if not line:
            continue
        vals = parse_line_to_values(line)
        if vals is None:
            # ignore non-numeric line
            continue
        ecg, ppg, spo2 = vals
        with samples_lock:
            samples.append({
                "t": time.time(),
                "ecg": float(ecg),
                "ppg": float(ppg),
                "spo2": float(spo2)
            })
    try:
        ser.close()
    except:
        pass
    print("Serial reader finished, samples captured:", len(samples))
    stop_event.set()

# Simple ECG peak detection (no scipy)
def estimate_hr_from_ecg(ecg_values, fs=40.0):
    # ecg_values: list or np.array of voltage-ish numbers sampled at ~fs Hz
    if len(ecg_values) < 10:
        return float('nan')
    arr = np.array(ecg_values)
    # remove baseline
    arr = arr - np.median(arr)
    thresh = np.mean(arr) + 0.5 * np.std(arr)
    peaks = []
    for i in range(1, len(arr)-1):
        if arr[i] > thresh and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks.append(i)
    if len(peaks) < 2:
        return float('nan')
    rr_samples = np.diff(peaks)
    rr_sec = rr_samples / fs
    mean_rr = np.mean(rr_sec)
    if mean_rr <= 0:
        return float('nan')
    hr = 60.0 / mean_rr
    return float(hr)

# Plotter (main thread; polls samples)
def live_plot(fs_est=40.0):
    plt.ion()
    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
    fig.suptitle("Live: ECG | PPG | SpO2")
    start_t = time.time()
    while not stop_event.is_set():
        with samples_lock:
            data = samples.copy()
        if data:
            times = np.array([d["t"] - start_t for d in data])
            ecg = np.array([d["ecg"] for d in data])
            ppg = np.array([d["ppg"] for d in data])
            spo2 = np.array([d["spo2"] for d in data])
            axs[0].cla(); axs[0].plot(times, ecg); axs[0].set_ylabel("ECG")
            axs[1].cla(); axs[1].plot(times, ppg); axs[1].set_ylabel("PPG")
            axs[2].cla(); axs[2].plot(times, spo2); axs[2].set_ylabel("SpO2")
            axs[-1].set_xlabel("Time (s)")
            plt.pause(0.1)
        else:
            plt.pause(0.1)
    plt.ioff()
    plt.show(block=False)

# Upload recording summary and raw samples to Firebase RTDB
def upload_recording(patient_id, patient_profile):
    with samples_lock:
        data_local = samples.copy()
    if not data_local:
        print("No samples to upload.")
        return None
    # compute aggregates
    ecg_vals = [d["ecg"] for d in data_local]
    ppg_vals = [d["ppg"] for d in data_local]
    spo2_vals = [d["spo2"] for d in data_local]
    hr_est = estimate_hr_from_ecg(ecg_vals, fs=100.0)  # guess fs 100 if Arduino at 10-25 ms
    spo2_avg = float(np.round(np.mean(spo2_vals), 2))
    ecg_mean = float(np.round(np.mean(ecg_vals), 4))
    ppg_mean = float(np.round(np.mean(ppg_vals), 4))
    # condition logic
    condition = "normal"
    if spo2_avg < 90 or (not math.isnan(hr_est) and (hr_est > 120 or hr_est < 40)):
        condition = "critical"
    elif spo2_avg < 94 or (not math.isnan(hr_est) and (hr_est > 100 or hr_est < 50)):
        condition = "warning"

    # fusion score simple composite
    conf = 0.0
    if not math.isnan(hr_est):
        hr_norm = min(max((hr_est - 50)/70.0, 0.0), 1.0)
    else:
        hr_norm = 0.0
    spo2_norm = 1.0 - min(max((95.0 - spo2_avg)/20.0, 0.0), 1.0)
    fusion_score = float(np.round(0.6*hr_norm + 0.4*(1-spo2_norm), 3))
    if fusion_score > 0.7:
        fusion_tier = "High"
    elif fusion_score > 0.4:
        fusion_tier = "Moderate"
    else:
        fusion_tier = "Normal"

    # summary node
    ts_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = {
        "timestamp": ts_iso,
        "hr": float(np.round(hr_est,1)) if not math.isnan(hr_est) else None,
        "spo2": spo2_avg,
        "ecg_mean": ecg_mean,
        "ppg_mean": ppg_mean,
        "condition": condition,
        "patient_id": patient_id,
        "patient_name": patient_profile.get("name"),
        "age": patient_profile.get("age"),
        "bmi": patient_profile.get("bmi"),
        "fusion_tier": fusion_tier,
        "fusion_score": fusion_score,
        "data_source": "Arduino"
    }

    # push to patients/{patient_id}/vitals
    vitals_ref = db_root.child("patients").child(patient_id).child("vitals")
    new_key = vitals_ref.push(summary)
    node_key = new_key.key
    print("Pushed summary to /patients/{}/vitals/{}".format(patient_id, node_key))

    # push raw samples as child 'samples' (list of small dicts) -- compress timestamps to offsets
    samples_payload = []
    start_ts = data_local[0]["t"]
    for d in data_local:
        samples_payload.append({
            "t_ms": int((d["t"] - start_ts)*1000),
            "ecg": float(d["ecg"]),
            "ppg": float(d["ppg"]),
            "spo2": float(d["spo2"])
        })
    vitals_ref.child(node_key).child("samples").set(samples_payload)
    print("Pushed raw samples (count {}) under the same vitals node.".format(len(samples_payload)))

    # create an alert if critical/warning
    if condition in ("warning", "critical"):
        alerts_ref = db_root.child("patients").child(patient_id).child("alerts")
        alert_obj = {
            "id": f"alert-{int(time.time())}",
            "type": "vitals_alert",
            "message": f"{condition.upper()} detected: hr={summary['hr']}, spo2={spo2_avg}",
            "priority": "high" if condition=="critical" else "medium",
            "timestamp": ts_iso,
            "acknowledged": False
        }
        alerts_ref.push(alert_obj)
        print("Alert created:", alert_obj["message"])

    return node_key

# Main interactive flow
def main():
    print("\n=== Firebase patients in DB ===")
    pts = list_patients()
    if pts:
        print("Existing patients:")
        for pid, pdoc in pts.items():
            prof = pdoc.get("profile", {}) if pdoc else {}
            print(f"  {pid}: {prof.get('name','(no name)')}, age={prof.get('age','?')}")
    else:
        print("No patients found in the DB yet.")

    ans = input("\nAre you recording for an (E)xisting patient or (N)ew? [E/N]: ").strip().upper() or "E"
    if ans == "N":
        name = input("Patient name: ").strip()
        age = int(input("Age: ").strip())
        sex = input("Sex (M/F/Other): ").strip()
        height_cm = float(input("Height (cm): ").strip())
        weight_kg = float(input("Weight (kg): ").strip())
        bmi = round(weight_kg / ((height_cm/100.0)**2), 2) if height_cm>0 else None
        profile = {
            "name": name,
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "medical_history": ""
        }
        patient_id = create_patient(profile)
        patient_profile = profile
    else:
        # choose existing
        if not pts:
            print("No existing patients — switching to new patient flow.")
            return main()
        patient_id = input("Enter patient id from list above (e.g. PXXXXXXXX): ").strip()
        if patient_id not in pts:
            print("Patient id not found. Try again.")
            return main()
        patient_profile = pts[patient_id].get("profile", {})

    input(f"\nPress ENTER to start recording for {RECORD_SECONDS} seconds for patient {patient_id} ({patient_profile.get('name')})...")
    print("Starting...")

    # start reader thread
    stop_event.clear()
    reader = threading.Thread(target=serial_reader, args=(PORT, BAUD), daemon=True)
    reader.start()

    # run live plot in main thread
    try:
        live_plot(fs_est=100.0)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        stop_event.set()

    # wait for reader to finish
    reader.join(timeout=5.0)

    # save CSV locally
    with samples_lock:
        if samples:
            df = pd.DataFrame(samples)
        else:
            df = pd.DataFrame(columns=["t", "ecg", "ppg", "spo2"])
    csv_path = os.path.join(os.getcwd(), "sensor_data.csv")
    df.to_csv(csv_path, index=False)
    print("Saved local CSV:", csv_path)

    # upload to Firebase
    node_key = upload_recording(patient_id, patient_profile)
    print("Done. Recording stored under patient", patient_id)
    print("Firebase node key:", node_key)

if __name__ == "__main__":
    main()
