import pynvml
import requests
import csv
import time
from datetime import datetime
from requests.auth import HTTPBasicAuth
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth


current_path = Path(__file__).resolve()
root_path = current_path.parent.parent
dotenv_path = root_path / '.env'
load_dotenv(dotenv_path=dotenv_path)


LHM_URL = os.getenv('MY_URL')
USER = os.getenv('MY_USER')
PASS = os.getenv('MY_PASS')
print(f"DEBUG: URL is {LHM_URL}") 
response = requests.get(LHM_URL, auth=HTTPBasicAuth(USER, PASS), timeout=1)

CSV_FILE = "data_logs.csv"
CURRENT_ACTIVITY = "high_load"
SURFACE_TYPE = "rough"
CPU_BOOST_TYPE = "agressive"
IS_CLOGGED = 1
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
START_TIME_EPOCH = time.time()

prev_gpu_temp = None
prev_cpu_temp = None

def clean_value(val_str):
    if val_str is None:
        return 0.0
    for char in [" W", " %", " MHz", " GB", " °C", " C"]:
        val_str = val_str.replace(char, "")
    try:
        return float(val_str)
    except ValueError:
        return 0.0
    
def get_data(node, metrics):
    text = node.get('Text')
    value = node.get('Value')

    if text == "Core (Tctl/Tdie)":
        metrics["cpu_temp_C"] = clean_value(value)

    if text == "Package" and node.get('Type') == "Power":
        metrics["cpu_power_W"] = clean_value(value)

    if text == "CPU Total":
        metrics["cpu_util_pct"] = clean_value(value)

    if text == "Cores (Average)":
        metrics["cpu_freq_MHz"] = clean_value(value)
        
    if text == "Memory Used" and "Kingston" not in node.get('Text'):
        metrics["ram_used_GB"] = clean_value(value)

    for child in node.get('Children', []):
        get_data(child, metrics)



        
def get_gpu_data(handle):
    return{
        "gpu_temp_C": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        "gpu_util_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
        "gpu_clock_MHz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
        "gpu_power_W": pynvml.nvmlDeviceGetPowerUsage(handle)/1000.0
    
    }

def gpu_data_safe(handle):
    if not handle:
        return{
            "gpu_temp_C": None,
            "gpu_util_pct": None,
            "gpu_clock_MHz": None,
            "gpu_power_W": None
        }
    try: 
        return get_gpu_data(handle)
    except pynvml.NVMLError:
        return{
            "gpu_temp_C": None,
            "gpu_util_pct": None,
            "gpu_clock_MHz": None,
            "gpu_power_W": None
        }
gpu_available = False
gpu_handle = None


try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_available = True
except pynvml.NVMLError:
    gpu_available = False
file_exists = os.path.isfile(CSV_FILE)
print(f"Logging to {CSV_FILE}")

with open(CSV_FILE, 'a', newline='') as f:
    features = ["timestamp", "session_id", "sec_since_start", "is_clogged", "activity", 
                "surface_type", "cpu_boost_mode", "cpu_temp_C", "cpu_temp_slope", 
                "cpu_power_W", "cpu_util_pct", "cpu_freq_MHz", "ram_used_GB", 
                "gpu_temp_C", "gpu_temp_slope", "gpu_util_pct", "gpu_clock_MHz", "gpu_power_W"]
    
    writer = csv.DictWriter(f, fieldnames = features)
    if not file_exists:
        writer.writeheader()

    try:
        while True:
           start_time = time.time()
           lhm_data = {"cpu_temp_C": 0, "cpu_power_W": 0, "cpu_util_pct": 0, "cpu_freq_MHz": 0, "ram_used_GB": 0,}
           response = requests.get(LHM_URL, auth=HTTPBasicAuth(USER, PASS), timeout=1)
           print(f"Status: {response.status_code}")
           print(f"Raw Text: '{response.text}'")
           get_data(response.json(), lhm_data)

           gpu_data = gpu_data_safe(gpu_handle)

           curr_cpu_temp = lhm_data["cpu_temp_C"]
           curr_gpu_temp = gpu_data["gpu_temp_C"]

           cpu_slope = (curr_cpu_temp - prev_cpu_temp) if prev_cpu_temp is not None else 0
           if curr_gpu_temp is not None and prev_gpu_temp is not None:
               gpu_slope = curr_gpu_temp - prev_gpu_temp
           else:
               gpu_slope = 0

        

           prev_cpu_temp, prev_gpu_temp = curr_cpu_temp, curr_gpu_temp
         
           row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "session_id": SESSION_ID, "sec_since_start": round(time.time() - START_TIME_EPOCH, 2), 
                  "is_clogged": IS_CLOGGED, "activity": CURRENT_ACTIVITY, 
                  "surface_type": SURFACE_TYPE, "cpu_boost_mode": CPU_BOOST_TYPE, 
                  "cpu_temp_slope": cpu_slope, "gpu_temp_slope": gpu_slope, **lhm_data, **gpu_data}
           
           writer.writerow(row)
           f.flush()
           elapsed = time.time() - start_time
           sleep_time = max(0, 1 - elapsed)
           time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n Logging stopped")
    finally:
        pynvml.nvmlShutdown()


    
    

