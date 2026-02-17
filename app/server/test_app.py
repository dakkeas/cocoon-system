import sys
import os
import platform
import json
import time
import glob
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# --- SMART PATH SETUP ---
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

OUTPUT_DIR = os.path.join(project_root, "cocoon", "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- IMPORTS ---
try:
    # FIXED IMPORT
    import cocoon.config as config
    from cocoon.test_inference import TestVisionSystem 
except ImportError as e:
    print("-" * 40)
    print(f"❌ IMPORT ERROR: {e}")
    print(f"📍 Debug: Python is looking in: {project_root}")
    print("-" * 40)
    sys.exit(1)

app = Flask(__name__)
CORS(app)

system_state = {
    "sorting_active": False,
    "cocoon_grid": [0] * 144, 
    "stats": { "g_count": 0, "ng_count": 0, "empty_count": 0 },
    "last_scan_time": None 
}

try:
    print("🚀 Initializing TEST Vision System...")
    vision = TestVisionSystem(
        model_name=config.MODEL_NAME, 
        model_dir="models"
    )
    print("✅ System Ready (Test Mode).")
except Exception as e:
    print(f"❌ Init Failed: {e}")

@app.route('/api/data', methods=['GET'])
def get_data():
    stats = system_state["stats"]
    total = stats["g_count"] + stats["ng_count"] + stats["empty_count"]
    real = stats["g_count"] + stats["ng_count"]
    rate = (stats["ng_count"] / real * 100) if real > 0 else 0

    return jsonify({
        "grid": system_state["cocoon_grid"],
        "g_count": stats["g_count"],
        "ng_count": stats["ng_count"],
        "empty_count": stats["empty_count"],
        "total": total,
        "defect_rate": round(rate, 1),
        "active": system_state["sorting_active"],
        "last_scan_time": system_state["last_scan_time"]
    })

@app.route('/api/latest_image')
def get_latest_image():
    try:
        list_of_files = glob.glob(os.path.join(OUTPUT_DIR, '*.jpg'))
        if not list_of_files:
            return "No image found", 404
        latest_file = max(list_of_files, key=os.path.getctime)
        return send_file(latest_file, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

@app.route('/api/action', methods=['POST'])
def handle_action():
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        print("\n▶ [START] Scanning...")
        system_state["sorting_active"] = True
        try:
            grid_dict = vision.run_inference() 
            
            flat_grid = []
            g, ng, empty = 0, 0, 0
            for r in range(1, 13):
                row_data = grid_dict.get(r, ["Empty"]*12)
                for val in row_data:
                    if val == "G": flat_grid.append(1); g += 1
                    elif val == "NG": flat_grid.append(2); ng += 1
                    else: flat_grid.append(3); empty += 1

            system_state["cocoon_grid"] = flat_grid
            system_state["stats"] = {"g_count": g, "ng_count": ng, "empty_count": empty}
            system_state["last_scan_time"] = time.time()
            system_state["sorting_active"] = False
            print("✅ Scan Complete.")

        except Exception as e:
            print(f"❌ Error: {e}")
            system_state["sorting_active"] = False

    elif action == 'reset':
        print("\n↺ [RESET]")
        system_state["cocoon_grid"] = [0] * 144
        system_state["stats"] = { "g_count": 0, "ng_count": 0, "empty_count": 0 }
        system_state["last_scan_time"] = None

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)