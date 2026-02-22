import requests
import json
from typing import Dict

class FlaskAPIClient:
    """
    Client to send data from main script to Flask server:
    - Inferred images
    - Live camera frames
    - Python dictionaries (JSON)
    - Logs / text
    """
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip("/")

    def send_image(self, image_path: str) -> bool:
        """Send an inferred image to Flask."""
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}
                response = requests.post(f"{self.base_url}/api/upload_image", files=files)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Failed to send inferred image: {e}")
            return False

    def send_live_frame(self, frame_path: str) -> bool:
        """Send the latest live camera frame to Flask."""
        try:
            with open(frame_path, "rb") as f:
                files = {"frame": f}
                response = requests.post(f"{self.base_url}/api/upload_live_frame", files=files)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Failed to send live frame: {e}")
            return False

    def send_json(self, data: Dict) -> bool:
        """Send a Python dictionary as JSON to Flask."""
        try:
            response = requests.post(f"{self.base_url}/api/upload_json", json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Failed to send JSON data: {e}")
            return False

    def send_log(self, log_text: str) -> bool:
        """Send a one-line log / text message to Flask."""
        try:
            response = requests.post(f"{self.base_url}/api/upload_log", json={"log": log_text})
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Failed to send log: {e}")
            return False