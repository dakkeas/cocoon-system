import cv2
import json
import requests
import threading
import base64
import time


class StreamBridge:
    def __init__(self, api_url="http://127.0.0.1:5000/update", timeout=0.5):
        self.api_url = api_url
        self.timeout = timeout

    def _post_payload(self, payload):
        """Generic worker to send any dictionary payload."""
        try:
            requests.post(self.api_url, json=payload, timeout=self.timeout)
        except Exception as e:
            # Keep error terse to avoid spamming console
            print(f"[Bridge Error] {e}")

    def send_inference(self, data_dict):
        """Sends just the dictionary data."""
        payload = {"inference_data": data_dict}
        threading.Thread(target=self._post_payload, args=(payload,), daemon=True).start()

    def send_log(self, log_msg):
        """Sends just a text log."""
        payload = {"logs": log_msg}
        threading.Thread(target=self._post_payload, args=(payload,), daemon=True).start()

    def send_frame(self, frame, quality=60):
        """
        Encodes and sends the frame. 
        Crucial: Encoding happens INSIDE the thread to keep main loop fast.
        """
        def worker(img):
            try:
                # 1. Compress (Heavy CPU op)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success, buffer = cv2.imencode('.jpg', img, encode_param)
                
                if success:
                    # 2. Base64 Encode
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    
                    # 3. Send
                    self._post_payload({"image": jpg_as_text})
            except Exception as e:
                print(f"[Frame Error] {e}")

        # Start the worker
        threading.Thread(target=worker, args=(frame,), daemon=True).start()