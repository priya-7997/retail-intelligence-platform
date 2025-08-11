import time
import requests

API_BASE = "http://localhost:8000/api/v1"
FILE_ID = "f6baf800-10f3-4f05-be32-0451d0138b2c"  # Replace with your actual file_id
MODEL_TYPE = "prophet"  # Or "auto" if needed
FORECAST_PERIODS = 30

# Poll for model training completion
def poll_model_training(file_id, model_type, interval=5, max_attempts=24):
    for attempt in range(max_attempts):
        print(f"Checking model status (attempt {attempt+1})...")
        resp = requests.get(f"{API_BASE}/models", params={"file_id": file_id})
        if resp.ok:
            data = resp.json()
            models = data.get("models", {})
            if model_type in models:
                print(f"Model '{model_type}' is available!")
                return True
            else:
                print(f"Model '{model_type}' not ready yet.")
        else:
            print("Failed to get model status.")
        time.sleep(interval)
    print("Model training did not complete in time.")
    return False

# Trigger prediction
def run_prediction(file_id, model_type, forecast_periods):
    payload = {
        "file_id": file_id,
        "model_type": model_type,
        "forecast_periods": forecast_periods
    }
    resp = requests.post(f"{API_BASE}/predict", json=payload)
    print("Prediction response:", resp.json())

if __name__ == "__main__":
    if poll_model_training(FILE_ID, MODEL_TYPE):
        run_prediction(FILE_ID, MODEL_TYPE, FORECAST_PERIODS)
    else:
        print("Model not ready. Check backend logs for errors.")
