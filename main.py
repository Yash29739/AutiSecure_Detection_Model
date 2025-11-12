import os
from fastapi import FastAPI, BackgroundTasks
import tempfile
import requests
from pydantic import BaseModel

# --- 1. Import your project's components ---
from utils.video_analyzer import analyze_video_for_traits
from utils.model_predictor import load_model, generate_prediction_and_report, train_and_save_model

train_and_save_model('./dataset/trained_labels.csv', './dataset/tested_labels.csv');


# --- 3. Initialize the FastAPI application ---
app = FastAPI(
    title="Autism Trait Detection API",
    description="An API to analyze videos for behavioral traits and predict autism likelihood.",
    version="1.0.0"
)

# --- 4. Load the ML model once when the server starts ---
# This is highly efficient as the model stays loaded in memory.
print("--- Initializing Prediction Model (this may take a moment)... ---")
PREDICTOR_MODEL = load_model()
print("--- Model Initialized. API is ready to accept requests. ---")



class VideoRequest(BaseModel):
    videoUrl: str
    userId: str  # add userId so Python can return it to Node

app = FastAPI()
MODEL = load_model()

NODE_CALLBACK_URL = "https://autisense-backend.onrender.com/api/video"

@app.post("/analyze")
async def analyze_video_endpoint(request: VideoRequest, background_tasks: BackgroundTasks):
    # Respond instantly
    background_tasks.add_task(run_analysis_and_callback, request.videoUrl, request.userId)
    return {"message": "Job accepted, running in background"}

def run_analysis_and_callback(videoUrl: str, userId: str):
    try:
        # --- Step 1: Download video ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            r = requests.get(videoUrl, stream=True, timeout=30)
            for chunk in r.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        # --- Step 2: Run analysis ---
        video_features = analyze_video_for_traits(tmp_path)
        report = generate_prediction_and_report(video_features, MODEL)

        # --- Step 3: Send result back to Node ---
        requests.post(
            NODE_CALLBACK_URL,
            json={
                "userId": userId,
                "videoUrl": videoUrl,
                "report": report
            },
            timeout=30
        )

        print('sucessfuly sent the response')

    except Exception as e:
        print(f"Background analysis failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/ping", tags=["Health"])
def ping():
    return {"status": "alive"}

if __name__ == "__main__":
    # --- Local Test Mode ---
    print("\n--- Running local test ---")

    # ✅ Replace this with your local video file path
    test_video_path = "./1.mp4"
    test_user_id = "TEMP_USER_001"

    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found at {test_video_path}")
    else:
        print(f"✅ Found test video: {test_video_path}")
        try:
            # Simulate analysis directly without FastAPI
            video_features = analyze_video_for_traits(test_video_path)
            report = generate_prediction_and_report(video_features, MODEL)
            print("\n--- Prediction Report ---")
            print(report)
        except Exception as e:
            print(f"❌ Test run failed: {e}")
