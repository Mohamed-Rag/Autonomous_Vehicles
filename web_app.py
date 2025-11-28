"""
FastAPI Web Interface for YOLOv11 Object Detection Predictions
Supports image upload and real-time predictions for autonomous vehicle object detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64
from typing import List, Dict, Any
import json
from datetime import datetime
import mlflow
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous Vehicle Object Detection API",
    description="Real-time object detection for autonomous vehicles",
    version="1.0.0"
)

# Configuration
MODEL_PATH = r"H:\Startups\Autonomus Car Detection DEPI\best68.pt"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PREDICTION_LOG_DIR = r"H:\Startups\Autonomus Car Detection DEPI\prediction_logs"

# Create directories if they don't exist
os.makedirs(PREDICTION_LOG_DIR, exist_ok=True)

# Load model
print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Setup MLflow for prediction tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "YOLOv11s_Autonomous_Driving_OD_Predictions"

try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
except:
    pass

# Templates (if you want to add HTML frontend)
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    templates = None


def log_prediction_to_mlflow(image_info: Dict, predictions: List[Dict], inference_time: float):
    """Log prediction metrics to MLflow for monitoring."""
    try:
        with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True):
            mlflow.log_metric("num_detections", len(predictions))
            mlflow.log_metric("inference_time_ms", inference_time * 1000)
            mlflow.log_param("image_size", f"{image_info.get('width')}x{image_info.get('height')}")
            
            # Log class distribution
            class_counts = {}
            for pred in predictions:
                cls = pred.get('class', 'unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            for cls, count in class_counts.items():
                mlflow.log_metric(f"detections_{cls}", count)
            
            mlflow.set_tag("prediction_type", "real_time")
    except Exception as e:
        print(f"⚠️ Warning: Could not log to MLflow: {e}")


def save_prediction_log(image_info: Dict, predictions: List[Dict], inference_time: float):
    """Save prediction log to file for monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_info": image_info,
        "num_detections": len(predictions),
        "inference_time_ms": inference_time * 1000,
        "predictions": predictions
    }
    
    log_file = os.path.join(PREDICTION_LOG_DIR, f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Home page with upload interface."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Autonomous Vehicle Object Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            #result { margin-top: 20px; }
            #result img { max-width: 100%; border: 1px solid #ccc; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🚗 Autonomous Vehicle Object Detection</h1>
        <div class="upload-area">
            <h2>Upload Image for Detection</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageInput" accept="image/*" required>
                <br><br>
                <button type="submit" class="btn">Detect Objects</button>
            </form>
        </div>
        <div id="result"></div>
        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('imageInput').files[0]);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = '<p style="color: red;">Error: ' + data.error + '</p>';
                    } else {
                        resultDiv.innerHTML = `
                            <h3>Detection Results</h3>
                            <p><strong>Detections:</strong> ${data.num_detections}</p>
                            <p><strong>Inference Time:</strong> ${data.inference_time_ms.toFixed(2)} ms</p>
                            <img src="data:image/jpeg;base64,${data.annotated_image}" alt="Result">
                            <h4>Detected Objects:</h4>
                            <ul>
                                ${data.predictions.map(p => `<li>${p.class} (confidence: ${(p.confidence*100).toFixed(1)}%)</li>`).join('')}
                            </ul>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in uploaded image.
    Returns JSON with detections and annotated image.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert RGB to BGR if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Get image info
        image_info = {
            "width": image.width,
            "height": image.height,
            "format": image.format
        }
        
        # Run prediction
        import time
        start_time = time.time()
        
        results = model.predict(
            image_np,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        # Process results
        predictions = []
        if results and len(results) > 0:
            result = results[0]
            
            # Extract detections
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    predictions.append({
                        "class": model.names[cls] if hasattr(model, 'names') else f"class_{cls}",
                        "class_id": cls,
                        "confidence": conf,
                        "bbox": {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3])
                        }
                    })
        
        # Get annotated image
        annotated_image_np = result.plot() if results and len(results) > 0 else image_np
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB))
        
        # Convert to base64
        buffer = io.BytesIO()
        annotated_image_pil.save(buffer, format="JPEG")
        annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Log prediction
        log_prediction_to_mlflow(image_info, predictions, inference_time)
        save_prediction_log(image_info, predictions, inference_time)
        
        return JSONResponse(content={
            "num_detections": len(predictions),
            "predictions": predictions,
            "inference_time_ms": inference_time * 1000,
            "annotated_image": annotated_image_b64,
            "image_info": image_info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get prediction statistics."""
    try:
        # Count predictions from log files
        log_files = [f for f in os.listdir(PREDICTION_LOG_DIR) if f.endswith('.jsonl')]
        total_predictions = 0
        total_detections = 0
        
        for log_file in log_files:
            log_path = os.path.join(PREDICTION_LOG_DIR, log_file)
            with open(log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        total_predictions += 1
                        total_detections += entry.get('num_detections', 0)
        
        return {
            "total_predictions": total_predictions,
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / total_predictions if total_predictions > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

