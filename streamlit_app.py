"""
Streamlit Web Interface for YOLOv11 Object Detection Predictions
Cloud deployment version with Databricks MLflow integration.
Supports image upload and real-time predictions for autonomous vehicle object detection.
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import io
import base64

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mlflow

# Page configuration
st.set_page_config(
    page_title="Autonomous Vehicle Object Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration - Use environment variables or Streamlit secrets
@st.cache_resource
def load_config():
    """Load configuration from environment variables or Streamlit secrets."""
    config = {
        "model_path": os.getenv("MODEL_PATH", "best68.pt"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "prediction_log_dir": os.getenv("PREDICTION_LOG_DIR", "./prediction_logs"),
        "experiment_name": "YOLOv11s_Autonomous_Driving_OD_Predictions"
    }
    
    # Try to get from Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'mlflow' in st.secrets:
            config["mlflow_tracking_uri"] = st.secrets.mlflow.tracking_uri
            if 'databricks_token' in st.secrets.mlflow:
                os.environ['DATABRICKS_TOKEN'] = st.secrets.mlflow.databricks_token
        if hasattr(st, 'secrets') and 'model' in st.secrets:
            config["model_path"] = st.secrets.model.path
    except:
        pass
    
    return config

# Initialize configuration
config = load_config()

# Create directories if they don't exist
os.makedirs(config["prediction_log_dir"], exist_ok=True)

# Setup MLflow
if config["mlflow_tracking_uri"]:
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    
    # For Databricks, set authentication if token is available
    if 'DATABRICKS_TOKEN' in os.environ:
        os.environ['DATABRICKS_HOST'] = config["mlflow_tracking_uri"].replace('https://', '').split('/')[0]
    
    try:
        experiment = mlflow.get_experiment_by_name(config["experiment_name"])
        if experiment is None:
            mlflow.create_experiment(config["experiment_name"])
        mlflow.set_experiment(config["experiment_name"])
    except Exception as e:
        st.warning(f"⚠️ MLflow setup warning: {e}")

# Load model with caching
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching."""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# Load model
model, model_error = load_model(config["model_path"])

if model_error:
    st.error(f"❌ Error loading model: {model_error}")
    st.stop()

# Helper functions
def log_prediction_to_mlflow(image_info: Dict, predictions: List[Dict], inference_time: float):
    """Log prediction metrics to MLflow for monitoring."""
    if not config["mlflow_tracking_uri"]:
        return
    
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
            mlflow.set_tag("platform", "streamlit_cloud")
    except Exception as e:
        st.warning(f"⚠️ Could not log to MLflow: {e}")

def save_prediction_log(image_info: Dict, predictions: List[Dict], inference_time: float):
    """Save prediction log to file for monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_info": image_info,
        "num_detections": len(predictions),
        "inference_time_ms": inference_time * 1000,
        "predictions": predictions
    }
    
    log_file = os.path.join(config["prediction_log_dir"], f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        st.warning(f"⚠️ Could not save log: {e}")

def process_prediction(image: Image.Image) -> Dict[str, Any]:
    """Process image and return predictions."""
    # Convert PIL to numpy array
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
    
    return {
        "predictions": predictions,
        "annotated_image": annotated_image_pil,
        "image_info": image_info,
        "inference_time": inference_time
    }

# Main UI
st.title("🚗 Autonomous Vehicle Object Detection")
st.markdown("Upload an image to detect objects using YOLOv11 model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"**Model:** {config['model_path']}")
    
    if config["mlflow_tracking_uri"]:
        st.success("✅ MLflow Connected")
        st.caption(f"URI: {config['mlflow_tracking_uri'][:50]}...")
    else:
        st.warning("⚠️ MLflow not configured")
    
    st.divider()
    
    st.header("📊 Statistics")
    try:
        log_files = [f for f in os.listdir(config["prediction_log_dir"]) if f.endswith('.jsonl')]
        total_predictions = 0
        total_detections = 0
        
        for log_file in log_files:
            log_path = os.path.join(config["prediction_log_dir"], log_file)
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            total_predictions += 1
                            total_detections += entry.get('num_detections', 0)
            except:
                pass
        
        st.metric("Total Predictions", total_predictions)
        st.metric("Total Detections", total_detections)
        if total_predictions > 0:
            st.metric("Avg Detections/Image", f"{total_detections / total_predictions:.2f}")
    except:
        st.caption("No statistics available")

# Main content
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    help="Upload an image to detect objects"
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        st.image(image, use_container_width=True)
    
    # Process button
    if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
        with st.spinner("Processing image..."):
            result = process_prediction(image)
            
            # Log prediction
            log_prediction_to_mlflow(
                result["image_info"],
                result["predictions"],
                result["inference_time"]
            )
            save_prediction_log(
                result["image_info"],
                result["predictions"],
                result["inference_time"]
            )
            
            with col2:
                st.subheader("🎯 Detection Results")
                st.image(result["annotated_image"], use_container_width=True)
            
            # Display metrics
            st.divider()
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Detections", len(result["predictions"]))
            with col4:
                st.metric("Inference Time", f"{result['inference_time'] * 1000:.2f} ms")
            with col5:
                st.metric("Image Size", f"{result['image_info']['width']}x{result['image_info']['height']}")
            
            # Display predictions table
            if result["predictions"]:
                st.subheader("📋 Detected Objects")
                predictions_data = []
                for pred in result["predictions"]:
                    predictions_data.append({
                        "Class": pred["class"],
                        "Confidence": f"{pred['confidence']*100:.2f}%",
                        "BBox": f"({pred['bbox']['x1']:.0f}, {pred['bbox']['y1']:.0f}, {pred['bbox']['x2']:.0f}, {pred['bbox']['y2']:.0f})"
                    })
                
                st.dataframe(predictions_data, use_container_width=True, hide_index=True)
                
                # Class distribution chart
                st.subheader("📊 Class Distribution")
                class_counts = {}
                for pred in result["predictions"]:
                    cls = pred["class"]
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                st.bar_chart(class_counts)
            else:
                st.info("No objects detected in this image.")
            
            st.success("✅ Prediction completed and logged to MLflow!")

# Footer
st.divider()
st.caption("Powered by YOLOv11 | MLflow Tracking | Streamlit Cloud")

