"""
Streamlit Monitoring Dashboard for Model Performance
Cloud deployment version with Databricks MLflow integration.
"""

import streamlit as st
import os
import json
import mlflow
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
@st.cache_resource
def load_config():
    """Load configuration from environment variables or Streamlit secrets."""
    config = {
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "prediction_log_dir": os.getenv("PREDICTION_LOG_DIR", "./prediction_logs"),
        "experiment_name": "YOLOv11s_Autonomous_Driving_OD_Predictions",
        "training_experiment_name": "YOLOv11s_Autonomous_Driving_OD"
    }
    
    # Try to get from Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'mlflow' in st.secrets:
            config["mlflow_tracking_uri"] = st.secrets.mlflow.tracking_uri
            if 'databricks_token' in st.secrets.mlflow:
                os.environ['DATABRICKS_TOKEN'] = st.secrets.mlflow.databricks_token
    except:
        pass
    
    return config

config = load_config()

# Alert thresholds
ALERT_THRESHOLDS = {
    "accuracy_drop": 0.10,
    "inference_time_ms": 1000,
    "error_rate": 0.05,
    "detection_drop": 0.20
}

# Setup MLflow
mlflow_client = None
if config["mlflow_tracking_uri"]:
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    
    # For Databricks, set authentication if token is available
    if 'DATABRICKS_TOKEN' in os.environ:
        os.environ['DATABRICKS_HOST'] = config["mlflow_tracking_uri"].replace('https://', '').split('/')[0]
    
    try:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow_tracking_uri"])
    except Exception as e:
        st.warning(f"⚠️ MLflow client setup warning: {e}")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_baseline_metrics() -> Dict[str, float]:
    """Load baseline metrics from training runs."""
    if not mlflow_client:
        return {"mAP50": 0.5, "precision": 0.7, "recall": 0.6}
    
    try:
        mlflow.set_experiment(config["training_experiment_name"])
        experiment = mlflow.get_experiment_by_name(config["training_experiment_name"])
        if experiment is None:
            return {"mAP50": 0.5, "precision": 0.7, "recall": 0.6}
        
        runs = mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100
        )
        
        runs_with_metric = []
        for run in runs:
            metric_value = run.data.metrics.get("final_mAP@0.5")
            if metric_value is not None:
                runs_with_metric.append((metric_value, run))
        
        if runs_with_metric:
            runs_with_metric.sort(key=lambda x: x[0], reverse=True)
            run = runs_with_metric[0][1]
            return {
                "mAP50": run.data.metrics.get("final_mAP@0.5", 0.0),
                "precision": run.data.metrics.get("metrics/precision", 0.0),
                "recall": run.data.metrics.get("metrics/recall", 0.0)
            }
    except Exception as e:
        st.warning(f"⚠️ Could not load baseline metrics: {e}")
    
    return {"mAP50": 0.5, "precision": 0.7, "recall": 0.6}

@st.cache_data(ttl=60)  # Cache for 1 minute
def analyze_prediction_logs(days: int = 7) -> Dict[str, Any]:
    """Analyze prediction logs for the last N days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    stats = {
        "total_predictions": 0,
        "total_detections": 0,
        "avg_inference_time_ms": [],
        "class_distribution": defaultdict(int),
        "errors": 0,
        "daily_stats": defaultdict(lambda: {"predictions": 0, "detections": 0, "inference_times": []})
    }
    
    try:
        log_files = [f for f in os.listdir(config["prediction_log_dir"]) if f.endswith('.jsonl')]
        
        for log_file in log_files:
            log_path = os.path.join(config["prediction_log_dir"], log_file)
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            entry = json.loads(line)
                            timestamp = datetime.fromisoformat(entry.get("timestamp", ""))
                            
                            if start_date <= timestamp <= end_date:
                                stats["total_predictions"] += 1
                                stats["total_detections"] += entry.get("num_detections", 0)
                                inference_time = entry.get("inference_time_ms", 0)
                                stats["avg_inference_time_ms"].append(inference_time)
                                
                                date_key = timestamp.strftime("%Y-%m-%d")
                                stats["daily_stats"][date_key]["predictions"] += 1
                                stats["daily_stats"][date_key]["detections"] += entry.get("num_detections", 0)
                                stats["daily_stats"][date_key]["inference_times"].append(inference_time)
                                
                                for pred in entry.get("predictions", []):
                                    cls = pred.get("class", "unknown")
                                    stats["class_distribution"][cls] += 1
                        except json.JSONDecodeError:
                            stats["errors"] += 1
                        except Exception:
                            stats["errors"] += 1
            except Exception as e:
                st.warning(f"⚠️ Error reading log file {log_file}: {e}")
    except FileNotFoundError:
        st.warning("⚠️ Prediction logs directory not found")
    
    # Calculate averages
    if stats["avg_inference_time_ms"]:
        stats["avg_inference_time_ms"] = np.mean(stats["avg_inference_time_ms"])
    else:
        stats["avg_inference_time_ms"] = 0
    
    stats["avg_detections_per_image"] = (
        stats["total_detections"] / stats["total_predictions"]
        if stats["total_predictions"] > 0 else 0
    )
    
    # Calculate daily averages
    for date_key in stats["daily_stats"]:
        daily = stats["daily_stats"][date_key]
        if daily["inference_times"]:
            daily["avg_inference_time"] = np.mean(daily["inference_times"])
        else:
            daily["avg_inference_time"] = 0
    
    return stats

def check_model_drift(current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for model drift and performance degradation."""
    alerts = []
    
    if current_stats["avg_inference_time_ms"] > ALERT_THRESHOLDS["inference_time_ms"]:
        alerts.append({
            "type": "performance_degradation",
            "severity": "high",
            "message": f"Average inference time ({current_stats['avg_inference_time_ms']:.2f} ms) exceeds threshold",
            "metric": "inference_time_ms"
        })
    
    if current_stats["total_predictions"] > 0:
        error_rate = current_stats["errors"] / current_stats["total_predictions"]
        if error_rate > ALERT_THRESHOLDS["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "high",
                "message": f"Error rate ({error_rate*100:.2f}%) exceeds threshold",
                "metric": "error_rate"
            })
    
    if current_stats["avg_detections_per_image"] < 0.5:
        alerts.append({
            "type": "low_detection_rate",
            "severity": "medium",
            "message": f"Average detections per image ({current_stats['avg_detections_per_image']:.2f}) is unusually low",
            "metric": "avg_detections_per_image"
        })
    
    total_detections = sum(current_stats["class_distribution"].values())
    if total_detections > 0:
        for cls, count in current_stats["class_distribution"].items():
            percentage = count / total_detections
            if percentage > 0.8:
                alerts.append({
                    "type": "class_distribution_shift",
                    "severity": "medium",
                    "message": f"Class '{cls}' dominates detections ({percentage*100:.1f}%)",
                    "metric": "class_distribution"
                })
    
    return alerts

# Main UI
st.title("📊 Model Monitoring Dashboard")
st.markdown("Monitor model performance and detect drift")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    days = st.slider("Monitoring Period (days)", 1, 30, 7)
    
    if config["mlflow_tracking_uri"]:
        st.success("✅ MLflow Connected")
    else:
        st.warning("⚠️ MLflow not configured")
    
    st.divider()
    st.header("📈 Alert Thresholds")
    st.caption(f"Inference Time: {ALERT_THRESHOLDS['inference_time_ms']} ms")
    st.caption(f"Error Rate: {ALERT_THRESHOLDS['error_rate']*100}%")
    st.caption(f"Detection Drop: {ALERT_THRESHOLDS['detection_drop']*100}%")

# Load data
baseline_metrics = load_baseline_metrics()
stats = analyze_prediction_logs(days=days)
alerts = check_model_drift(stats)

# Status indicator
status = "healthy" if len(alerts) == 0 else "degraded"
status_color = "🟢" if status == "healthy" else "🔴"

st.header(f"{status_color} System Status: {status.upper()}")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", stats["total_predictions"])
col2.metric("Total Detections", stats["total_detections"])
col3.metric("Avg Detections/Image", f"{stats['avg_detections_per_image']:.2f}")
col4.metric("Avg Inference Time", f"{stats['avg_inference_time_ms']:.2f} ms")

# Alerts
if alerts:
    st.divider()
    st.header("🚨 Alerts")
    for alert in alerts:
        severity_color = "🔴" if alert["severity"] == "high" else "🟡"
        st.warning(f"{severity_color} **{alert['type']}**: {alert['message']}")

# Charts
st.divider()
st.header("📈 Performance Charts")

# Daily statistics
if stats["daily_stats"]:
    daily_df = pd.DataFrame([
        {
            "Date": date,
            "Predictions": data["predictions"],
            "Detections": data["detections"],
            "Avg Inference Time (ms)": data["avg_inference_time"]
        }
        for date, data in sorted(stats["daily_stats"].items())
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Predictions")
        fig = px.line(daily_df, x="Date", y="Predictions", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Daily Detections")
        fig = px.line(daily_df, x="Date", y="Detections", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Daily Inference Time")
    fig = px.line(daily_df, x="Date", y="Avg Inference Time (ms)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# Class distribution
if stats["class_distribution"]:
    st.subheader("Class Distribution")
    class_df = pd.DataFrame([
        {"Class": cls, "Count": count}
        for cls, count in sorted(stats["class_distribution"].items(), key=lambda x: x[1], reverse=True)
    ])
    
    fig = px.bar(class_df, x="Class", y="Count", color="Count", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# Baseline comparison
st.divider()
st.header("📊 Baseline Comparison")
col1, col2, col3 = st.columns(3)
col1.metric("Baseline mAP@0.5", f"{baseline_metrics['mAP50']:.3f}")
col2.metric("Baseline Precision", f"{baseline_metrics['precision']:.3f}")
col3.metric("Baseline Recall", f"{baseline_metrics['recall']:.3f}")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

