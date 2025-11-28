# MLOps Pipeline Documentation
## Autonomous Vehicle Object Detection Project

This document details the MLOps practices used for tracking experiments, managing model versions, and streamlining deployment pipelines for the YOLOv11s autonomous vehicle object detection system.

---

## Table of Contents

1. [Overview](#overview)
2. [MLflow Setup](#mlflow-setup)
3. [Experiment Tracking](#experiment-tracking)
4. [Model Versioning](#model-versioning)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Retraining Strategy](#retraining-strategy)
7. [Web Interface](#web-interface)
8. [Deployment Pipeline](#deployment-pipeline)

---

## Overview

The MLOps pipeline for this project includes:

- **Experiment Tracking**: MLflow for tracking all training runs, metrics, and artifacts
- **Model Versioning**: Automatic versioning of trained models with metadata
- **Performance Monitoring**: Continuous monitoring of model performance in production
- **Automated Alerting**: Alerts for performance degradation and model drift
- **Retraining Strategy**: Automated retraining based on monitoring data
- **Web Interface**: FastAPI-based interface for real-time predictions

---

## MLflow Setup

### Installation

```bash
pip install mlflow
```

### Starting MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access the UI at: `http://localhost:5000`

### Configuration

The tracking URI is configured in `train.py`:

```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
```

To use a remote MLflow server, set the environment variable:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

---

## Experiment Tracking

### Training Experiments

All training runs are tracked in the experiment: `YOLOv11s_Autonomous_Driving_OD`

#### Tracked Parameters

- **Training Hyperparameters**:
  - `epochs`: Number of training epochs
  - `optimizer`: Optimizer type (AdamW)
  - `lr0`: Initial learning rate
  - `lrf`: Final learning rate factor
  - `batch_size`: Batch size
  - `img_size`: Image size
  - `freeze_layers`: Number of frozen layers
  - `dropout`: Dropout rate
  - Loss weights: `box_loss_weight`, `cls_loss_weight`, `dfl_loss_weight`

- **Hardware Configuration**:
  - `gpu_name`: GPU model
  - `vram_gb`: VRAM in GB
  - `ram_gb`: System RAM in GB
  - `device`: Training device (GPU/CPU)

#### Tracked Metrics

- **Performance Metrics**:
  - `final_mAP@0.5`: Mean Average Precision at IoU 0.5
  - `metrics/mAP50-95`: mAP across IoU 0.5-0.95
  - `metrics/precision`: Precision
  - `metrics/recall`: Recall

- **Loss Metrics**:
  - `metrics/box_loss`: Bounding box loss
  - `metrics/cls_loss`: Classification loss
  - `metrics/dfl_loss`: Distribution Focal Loss

- **Training Metrics**:
  - `training_duration_seconds`: Total training time

#### Tracked Artifacts

- **Model Weights**:
  - `best.pt`: Best model weights
  - `last.pt`: Last epoch weights

- **Training Visualizations**:
  - `training_plots/`: All training plots
  - `confusion_matrix.png`: Confusion matrix
  - `results.csv`: Training results CSV

- **Configuration**:
  - `args.yaml`: Training configuration

### Prediction Experiments

Production predictions are tracked in: `YOLOv11s_Autonomous_Driving_OD_Predictions`

#### Tracked Metrics

- `num_detections`: Number of objects detected
- `inference_time_ms`: Inference time in milliseconds
- `detections_{class}`: Count of detections per class

---

## Model Versioning

### Automatic Versioning

Each training run creates a new model version with:
- Unique run ID
- Timestamp
- All training parameters and metrics
- Model artifacts

### Model Registry (Optional)

To use MLflow Model Registry for production models:

```python
mlflow.register_model(
    f"runs:/{run.info.run_id}/models/best.pt",
    "YOLOv11s_Autonomous_Driving"
)
```

### Model Loading

Load a specific model version:

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/YOLOv11s_Autonomous_Driving/1")
```

---

## Monitoring & Alerting

### Monitoring System

The monitoring system (`monitor.py`) tracks:

1. **Performance Metrics**:
   - Average inference time
   - Detection rate
   - Error rate
   - Class distribution

2. **Drift Detection**:
   - Performance degradation
   - Class distribution shifts
   - Inference time increases

3. **Alert Thresholds**:
   - Accuracy drop: >10%
   - Inference time: >1000ms
   - Error rate: >5%
   - Detection drop: >20%

### Running Monitoring

```bash
python monitor.py
```

This generates a monitoring report and checks for alerts.

### Alert Types

1. **Performance Degradation**: High inference time or low performance
2. **High Error Rate**: Too many prediction errors
3. **Low Detection Rate**: Unusually low detections per image
4. **Class Distribution Shift**: Single class dominating detections

### Email Alerts (Optional)

Configure email alerts in `monitor.py`:

```python
EMAIL_CONFIG = {
    "enabled": True,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-app-password",
    "recipient_emails": ["team@example.com"]
}
```

---

## Retraining Strategy

### Automated Retraining

The retraining strategy (`retrain_strategy.py`) automatically determines when to retrain based on:

1. **Time-based**: Retrain every 30 days
2. **Performance-based**: Retrain when performance degrades
3. **Error-based**: Retrain when error rate exceeds threshold
4. **Drift-based**: Retrain when model drift is detected

### Retraining Triggers

- Performance drop: >10% decrease in mAP
- Error rate: >5%
- Days since training: >30 days
- Detection drop: >20% decrease

### Running Retraining Check

```bash
python retrain_strategy.py
```

### Manual Retraining

Execute training:

```bash
python train.py
```

---

## Web Interface

### FastAPI Application

The web interface (`web_app.py`) provides:

1. **Image Upload**: Upload images for object detection
2. **Real-time Predictions**: Get predictions with bounding boxes
3. **Statistics**: View prediction statistics
4. **Health Check**: Monitor API health

### Starting the Web Server

```bash
python web_app.py
```

Or with uvicorn:

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `GET /`: Web interface for image upload
- `POST /predict`: Upload image and get predictions
- `GET /health`: Health check endpoint
- `GET /stats`: Get prediction statistics

### Example API Usage

```python
import requests

# Upload image
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

result = response.json()
print(f"Detections: {result['num_detections']}")
print(f"Inference time: {result['inference_time_ms']} ms")
```

---

## Deployment Pipeline

### Development Workflow

1. **Data Collection**: Collect and label new data
2. **Training**: Run `train.py` to train model
3. **Tracking**: Metrics automatically logged to MLflow
4. **Evaluation**: Review metrics in MLflow UI
5. **Deployment**: Deploy best model to production

### Production Workflow

1. **Monitoring**: Run `monitor.py` regularly (e.g., daily)
2. **Alerting**: Receive alerts for performance issues
3. **Retraining**: Run `retrain_strategy.py` to check if retraining needed
4. **Validation**: Validate retrained model before deployment
5. **Deployment**: Deploy new model version

### CI/CD Integration (Optional)

Example GitHub Actions workflow:

```yaml
name: Model Training
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Train Model
        run: python train.py
      - name: Run Monitoring
        run: python monitor.py
      - name: Check Retraining
        run: python retrain_strategy.py
```

---

## Best Practices

### 1. Experiment Organization

- Use descriptive experiment names
- Tag runs with relevant metadata
- Keep experiments focused on specific goals

### 2. Model Versioning

- Always log model artifacts
- Use model registry for production models
- Document model changes in run notes

### 3. Monitoring

- Run monitoring regularly (daily recommended)
- Review alerts promptly
- Keep monitoring thresholds updated

### 4. Retraining

- Validate new data before retraining
- Compare retrained model with baseline
- Test retrained model before deployment

### 5. Documentation

- Document all experiments
- Keep track of hyperparameter changes
- Maintain changelog for model versions

---

## Troubleshooting

### MLflow Not Tracking

1. Check MLflow server is running: `mlflow ui`
2. Verify tracking URI: `mlflow.get_tracking_uri()`
3. Check experiment exists: `mlflow.get_experiment_by_name()`

### Monitoring Not Working

1. Check prediction logs directory exists
2. Verify log file format (JSONL)
3. Check file permissions

### Web Interface Errors

1. Verify model path is correct
2. Check model file exists
3. Verify dependencies installed

---

## Future Enhancements

- [ ] Model registry integration
- [ ] A/B testing framework
- [ ] Automated hyperparameter tuning
- [ ] Data versioning with DVC
- [ ] Kubernetes deployment
- [ ] Real-time streaming predictions
- [ ] Advanced drift detection algorithms

---

## Contact & Support

For questions or issues, please refer to the project repository or contact the development team.

