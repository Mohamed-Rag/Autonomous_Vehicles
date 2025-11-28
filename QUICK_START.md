# Quick Start Guide - MLOps Setup

## What's Missing and What Was Added

### ✅ What Was Fixed/Added:

1. **Enhanced MLflow Tracking in `train.py`**:
   - ✅ Fixed duplicate code blocks
   - ✅ Added comprehensive metrics logging (all YOLO metrics)
   - ✅ Added proper MLflow tracking URI configuration
   - ✅ Added tags and metadata for better organization
   - ✅ Added artifact logging (plots, configs, models)
   - ✅ Added training duration tracking

2. **Web Interface (`web_app.py`)**:
   - ✅ FastAPI-based web application
   - ✅ Image upload and prediction endpoint
   - ✅ Real-time object detection
   - ✅ Prediction logging to MLflow
   - ✅ Statistics endpoint

3. **Model Monitoring (`monitor.py`)**:
   - ✅ Performance monitoring
   - ✅ Drift detection
   - ✅ Automated alerting
   - ✅ MLflow integration for monitoring metrics

4. **Retraining Strategy (`retrain_strategy.py`)**:
   - ✅ Automated retraining triggers
   - ✅ Performance-based retraining decisions
   - ✅ Retraining execution

5. **Documentation**:
   - ✅ Comprehensive MLOps documentation
   - ✅ Quick start guide

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow UI

**Option A: Using the batch file (Windows)**
```bash
start_mlflow.bat
```

**Option B: Using command line**
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access MLflow at: `http://localhost:5000`

### 3. Run Training with MLflow Tracking

```bash
python train.py
```

This will:
- ✅ Track all parameters and metrics in MLflow
- ✅ Log model artifacts
- ✅ Log training plots and visualizations
- ✅ Create a new experiment run

**View results**: Open `http://localhost:5000` and navigate to the experiment `YOLOv11s_Autonomous_Driving_OD`

### 4. Start Web Interface

**Option A: Using the batch file (Windows)**
```bash
start_web_app.bat
```

**Option B: Using command line**
```bash
python web_app.py
```

Or with uvicorn:
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
```

Access web interface at: `http://localhost:8000`

### 5. Run Monitoring

```bash
python monitor.py
```

This will:
- ✅ Analyze prediction logs
- ✅ Check for model drift
- ✅ Generate monitoring report
- ✅ Send alerts if issues detected

### 6. Check Retraining Strategy

```bash
python retrain_strategy.py
```

This will:
- ✅ Check if retraining is needed
- ✅ Generate retraining plan
- ✅ Execute retraining if urgent

---

## What to Check in MLflow UI

After running `train.py`, you should see:

### 1. **Parameters Tab**
- All training hyperparameters
- Hardware configuration
- Data paths

### 2. **Metrics Tab**
- `final_mAP@0.5`: Model accuracy
- `metrics/precision`: Precision score
- `metrics/recall`: Recall score
- `training_duration_seconds`: Training time
- All loss metrics

### 3. **Artifacts Tab**
- `model_weights/best.pt`: Best model
- `training_plots/`: All training visualizations
- `config/args.yaml`: Training configuration
- `metrics/results.csv`: Training results

### 4. **Tags**
- `model_type`: YOLOv11s
- `task`: object_detection
- `domain`: autonomous_driving

---

## Key Features

### ✅ Comprehensive Metrics Tracking

The enhanced `train.py` now logs:
- All YOLO metrics (mAP, precision, recall)
- All loss metrics (box, cls, dfl)
- Training duration
- Hardware information

### ✅ Prediction Tracking

The web interface logs:
- Number of detections per prediction
- Inference time
- Class distribution
- All predictions to MLflow

### ✅ Monitoring & Alerts

The monitoring system:
- Tracks performance over time
- Detects model drift
- Sends alerts for issues
- Generates monitoring reports

### ✅ Automated Retraining

The retraining strategy:
- Checks retraining conditions
- Triggers retraining when needed
- Manages model updates

---

## Troubleshooting

### MLflow Not Showing Data

1. **Check MLflow is running**:
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **Verify tracking URI**:
   The code uses `http://localhost:5000` by default. If your MLflow is on a different port, set:
   ```bash
   set MLFLOW_TRACKING_URI=http://localhost:5000
   ```

3. **Check experiment name**:
   The experiment should be: `YOLOv11s_Autonomous_Driving_OD`

### Web Interface Not Working

1. **Check model path**:
   Verify `MODEL_PATH` in `web_app.py` points to your model file

2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

3. **Check port availability**:
   Make sure port 8000 is not in use

### Monitoring Not Working

1. **Create prediction logs directory**:
   The directory `prediction_logs` is created automatically, but verify it exists

2. **Check log file format**:
   Logs should be in JSONL format (one JSON object per line)

---

## Next Steps

1. ✅ Run training and verify metrics appear in MLflow
2. ✅ Test web interface with sample images
3. ✅ Run monitoring to check system health
4. ✅ Review retraining strategy
5. ✅ Read full documentation in `MLOPS_DOCUMENTATION.md`

---

## Summary

**Before**: Basic MLflow setup with limited metrics logging
**After**: Complete MLOps pipeline with:
- ✅ Comprehensive experiment tracking
- ✅ Web interface for predictions
- ✅ Model monitoring and drift detection
- ✅ Automated retraining strategy
- ✅ Full documentation

All components are now integrated and ready to use! 🚀

