# Render Deployment - Quick Start

## 🚀 Quick Deployment Steps

### 1. Verify Files in Repository
Ensure these files are in your GitHub repo:
- ✅ `web_app.py` (cloud-ready)
- ✅ `yolov11s_final.pt` (model file)
- ✅ `render.yaml`
- ✅ `Procfile`
- ✅ `requirements_render.txt`

### 2. Deploy to Render

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect GitHub and select: `Mohamed-Rag/Autonomous_Vehicles`
4. Configure:
   - **Name:** `autonomous-car-detection`
   - **Build Command:** `pip install -r requirements_render.txt`
   - **Start Command:** `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

### 3. Add Environment Variables

In Render dashboard → Environment Variables, add:

```
MODEL_PATH=yolov11s_final.pt
MLFLOW_TRACKING_URI=https://dbc-2d106fc4-eb47.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-token-here
PREDICTION_LOG_DIR=./prediction_logs
```

⚠️ Mark `DATABRICKS_TOKEN` as **Secret**!

### 4. Deploy

Click **"Create Web Service"** and wait 5-10 minutes.

### 5. Test

- Health: `https://your-app.onrender.com/health`
- Web UI: `https://your-app.onrender.com/`
- Stats: `https://your-app.onrender.com/stats`

## 📊 Verify MLflow

1. Go to Databricks: `https://dbc-2d106fc4-eb47.cloud.databricks.com`
2. Navigate to Experiments → `YOLOv11s_Autonomous_Driving_OD_Predictions`
3. Check for new runs after making predictions

## 🔧 Troubleshooting

**Service won't start?**
- Check build logs in Render dashboard
- Verify model file exists in repo

**MLflow not working?**
- Verify token is correct
- Check environment variables are set

**Slow first request?**
- Normal for free tier (cold start ~30s)
- Service sleeps after 15 min inactivity

## 📚 Full Guide

See `RENDER_DEPLOYMENT.md` for complete instructions.

---

**Your app URL:** `https://autonomous-car-detection.onrender.com`

