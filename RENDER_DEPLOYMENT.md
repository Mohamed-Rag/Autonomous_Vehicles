# Render Cloud Deployment Guide

Complete guide for deploying your Autonomous Vehicle Object Detection API to Render with Databricks MLflow integration.

## 📋 Prerequisites

✅ **Completed:**
- Databricks Community Edition account
- Databricks Workspace URL: `https://dbc-2d106fc4-eb47.cloud.databricks.com`
- Personal Access Token: `your-databricks-token-here` (set in Render dashboard)
- Model file: `yolov11s_final.pt` (18.7MB)
- GitHub repository: `https://github.com/Mohamed-Rag/Autonomous_Vehicles`
- Render account

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure all files are committed:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Verify these files are in your repo:**
   - ✅ `web_app.py` (updated for cloud)
   - ✅ `monitor.py` (updated for cloud)
   - ✅ `yolov11s_final.pt` (model file)
   - ✅ `render.yaml` (Render configuration)
   - ✅ `Procfile` (process file)
   - ✅ `requirements_render.txt` (dependencies)

### Step 2: Set Up Databricks MLflow

1. **Access your Databricks workspace:**
   - Go to: `https://dbc-2d106fc4-eb47.cloud.databricks.com`
   - Sign in with your account

2. **Create MLflow experiments** (if not already created):
   - Open a Databricks notebook
   - Run:
   ```python
   import mlflow
   mlflow.set_tracking_uri("https://dbc-2d106fc4-eb47.cloud.databricks.com")
   
   # Create prediction experiment
   try:
       mlflow.create_experiment("YOLOv11s_Autonomous_Driving_OD_Predictions")
   except:
       print("Experiment already exists")
   
   # Create training experiment (if needed)
   try:
       mlflow.create_experiment("YOLOv11s_Autonomous_Driving_OD")
   except:
       print("Experiment already exists")
   ```

3. **Verify your access token:**
   - Token: `your-databricks-token-here` (get from Databricks User Settings → Access Tokens)
   - Make sure it's still valid (tokens can expire)

### Step 3: Deploy to Render

1. **Log in to Render:**
   - Go to [render.com](https://render.com)
   - Sign in to your account

2. **Create New Web Service:**
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub account if not already connected
   - Select repository: `Mohamed-Rag/Autonomous_Vehicles`
   - Choose branch: `main` (or your deployment branch)

3. **Configure Service:**
   - **Name:** `autonomous-car-detection` (or your preferred name)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements_render.txt`
   - **Start Command:** `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Select **Free** tier

4. **Add Environment Variables:**
   Click **"Advanced"** → **"Environment Variables"** and add:
   
   ```
   MODEL_PATH=yolov11s_final.pt
   MLFLOW_TRACKING_URI=https://dbc-2d106fc4-eb47.cloud.databricks.com
   DATABRICKS_TOKEN=your-databricks-token-here
   PREDICTION_LOG_DIR=./prediction_logs
   PYTHON_VERSION=3.11.0
   ```

   ⚠️ **Important:** Mark `DATABRICKS_TOKEN` as **Secret** (click the lock icon)

5. **Deploy:**
   - Click **"Create Web Service"**
   - Render will start building and deploying your app
   - This may take 5-10 minutes for the first deployment

### Step 4: Monitor Deployment

1. **Watch the build logs:**
   - Render will show real-time build progress
   - Check for any errors in the logs

2. **Common issues and fixes:**
   - **Model not found:** Ensure `yolov11s_final.pt` is in the root directory
   - **MLflow connection error:** Verify token and workspace URL
   - **Memory issues:** Free tier has 512MB RAM - model should fit (18.7MB)

3. **Verify deployment:**
   - Once deployed, your app will be at: `https://autonomous-car-detection.onrender.com`
   - Test the health endpoint: `https://autonomous-car-detection.onrender.com/health`
   - Test prediction: `https://autonomous-car-detection.onrender.com/`

### Step 5: Verify MLflow Integration

1. **Test a prediction:**
   - Upload an image via the web interface
   - Check that it processes successfully

2. **Check MLflow:**
   - Go to your Databricks workspace
   - Navigate to **Experiments** → `YOLOv11s_Autonomous_Driving_OD_Predictions`
   - You should see new runs logged with each prediction

3. **Verify metrics:**
   - Check that metrics like `num_detections`, `inference_time_ms` are being logged
   - Verify tags like `platform: render_cloud` and `model_version: yolov11s_final`

## 🔧 Configuration Details

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `MODEL_PATH` | `yolov11s_final.pt` | Path to model file |
| `MLFLOW_TRACKING_URI` | `https://dbc-2d106fc4-eb47.cloud.databricks.com` | Databricks MLflow URI |
| `DATABRICKS_TOKEN` | `your-databricks-token-here` | Authentication token (get from Databricks) |
| `PREDICTION_LOG_DIR` | `./prediction_logs` | Local log directory |
| `PYTHON_VERSION` | `3.11.0` | Python version |

### Render Free Tier Limits

- **RAM:** 512MB (sufficient for your 18.7MB model)
- **CPU:** 0.5 CPU cores
- **Bandwidth:** 100GB/month
- **Sleep:** Service sleeps after 15 minutes of inactivity
- **Build time:** ~5-10 minutes

## 📊 Monitoring Your Deployment

### Health Check

```bash
curl https://autonomous-car-detection.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-29T..."
}
```

### Statistics Endpoint

```bash
curl https://autonomous-car-detection.onrender.com/stats
```

### View Logs

- In Render dashboard, go to your service
- Click **"Logs"** tab to see real-time logs
- Check for MLflow connection messages

## 🔍 Troubleshooting

### Issue: Service won't start

**Symptoms:** Service shows "Failed" status

**Solutions:**
1. Check build logs for errors
2. Verify all dependencies in `requirements_render.txt`
3. Ensure model file exists in repository
4. Check Python version compatibility

### Issue: MLflow connection failed

**Symptoms:** Warnings in logs about MLflow

**Solutions:**
1. Verify `MLFLOW_TRACKING_URI` is correct
2. Check `DATABRICKS_TOKEN` is valid and not expired
3. Ensure token has proper permissions
4. Test connection locally first

### Issue: Model not loading

**Symptoms:** `model is None` in health check

**Solutions:**
1. Verify `yolov11s_final.pt` is in root directory
2. Check file size matches (18.7MB)
3. Ensure file is committed to git
4. Check `MODEL_PATH` environment variable

### Issue: Slow cold starts

**Symptoms:** First request takes 30+ seconds

**Solutions:**
- This is normal for Render free tier
- Service sleeps after 15 min inactivity
- First request wakes it up (takes ~30s)
- Subsequent requests are fast
- Consider upgrading to paid tier for always-on

### Issue: Out of memory

**Symptoms:** Service crashes or errors

**Solutions:**
1. Free tier has 512MB RAM limit
2. Your model (18.7MB) should fit easily
3. If issues persist, optimize model or upgrade tier

## 🔄 Updating Your Deployment

1. **Make changes to code**
2. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Update deployment"
   git push origin main
   ```
3. **Render auto-deploys** on push to main branch
4. **Monitor deployment** in Render dashboard

## 📈 Next Steps

1. **Set up monitoring:**
   - Run `monitor.py` periodically (can deploy as separate service)
   - Set up alerts for model drift

2. **Optimize performance:**
   - Consider model quantization for faster inference
   - Implement caching for frequently accessed images

3. **Scale up (if needed):**
   - Upgrade to Render paid tier for always-on service
   - Add more instances for load balancing

4. **Add features:**
   - API rate limiting
   - Authentication
   - Batch prediction endpoint

## 🔐 Security Best Practices

1. **Never commit tokens:**
   - ✅ Token is set in Render dashboard (not in code)
   - ✅ `.gitignore` should exclude sensitive files

2. **Rotate tokens:**
   - Update Databricks token periodically
   - Update in Render dashboard when changed

3. **Monitor usage:**
   - Check Render usage dashboard
   - Monitor Databricks workspace usage

## 📚 Resources

- [Render Documentation](https://render.com/docs)
- [Databricks MLflow Guide](https://docs.databricks.com/mlflow/index.html)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Your GitHub Repo](https://github.com/Mohamed-Rag/Autonomous_Vehicles)

## ✅ Deployment Checklist

- [ ] All files committed to GitHub
- [ ] `yolov11s_final.pt` in repository
- [ ] Databricks experiments created
- [ ] Render service created
- [ ] Environment variables set in Render
- [ ] Service deployed successfully
- [ ] Health check passes
- [ ] Test prediction works
- [ ] MLflow logging verified
- [ ] Monitoring set up

---

**Your app will be live at:** `https://autonomous-car-detection.onrender.com`

**Need help?** Check Render logs or Databricks workspace for detailed error messages.

