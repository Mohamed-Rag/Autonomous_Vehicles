# Streamlit Cloud Deployment Guide

This guide will help you deploy your Autonomous Vehicle Object Detection app to Streamlit Cloud with Databricks MLflow integration.

## 📋 Prerequisites

1. **GitHub Account** - Streamlit Cloud deploys from GitHub repositories
2. **Streamlit Cloud Account** - Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **Databricks Community Edition** - Free MLflow server
4. **Model File** - Your trained YOLOv11 model (`best68.pt`)

## 🚀 Step 1: Set Up Databricks Community Edition

### 1.1 Create Databricks Account

1. Go to [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. Sign up for a free account
3. Create a new workspace

### 1.2 Get MLflow Tracking URI

1. In your Databricks workspace, go to **Compute** → **Create Cluster** (or use existing)
2. Open a **Notebook** and run:
   ```python
   import mlflow
   print(mlflow.get_tracking_uri())
   ```
3. Copy the tracking URI (format: `https://<workspace-id>.cloud.databricks.com`)

### 1.3 Create Personal Access Token

1. Click on your **profile icon** (top right)
2. Go to **User Settings** → **Access Tokens**
3. Click **Generate New Token**
4. Copy the token (you won't see it again!)

### 1.4 Set Up MLflow Experiment

In a Databricks notebook, run:
```python
import mlflow
mlflow.set_tracking_uri("https://<your-workspace>.cloud.databricks.com")
mlflow.create_experiment("YOLOv11s_Autonomous_Driving_OD_Predictions")
mlflow.create_experiment("YOLOv11s_Autonomous_Driving_OD")  # For training runs
```

## 📁 Step 2: Prepare Your Repository

### 2.1 File Structure

Your repository should have:
```
your-repo/
├── streamlit_app.py          # Main Streamlit app
├── streamlit_monitor.py       # Monitoring dashboard
├── requirements_streamlit.txt # Dependencies
├── .streamlit/
│   ├── config.toml           # Streamlit config
│   └── secrets.toml          # Secrets (not in git!)
├── best68.pt                  # Model file (or host separately)
└── README.md
```

### 2.2 Add Model File

**Option A: Include in Repository** (for small models < 100MB)
- Add `best68.pt` to your repository
- Update `.gitignore` if needed

**Option B: Host Separately** (recommended for large models)
- Upload to cloud storage (S3, GCS, Azure Blob)
- Download in app using `st.cache_resource`
- Or use Hugging Face Model Hub

### 2.3 Update Configuration

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Fill in your Databricks credentials:
   ```toml
   [mlflow]
   tracking_uri = "https://your-workspace.cloud.databricks.com"
   databricks_token = "your-token-here"
   
   [model]
   path = "best68.pt"
   ```

**⚠️ Important:** Add `.streamlit/secrets.toml` to `.gitignore` to keep secrets safe!

## 🌐 Step 3: Deploy to Streamlit Cloud

### 3.1 Push to GitHub

1. Initialize git repository (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit with Streamlit app"
   ```

2. Create a GitHub repository and push:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

### 3.2 Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub account
4. Select your repository
5. Choose branch (usually `main`)
6. Set main file path: `streamlit_app.py`
7. Click **Deploy**

### 3.3 Configure Secrets

1. In Streamlit Cloud, go to your app settings
2. Click **Secrets** tab
3. Add your secrets in TOML format:
   ```toml
   [mlflow]
   tracking_uri = "https://your-workspace.cloud.databricks.com"
   databricks_token = "your-token-here"
   
   [model]
   path = "best68.pt"
   ```
4. Save and redeploy

## 🔧 Step 4: Configure Environment Variables (Alternative)

Instead of secrets.toml, you can use environment variables in Streamlit Cloud:

1. Go to app settings → **Advanced settings**
2. Add environment variables:
   - `MLFLOW_TRACKING_URI=https://your-workspace.cloud.databricks.com`
   - `DATABRICKS_TOKEN=your-token-here`
   - `MODEL_PATH=best68.pt`
   - `PREDICTION_LOG_DIR=./prediction_logs`

## 📊 Step 5: Deploy Monitoring Dashboard

To deploy the monitoring dashboard as a separate app:

1. Create a new app in Streamlit Cloud
2. Set main file path: `streamlit_monitor.py`
3. Use the same secrets/environment variables
4. Deploy

## 🧪 Step 6: Test Your Deployment

1. Open your deployed app URL
2. Upload a test image
3. Check that predictions work
4. Verify MLflow logging:
   - Go to Databricks workspace
   - Navigate to **Experiments** → `YOLOv11s_Autonomous_Driving_OD_Predictions`
   - Check that runs are being logged

## 🔍 Troubleshooting

### Issue: Model not loading
- **Solution**: Ensure model file is in repository or accessible via URL
- Check file path in secrets/config

### Issue: MLflow connection failed
- **Solution**: Verify tracking URI and token are correct
- Check Databricks workspace is accessible
- Ensure token has proper permissions

### Issue: App crashes on startup
- **Solution**: Check `requirements_streamlit.txt` has all dependencies
- Review Streamlit Cloud logs for errors
- Ensure Python version compatibility (Streamlit Cloud uses Python 3.9+)

### Issue: Slow inference
- **Solution**: Consider using GPU-enabled Streamlit Cloud (paid tier)
- Optimize model size
- Use model quantization

## 📝 Additional Configuration

### Custom Domain (Optional)
1. In Streamlit Cloud settings
2. Add custom domain
3. Configure DNS records

### Auto-deploy from GitHub
- Streamlit Cloud auto-deploys on push to main branch
- Configure branch in app settings

### Resource Limits
- **Free tier**: Limited CPU/RAM
- **Team tier**: More resources, custom domains
- Consider upgrading for production workloads

## 🔐 Security Best Practices

1. **Never commit secrets** - Use `.gitignore` for `secrets.toml`
2. **Rotate tokens regularly** - Update Databricks tokens periodically
3. **Use environment variables** - For CI/CD pipelines
4. **Limit token permissions** - Create tokens with minimal required permissions
5. **Monitor usage** - Check Streamlit Cloud and Databricks usage regularly

## 📚 Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Databricks Community Edition](https://www.databricks.com/try-databricks)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

## 🎉 Next Steps

1. Set up automated monitoring alerts
2. Configure model versioning in MLflow
3. Set up CI/CD for model updates
4. Add A/B testing for model versions
5. Implement model retraining pipeline

---

**Need Help?** Check the Streamlit community forum or open an issue in your repository.

