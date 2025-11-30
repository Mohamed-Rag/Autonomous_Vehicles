# Streamlit Quick Start Guide

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Configure Secrets

Copy the example secrets file and fill in your values:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your Databricks credentials:
```toml
[mlflow]
tracking_uri = "https://your-workspace.cloud.databricks.com"
databricks_token = "your-token-here"

[model]
path = "best68.pt"
```

### 3. Run the App Locally

**Main App:**
```bash
streamlit run streamlit_app.py
```

**Monitoring Dashboard:**
```bash
streamlit run streamlit_monitor.py
```

## 📁 Files Created

- `streamlit_app.py` - Main Streamlit application
- `streamlit_monitor.py` - Monitoring dashboard
- `requirements_streamlit.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `.streamlit/secrets.toml.example` - Secrets template
- `STREAMLIT_DEPLOYMENT.md` - Full deployment guide

## 🔑 Getting Databricks Credentials

1. **Sign up**: [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. **Get Tracking URI**: 
   - Format: `https://<workspace-id>.cloud.databricks.com`
   - Found in your workspace URL
3. **Create Access Token**:
   - User Settings → Access Tokens → Generate New Token

## 🌐 Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set main file: `streamlit_app.py`
5. Add secrets in Streamlit Cloud settings
6. Deploy!

See `STREAMLIT_DEPLOYMENT.md` for detailed instructions.

## ⚠️ Important Notes

- **Never commit** `.streamlit/secrets.toml` (add to `.gitignore`)
- Model file (`best68.pt`) should be in the repository or accessible via URL
- For large models, consider hosting separately (S3, GCS, etc.)

## 🆘 Troubleshooting

**MLflow connection issues?**
- Verify tracking URI format
- Check token is valid
- Ensure Databricks workspace is accessible

**Model not loading?**
- Check model path in secrets/config
- Verify model file exists
- Check file permissions

**App crashes?**
- Check all dependencies in `requirements_streamlit.txt`
- Review Streamlit logs
- Verify Python version (3.9+)

