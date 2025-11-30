"""
Model Monitoring and Drift Detection System
Tracks model performance over time and detects issues like model drift.
"""

import os
import json
import mlflow
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration - Cloud-ready with environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
EXPERIMENT_NAME = "YOLOv11s_Autonomous_Driving_OD_Predictions"
PREDICTION_LOG_DIR = os.getenv("PREDICTION_LOG_DIR", "./prediction_logs")

# Setup Databricks authentication if token is provided
if DATABRICKS_TOKEN:
    os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
    if MLFLOW_TRACKING_URI and 'databricks' in MLFLOW_TRACKING_URI:
        try:
            host = MLFLOW_TRACKING_URI.replace('https://', '').split('/')[0]
            os.environ['DATABRICKS_HOST'] = host
        except:
            pass
ALERT_THRESHOLDS = {
    "accuracy_drop": 0.10,  # 10% drop in accuracy
    "inference_time_ms": 1000,  # 1 second
    "error_rate": 0.05,  # 5% error rate
    "detection_drop": 0.20  # 20% drop in detections
}

# Email configuration (optional - set these if you want email alerts)
EMAIL_CONFIG = {
    "enabled": False,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "",
    "sender_password": "",
    "recipient_emails": []
}


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self):
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        self.baseline_metrics = self._load_baseline_metrics()
        
    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics from training runs."""
        try:
            mlflow.set_experiment("YOLOv11s_Autonomous_Driving_OD")
            experiment = mlflow.get_experiment_by_name("YOLOv11s_Autonomous_Driving_OD")
            if experiment is None:
                raise ValueError("Experiment not found")
            
            # Fetch all runs and sort in Python to avoid parsing issues with special characters
            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=100  # Adjust if you have more runs
            )
            
            # Sort by final_mAP@0.5 metric in descending order
            runs_with_metric = []
            for run in runs:
                metric_value = run.data.metrics.get("final_mAP@0.5")
                if metric_value is not None:
                    runs_with_metric.append((metric_value, run))
            
            if runs_with_metric:
                # Sort by metric value descending
                runs_with_metric.sort(key=lambda x: x[0], reverse=True)
                run = runs_with_metric[0][1]  # Get the run with highest metric
                return {
                    "mAP50": run.data.metrics.get("final_mAP@0.5", 0.0),
                    "precision": run.data.metrics.get("metrics/precision", 0.0),
                    "recall": run.data.metrics.get("metrics/recall", 0.0)
                }
        except Exception as e:
            print(f"⚠️ Warning: Could not load baseline metrics: {e}")
        
        return {
            "mAP50": 0.5,
            "precision": 0.7,
            "recall": 0.6
        }
    
    def analyze_prediction_logs(self, days: int = 7) -> Dict[str, Any]:
        """Analyze prediction logs for the last N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stats = {
            "total_predictions": 0,
            "total_detections": 0,
            "avg_inference_time_ms": [],
            "class_distribution": defaultdict(int),
            "errors": 0,
            "daily_stats": defaultdict(lambda: {"predictions": 0, "detections": 0})
        }
        
        log_files = [f for f in os.listdir(PREDICTION_LOG_DIR) if f.endswith('.jsonl')]
        
        for log_file in log_files:
            log_path = os.path.join(PREDICTION_LOG_DIR, log_file)
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
                                stats["avg_inference_time_ms"].append(entry.get("inference_time_ms", 0))
                                
                                date_key = timestamp.strftime("%Y-%m-%d")
                                stats["daily_stats"][date_key]["predictions"] += 1
                                stats["daily_stats"][date_key]["detections"] += entry.get("num_detections", 0)
                                
                                # Count class distribution
                                for pred in entry.get("predictions", []):
                                    cls = pred.get("class", "unknown")
                                    stats["class_distribution"][cls] += 1
                        except json.JSONDecodeError:
                            stats["errors"] += 1
                        except Exception as e:
                            stats["errors"] += 1
                            
            except Exception as e:
                print(f"⚠️ Error reading log file {log_file}: {e}")
        
        # Calculate averages
        if stats["avg_inference_time_ms"]:
            stats["avg_inference_time_ms"] = np.mean(stats["avg_inference_time_ms"])
        else:
            stats["avg_inference_time_ms"] = 0
        
        stats["avg_detections_per_image"] = (
            stats["total_detections"] / stats["total_predictions"]
            if stats["total_predictions"] > 0 else 0
        )
        
        return stats
    
    def check_model_drift(self, current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for model drift and performance degradation."""
        alerts = []
        
        # Check inference time
        if current_stats["avg_inference_time_ms"] > ALERT_THRESHOLDS["inference_time_ms"]:
            alerts.append({
                "type": "performance_degradation",
                "severity": "high",
                "message": f"Average inference time ({current_stats['avg_inference_time_ms']:.2f} ms) exceeds threshold ({ALERT_THRESHOLDS['inference_time_ms']} ms)",
                "metric": "inference_time_ms",
                "value": current_stats["avg_inference_time_ms"],
                "threshold": ALERT_THRESHOLDS["inference_time_ms"]
            })
        
        # Check error rate
        if current_stats["total_predictions"] > 0:
            error_rate = current_stats["errors"] / current_stats["total_predictions"]
            if error_rate > ALERT_THRESHOLDS["error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "high",
                    "message": f"Error rate ({error_rate*100:.2f}%) exceeds threshold ({ALERT_THRESHOLDS['error_rate']*100}%)",
                    "metric": "error_rate",
                    "value": error_rate,
                    "threshold": ALERT_THRESHOLDS["error_rate"]
                })
        
        # Check detection drop (compare with baseline or recent average)
        # This is a simplified check - you might want to compare with historical data
        if current_stats["avg_detections_per_image"] < 0.5:
            alerts.append({
                "type": "low_detection_rate",
                "severity": "medium",
                "message": f"Average detections per image ({current_stats['avg_detections_per_image']:.2f}) is unusually low",
                "metric": "avg_detections_per_image",
                "value": current_stats["avg_detections_per_image"]
            })
        
        # Check for class distribution shifts (simplified)
        total_detections = sum(current_stats["class_distribution"].values())
        if total_detections > 0:
            for cls, count in current_stats["class_distribution"].items():
                percentage = count / total_detections
                # Alert if a single class dominates (might indicate drift)
                if percentage > 0.8:
                    alerts.append({
                        "type": "class_distribution_shift",
                        "severity": "medium",
                        "message": f"Class '{cls}' dominates detections ({percentage*100:.1f}%) - possible drift",
                        "metric": "class_distribution",
                        "class": cls,
                        "percentage": percentage
                    })
        
        return alerts
    
    def generate_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        stats = self.analyze_prediction_logs(days=days)
        alerts = self.check_model_drift(stats)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_period_days": days,
            "statistics": stats,
            "alerts": alerts,
            "baseline_metrics": self.baseline_metrics,
            "status": "healthy" if len(alerts) == 0 else "degraded"
        }
        
        return report
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert via email (if configured)."""
        if not EMAIL_CONFIG["enabled"]:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG["sender_email"]
            msg['To'] = ", ".join(EMAIL_CONFIG["recipient_emails"])
            msg['Subject'] = f"🚨 Model Monitoring Alert: {alert['type']}"
            
            body = f"""
            Model Monitoring Alert
            
            Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            
            Timestamp: {datetime.now().isoformat()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Alert sent via email")
        except Exception as e:
            print(f"⚠️ Failed to send alert email: {e}")
    
    def log_monitoring_metrics(self, report: Dict[str, Any]):
        """Log monitoring metrics to MLflow."""
        try:
            mlflow.set_experiment("Model_Monitoring")
            
            with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log statistics
                stats = report["statistics"]
                mlflow.log_metric("total_predictions", stats["total_predictions"])
                mlflow.log_metric("total_detections", stats["total_detections"])
                mlflow.log_metric("avg_detections_per_image", stats["avg_detections_per_image"])
                mlflow.log_metric("avg_inference_time_ms", stats["avg_inference_time_ms"])
                mlflow.log_metric("error_count", stats["errors"])
                
                # Log alerts count
                mlflow.log_metric("alerts_count", len(report["alerts"]))
                mlflow.set_tag("status", report["status"])
                
                # Log class distribution
                for cls, count in stats["class_distribution"].items():
                    mlflow.log_metric(f"detections_{cls}", count)
                
                print("✅ Monitoring metrics logged to MLflow")
        except Exception as e:
            print(f"⚠️ Warning: Could not log to MLflow: {e}")


def main():
    """Run monitoring check."""
    print("=" * 70)
    print(" Model Monitoring System")
    print("=" * 70)
    
    monitor = ModelMonitor()
    
    # Generate report for last 7 days
    report = monitor.generate_monitoring_report(days=7)
    
    # Print report
    print(f"\n📊 Monitoring Report ({report['monitoring_period_days']} days)")
    print("-" * 70)
    print(f"Status: {report['status']}")
    print(f"Total Predictions: {report['statistics']['total_predictions']}")
    print(f"Total Detections: {report['statistics']['total_detections']}")
    print(f"Avg Detections/Image: {report['statistics']['avg_detections_per_image']:.2f}")
    print(f"Avg Inference Time: {report['statistics']['avg_inference_time_ms']:.2f} ms")
    print(f"Errors: {report['statistics']['errors']}")
    
    # Print alerts
    if report['alerts']:
        print(f"\n🚨 Alerts ({len(report['alerts'])}):")
        for alert in report['alerts']:
            print(f"  - [{alert['severity'].upper()}] {alert['message']}")
            monitor.send_alert(alert)
    else:
        print("\n✅ No alerts - model performing normally")
    
    # Log to MLflow
    monitor.log_monitoring_metrics(report)
    
    # Save report to file
    report_file = os.path.join(PREDICTION_LOG_DIR, f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

