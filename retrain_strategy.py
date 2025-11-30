"""
Model Retraining Strategy
Automated retraining plan to update models with new data or correct performance degradation.
"""

import os
import json
import mlflow
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PREDICTION_LOG_DIR = r"H:\Startups\Autonomus Car Detection DEPI\prediction_logs"
MONITORING_REPORT_DIR = r"H:\Startups\Autonomus Car Detection DEPI\prediction_logs"
TRAIN_SCRIPT = r"H:\Startups\Autonomus Car Detection DEPI\train.py"

# Retraining triggers
RETRAINING_TRIGGERS = {
    "performance_drop_threshold": 0.10,  # 10% drop in mAP
    "error_rate_threshold": 0.05,  # 5% error rate
    "days_since_last_training": 30,  # Retrain every 30 days
    "min_new_data_samples": 1000,  # Minimum new data samples
    "drift_detection_threshold": 0.15  # 15% distribution shift
}


class RetrainingStrategy:
    """Manages model retraining based on monitoring data."""
    
    def __init__(self):
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    def get_latest_training_metrics(self) -> Optional[Dict[str, float]]:
        """Get metrics from the latest training run."""
        try:
            mlflow.set_experiment("YOLOv11s_Autonomous_Driving_OD")
            experiment = mlflow.get_experiment_by_name("YOLOv11s_Autonomous_Driving_OD")
            
            if experiment:
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if runs:
                    run = runs[0]
                    return {
                        "mAP50": run.data.metrics.get("final_mAP@0.5", 0.0),
                        "mAP50_95": run.data.metrics.get("metrics/mAP50-95", 0.0),
                        "precision": run.data.metrics.get("metrics/precision", 0.0),
                        "recall": run.data.metrics.get("metrics/recall", 0.0),
                        "training_date": datetime.fromtimestamp(run.info.start_time / 1000).isoformat()
                    }
        except Exception as e:
            print(f"⚠️ Warning: Could not get latest training metrics: {e}")
        
        return None
    
    def get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from monitoring."""
        try:
            # Get latest monitoring report
            report_files = sorted(
                [f for f in os.listdir(MONITORING_REPORT_DIR) if f.startswith("monitoring_report_")],
                reverse=True
            )
            
            if report_files:
                latest_report_path = os.path.join(MONITORING_REPORT_DIR, report_files[0])
                with open(latest_report_path, 'r') as f:
                    report = json.load(f)
                    return {
                        "avg_inference_time_ms": report["statistics"].get("avg_inference_time_ms", 0),
                        "error_rate": report["statistics"].get("errors", 0) / max(report["statistics"].get("total_predictions", 1), 1),
                        "avg_detections_per_image": report["statistics"].get("avg_detections_per_image", 0),
                        "alerts_count": len(report.get("alerts", [])),
                        "status": report.get("status", "unknown")
                    }
        except Exception as e:
            print(f"⚠️ Warning: Could not get current performance metrics: {e}")
        
        return {}
    
    def check_retraining_conditions(self) -> Dict[str, Any]:
        """Check if retraining conditions are met."""
        conditions = {
            "should_retrain": False,
            "reasons": [],
            "urgency": "low"
        }
        
        # Get latest training metrics
        latest_training = self.get_latest_training_metrics()
        current_performance = self.get_current_performance_metrics()
        
        if not latest_training:
            conditions["should_retrain"] = True
            conditions["reasons"].append("No previous training found")
            conditions["urgency"] = "medium"
            return conditions
        
        # Check 1: Days since last training
        last_training_date = datetime.fromisoformat(latest_training["training_date"])
        days_since_training = (datetime.now() - last_training_date).days
        
        if days_since_training >= RETRAINING_TRIGGERS["days_since_last_training"]:
            conditions["should_retrain"] = True
            conditions["reasons"].append(f"Last training was {days_since_training} days ago (threshold: {RETRAINING_TRIGGERS['days_since_last_training']} days)")
            conditions["urgency"] = "medium"
        
        # Check 2: Performance degradation
        if current_performance.get("status") == "degraded":
            conditions["should_retrain"] = True
            conditions["reasons"].append("Model performance degraded (monitoring alerts detected)")
            conditions["urgency"] = "high"
        
        # Check 3: High error rate
        error_rate = current_performance.get("error_rate", 0)
        if error_rate > RETRAINING_TRIGGERS["error_rate_threshold"]:
            conditions["should_retrain"] = True
            conditions["reasons"].append(f"Error rate ({error_rate*100:.2f}%) exceeds threshold ({RETRAINING_TRIGGERS['error_rate_threshold']*100}%)")
            conditions["urgency"] = "high"
        
        # Check 4: Low detection rate (potential drift)
        avg_detections = current_performance.get("avg_detections_per_image", 0)
        if avg_detections < 0.3:  # Very low detection rate
            conditions["should_retrain"] = True
            conditions["reasons"].append(f"Low average detections per image ({avg_detections:.2f}) - possible model drift")
            conditions["urgency"] = "high"
        
        return conditions
    
    def prepare_retraining_data(self) -> Dict[str, Any]:
        """Prepare data summary for retraining."""
        # This would typically involve:
        # 1. Collecting new labeled data
        # 2. Data validation
        # 3. Data augmentation strategy
        # 4. Train/val/test split
        
        data_info = {
            "new_samples_count": 0,  # Would be calculated from new data
            "total_samples": 0,
            "class_distribution": {},
            "data_quality_score": 1.0,
            "augmentation_strategy": "standard"
        }
        
        # In a real scenario, you would:
        # - Check for new labeled data in your data directory
        # - Validate data quality
        # - Calculate statistics
        
        return data_info
    
    def execute_retraining(self, reason: str = "Scheduled retraining"):
        """Execute model retraining."""
        print("=" * 70)
        print(" Starting Model Retraining")
        print("=" * 70)
        print(f"Reason: {reason}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        try:
            # Log retraining start to MLflow
            mlflow.set_experiment("Model_Retraining")
            with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("retraining_reason", reason)
                mlflow.log_param("retraining_timestamp", datetime.now().isoformat())
                
                # Execute training script
                print("\n Executing training script...")
                result = subprocess.run(
                    [sys.executable, TRAIN_SCRIPT],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    mlflow.log_param("retraining_status", "success")
                    print("✅ Retraining completed successfully")
                else:
                    mlflow.log_param("retraining_status", "failed")
                    mlflow.log_param("retraining_error", result.stderr)
                    print(f"❌ Retraining failed: {result.stderr}")
                    return False
                
        except Exception as e:
            print(f"❌ Error during retraining: {e}")
            return False
        
        print("=" * 70)
        return True
    
    def generate_retraining_plan(self) -> Dict[str, Any]:
        """Generate comprehensive retraining plan."""
        conditions = self.check_retraining_conditions()
        data_info = self.prepare_retraining_data()
        
        plan = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": conditions["should_retrain"],
            "urgency": conditions["urgency"],
            "reasons": conditions["reasons"],
            "data_info": data_info,
            "recommended_actions": []
        }
        
        if conditions["should_retrain"]:
            if conditions["urgency"] == "high":
                plan["recommended_actions"].append("Execute retraining immediately")
            elif conditions["urgency"] == "medium":
                plan["recommended_actions"].append("Schedule retraining within 24-48 hours")
            else:
                plan["recommended_actions"].append("Schedule retraining within 1 week")
            
            plan["recommended_actions"].extend([
                "Collect and validate new training data",
                "Review monitoring reports for specific issues",
                "Update hyperparameters if needed",
                "Validate retrained model before deployment"
            ])
        else:
            plan["recommended_actions"].append("Continue monitoring - no retraining needed at this time")
        
        return plan


def main():
    """Main retraining strategy check."""
    print("=" * 70)
    print(" Model Retraining Strategy")
    print("=" * 70)
    
    strategy = RetrainingStrategy()
    
    # Generate retraining plan
    plan = strategy.generate_retraining_plan()
    
    # Print plan
    print(f"\n📋 Retraining Plan")
    print("-" * 70)
    print(f"Should Retrain: {plan['should_retrain']}")
    print(f"Urgency: {plan['urgency']}")
    
    if plan['reasons']:
        print(f"\nReasons:")
        for reason in plan['reasons']:
            print(f"  - {reason}")
    
    print(f"\nRecommended Actions:")
    for action in plan['recommended_actions']:
        print(f"  - {action}")
    
    # Execute retraining if needed and urgent
    if plan['should_retrain'] and plan['urgency'] == "high":
        print("\n" + "=" * 70)
        response = input("High urgency retraining detected. Execute retraining now? (y/n): ")
        if response.lower() == 'y':
            reason = "; ".join(plan['reasons'])
            strategy.execute_retraining(reason=reason)
    
    # Save plan
    plan_file = os.path.join(PREDICTION_LOG_DIR, f"retraining_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"\n✅ Plan saved to: {plan_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

