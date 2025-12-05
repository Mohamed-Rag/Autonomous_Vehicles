import os
import json
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLOPS_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLOPS_TRACKING_PASSWORD")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")

INFERENCE_EXPERIMENT_NAME = "YOLOv11s_Autonomous_Driving_OD_Predictions"
TRAINING_EXPERIMENT_NAME = "YOLOv11s_Autonomous_Driving_OD"

THRESHOLDS = {
    "latency_p95_ms": 1000,
    "latency_p99_ms": 1500,
    "drift_psi": 0.1,
    "drift_ks_pvalue": 0.05,
    "error_rate": 0.05,
    "min_predictions_per_day": 10,
    "outlier_percentage": 0.05,
}

METRICS_HISTORY_FILE = "./monitoring_history.json"
DASHBOARD_OUTPUT = "./monitoring_dashboard.html"

def setup_dagshub_mlflow():
    global MLFLOW_TRACKING_URI
    if not MLFLOW_TRACKING_URI and DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME:
        constructed = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
        print(f"ℹ️ Constructed MLflow URI from DAGSHUB envs: {constructed}")
        MLFLOW_TRACKING_URI = constructed
    if not MLFLOW_TRACKING_URI:
        print("⚠️ MLFLOW_TRACKING_URI not set and DAGSHUB envs not present. Can't connect.")
        return False
    if MLFLOW_USERNAME:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    if MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _ = mlflow.tracking.MlflowClient()
        print(f"✅ Connected to MLflow: {MLFLOW_TRACKING_URI}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to MLflow at {MLFLOW_TRACKING_URI}: {e}")
        return False

class MLflowDataFetcher:
    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()

    def get_inference_runs(self, days_back: int = 7) -> pd.DataFrame:
        try:
            experiment = mlflow.get_experiment_by_name(INFERENCE_EXPERIMENT_NAME)
            if experiment is None:
                print(f"⚠️ Experiment '{INFERENCE_EXPERIMENT_NAME}' not found.")
                return pd.DataFrame()
            start_time = datetime.now() - timedelta(days=days_back)
            start_time_ms = int(start_time.timestamp() * 1000)

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attributes.start_time >= {start_time_ms}",
                order_by=["start_time DESC"],
                max_results=10000
            )

            if runs is None or len(runs) == 0:
                print(f"⚠️ No runs found in experiment '{INFERENCE_EXPERIMENT_NAME}'.")
                return pd.DataFrame()

            df = runs.copy()
            df = self._normalize_runs_df(df)
            print(f"✅ Fetched and normalized {len(df)} inference runs from the last {days_back} days.")
            return df

        except Exception as e:
            print(f"❌ Error fetching runs: {e}")
            return pd.DataFrame()

    def _normalize_runs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        latency_candidates = [
            'metrics.latency_ms', 'latency_ms',
            'metrics.inference_time_seconds', 'inference_time_seconds',
            'metrics.inference_time_s', 'inference_time_s'
        ]
        df_cols = df.columns.tolist()
        found_latency_col = None
        for c in latency_candidates:
            if c in df_cols:
                found_latency_col = c
                break
        if found_latency_col:
            if 'ms' in found_latency_col:
                df['metrics.latency_ms'] = df[found_latency_col]
            else:
                df['metrics.latency_ms'] = df[found_latency_col].astype(float) * 1000.0
        else:
            df['metrics.latency_ms'] = np.nan

        detections_candidates = [
            'metrics.num_detections', 'metrics.total_detections',
            'total_detections', 'num_detections'
        ]
        found_det_col = None
        for c in detections_candidates:
            if c in df_cols:
                found_det_col = c
                break
        if found_det_col:
            df['metrics.num_detections'] = df[found_det_col].astype(float)
        else:
            count_cols = [c for c in df_cols if ('detection_' in c and not c.endswith('_conf')) or c.startswith('metrics.count_')]
            if count_cols:
                df['metrics.num_detections'] = df[count_cols].notna().sum(axis=1)
            else:
                df['metrics.num_detections'] = np.nan

        class_count_map = defaultdict(list)
        for c in df_cols:
            if 'detection_' in c:
                after = c.split('detection_', 1)[1]
                parts = after.split('_')
                if len(parts) >= 2 and parts[-1].isdigit():
                    class_name = "_".join(parts[:-1])
                else:
                    class_name = after.split('_conf')[0]
                class_count_map[class_name].append(c)
            elif c.startswith('metrics.count_'):
                class_name = c.replace('metrics.count_', '')
                class_count_map[class_name].append(c)

        for cls, cols in class_count_map.items():
            df[f'metrics.count_{cls}'] = df[cols].notna().sum(axis=1)

        return df

    def get_baseline_metrics(self) -> Dict[str, float]:
        try:
            experiment = mlflow.get_experiment_by_name(TRAINING_EXPERIMENT_NAME)
            if experiment is None:
                print(f"⚠️ Training experiment '{TRAINING_EXPERIMENT_NAME}' not found.")
                return {}

            runs = self.client.search_runs(experiment_ids=[experiment.experiment_id], max_results=100)
            best_map = 0.0
            best_metrics = {}
            for run in runs:
                map_candidates = [
                    "metrics/mAP50-95", "mAP50-95", "metrics.mAP50-95",
                    "metrics/mAP", "mAP", "metrics.mAP"
                ]
                map_value = 0.0
                for mc in map_candidates:
                    mv = run.data.metrics.get(mc, None)
                    if mv is not None:
                        try:
                            map_value = float(mv)
                            break
                        except:
                            continue
                if map_value > best_map:
                    best_map = map_value
                    best_metrics = {
                        "mAP50-95": map_value,
                        "precision": run.data.metrics.get("metrics/precision", run.data.metrics.get("precision", 0)),
                        "recall": run.data.metrics.get("metrics/recall", run.data.metrics.get("recall", 0))
                    }
            if best_metrics:
                print(f"✅ Loaded baseline metrics: mAP={best_map:.4f}")
            return best_metrics
        except Exception as e:
            print(f"⚠️ Could not load baseline: {e}")
            return {}

class PerformanceAnalyzer:
    @staticmethod
    def calculate_percentiles(df: pd.DataFrame, metric: str) -> Dict[str, float]:
        if metric not in df.columns or df[metric].isna().all():
            return {"p50": 0, "p75": 0, "p95": 0, "p99": 0, "mean": 0, "std": 0}
        values = df[metric].dropna().astype(float)
        return {
            "p50": np.percentile(values, 50),
            "p75": np.percentile(values, 75),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "mean": np.mean(values),
            "std": np.std(values)
        }

    @staticmethod
    def calculate_throughput(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"predictions_per_day": 0, "predictions_per_hour": 0}
        if 'start_time' not in df.columns:
            if 'attributes.start_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['attributes.start_time'], unit='ms')
            else:
                return {"predictions_per_day": len(df), "predictions_per_hour": len(df)}
        else:
            df['timestamp'] = pd.to_datetime(df['start_time'])
        time_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        if time_range == 0:
            return {"predictions_per_day": len(df), "predictions_per_hour": len(df)}
        predictions_per_second = len(df) / time_range
        return {
            "predictions_per_day": predictions_per_second * 86400,
            "predictions_per_hour": predictions_per_second * 3600,
            "time_range_hours": time_range / 3600
        }

    @staticmethod
    def detect_outliers(df: pd.DataFrame, metric: str, threshold: float = 3.0) -> Tuple[np.ndarray, float]:
        if metric not in df.columns:
            return np.array([]), 0.0
        values = df[metric].dropna().astype(float)
        if len(values) < 2:
            return np.array([]), 0.0
        z_scores = np.abs(stats.zscore(values))
        outliers = z_scores > threshold
        outlier_percentage = np.sum(outliers) / len(values)
        return outliers, outlier_percentage

class DriftDetector:
    @staticmethod
    def ks_test(baseline: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        if len(baseline) < 2 or len(current) < 2:
            return {"statistic": 0, "pvalue": 1.0, "drift_detected": False}
        statistic, pvalue = stats.ks_2samp(baseline, current)
        drift_detected = pvalue < THRESHOLDS["drift_ks_pvalue"]
        return {"statistic": statistic, "pvalue": pvalue, "drift_detected": drift_detected}

    @staticmethod
    def chi_square_test(baseline_dist: Dict, current_dist: Dict) -> Dict[str, Any]:
        all_classes = set(baseline_dist.keys()) | set(current_dist.keys())
        baseline_counts = np.array([baseline_dist.get(c, 0) for c in all_classes]) + 1
        current_counts = np.array([current_dist.get(c, 0) for c in all_classes]) + 1
        if np.sum(baseline_counts) == 0 or np.sum(current_counts) == 0:
            return {"statistic": 0, "pvalue": 1.0, "drift_detected": False}
        try:
            statistic, pvalue = stats.chisquare(current_counts, baseline_counts)
            drift_detected = pvalue < THRESHOLDS["drift_ks_pvalue"]
            return {"statistic": statistic, "pvalue": pvalue, "drift_detected": drift_detected}
        except:
            return {"statistic": 0, "pvalue": 1.0, "drift_detected": False}

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        if len(baseline) < 2 or len(current) < 2:
            return 0.0
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)
        baseline_dist = baseline_dist / len(baseline) + 1e-10
        current_dist = current_dist / len(current) + 1e-10
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
        return psi

class DataQualityChecker:
    @staticmethod
    def check_quality(df: pd.DataFrame) -> Dict[str, Any]:
        checks = {}
        if 'metrics.num_detections' in df.columns:
            missing_detections = df['metrics.num_detections'].isna().sum()
            checks["missing_detections"] = {
                "count": int(missing_detections),
                "percentage": missing_detections / len(df) if len(df) > 0 else 0
            }
        else:
            checks["missing_detections"] = {"count": len(df), "percentage": 1.0}

        if 'metrics.latency_ms' in df.columns:
            _, outlier_pct = PerformanceAnalyzer.detect_outliers(df, 'metrics.latency_ms')
            checks["latency_outliers"] = {
                "percentage": outlier_pct,
                "exceeds_threshold": outlier_pct > THRESHOLDS["outlier_percentage"]
            }
        else:
            checks["latency_outliers"] = {"percentage": 0.0, "exceeds_threshold": False}

        throughput = PerformanceAnalyzer.calculate_throughput(df)
        checks["throughput"] = {
            "predictions_per_day": throughput["predictions_per_day"],
            "below_threshold": throughput["predictions_per_day"] < THRESHOLDS["min_predictions_per_day"]
        }

        required_fields = ['metrics.latency_ms', 'metrics.num_detections']
        missing_fields = [f for f in required_fields if f not in df.columns]
        checks["missing_fields"] = missing_fields

        return checks

class RetrainingTrigger:
    @staticmethod
    def should_retrain(drift_results: Dict, quality_results: Dict, performance: Dict) -> Dict[str, Any]:
        triggers = []
        if drift_results.get("latency_ks", {}).get("drift_detected", False):
            triggers.append("Latency distribution drift detected (KS test)")
        if drift_results.get("psi", 0) > THRESHOLDS["drift_psi"]:
            triggers.append(f"High PSI score: {drift_results.get('psi', 0):.4f}")
        if performance.get("p95", 0) > THRESHOLDS["latency_p95_ms"]:
            triggers.append(f"P95 latency exceeds threshold: {performance.get('p95', 0):.2f}ms")
        if quality_results.get("latency_outliers", {}).get("exceeds_threshold", False):
            triggers.append("High percentage of latency outliers")
        if quality_results.get("throughput", {}).get("below_threshold", False):
            triggers.append("Low prediction throughput")

        return {
            "retrain_recommended": len(triggers) > 0,
            "triggers": triggers,
            "severity": "high" if len(triggers) >= 2 else "medium" if len(triggers) == 1 else "low"
        }

class DashboardGenerator:
    @staticmethod
    def create_dashboard(df: pd.DataFrame, performance: Dict, drift: Dict, quality: Dict, retraining: Dict):
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Latency Over Time',
                'Latency Distribution',
                'Throughput (Predictions/Hour)',
                'Drift Detection Scores',
                'Class Distribution',
                'Data Quality Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )

        if 'metrics.latency_ms' in df.columns and 'start_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['start_time'])
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['metrics.latency_ms'], mode='markers', name='Latency'), row=1, col=1)
            fig.add_hline(y=performance.get('p95', 0), line_dash="dash", line_color="red", annotation_text=f"P95: {performance.get('p95', 0):.2f}ms", row=1, col=1)

        if 'metrics.latency_ms' in df.columns:
            fig.add_trace(go.Histogram(x=df['metrics.latency_ms'].dropna(), nbinsx=50, name='Latency Distribution'), row=1, col=2)

        if 'start_time' in df.columns:
            df['date'] = pd.to_datetime(df['start_time']).dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            fig.add_trace(go.Bar(x=daily_counts['date'], y=daily_counts['count'], name='Daily Predictions'), row=2, col=1)

        drift_scores = {'PSI': drift.get('psi', 0), 'KS Statistic': drift.get('latency_ks', {}).get('statistic', 0)}
        fig.add_trace(go.Bar(x=list(drift_scores.keys()), y=list(drift_scores.values()), name='Drift Metrics'), row=2, col=2)

        class_cols = [c for c in df.columns if c.startswith('metrics.count_')]
        if class_cols:
            class_counts = df[class_cols].sum().sort_values(ascending=False)
            class_names = [c.replace('metrics.count_', '') for c in class_counts.index]
            fig.add_trace(go.Bar(x=class_names, y=class_counts.values, name='Class Distribution'), row=3, col=1)

        quality_score = 100
        if quality.get("latency_outliers", {}).get("exceeds_threshold", False):
            quality_score -= 30
        if quality.get("throughput", {}).get("below_threshold", False):
            quality_score -= 30

        fig.add_trace(go.Indicator(mode="gauge+number", value=quality_score, title={'text': "Data Quality Score"}, gauge={'axis': {'range': [0, 100]}}), row=3, col=2)

        fig.update_layout(height=1200, showlegend=True, title_text="🚗 YOLO Model Monitoring Dashboard - Hugging Face Deployment")
        fig.write_html(DASHBOARD_OUTPUT)
        print(f"✅ Dashboard saved to: {DASHBOARD_OUTPUT}")
        return fig

class HistoricalTracker:
    @staticmethod
    def save_metrics(metrics: Dict):
        def _conv(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, (np.bool_,)):
                return bool(x)
            if isinstance(x, (np.ndarray,)):
                return x.tolist()
            if isinstance(x, pd.Timestamp):
                return x.isoformat()
            return x

        history = []
        if os.path.exists(METRICS_HISTORY_FILE):
            try:
                with open(METRICS_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except Exception:
                history = []

        metrics['timestamp'] = datetime.now().isoformat()
        history.append(metrics)

        def _recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: _recursive_convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_recursive_convert(i) for i in obj]
            return _conv(obj)

        safe_history = _recursive_convert(history)

        with open(METRICS_HISTORY_FILE, 'w') as f:
            json.dump(safe_history, f, indent=2)
        print(f"✅ Metrics saved to history: {METRICS_HISTORY_FILE}")

    @staticmethod
    def load_history() -> List[Dict]:
        if not os.path.exists(METRICS_HISTORY_FILE):
            return []
        try:
            with open(METRICS_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []

class ModelMonitor:
    def __init__(self):
        self.fetcher = MLflowDataFetcher()
        self.performance_analyzer = PerformanceAnalyzer()
        self.drift_detector = DriftDetector()
        self.quality_checker = DataQualityChecker()
        self.retraining_trigger = RetrainingTrigger()
        self.dashboard_gen = DashboardGenerator()
        self.historical_tracker = HistoricalTracker()

    def run_monitoring(self, days_back: int = 7):
        print("=" * 70)
        print("🚀 Starting Enhanced Model Monitoring")
        print("=" * 70)
        df = self.fetcher.get_inference_runs(days_back)
        if df.empty:
            print("⚠️ No data found. Exiting.")
            return

        baseline_metrics = self.fetcher.get_baseline_metrics()

        print("\n📊 Analyzing Performance...")
        latency_stats = self.performance_analyzer.calculate_percentiles(df, 'metrics.latency_ms')
        throughput_stats = self.performance_analyzer.calculate_throughput(df)
        print(f"  Latency - P50: {latency_stats['p50']:.2f}ms, P95: {latency_stats['p95']:.2f}ms, P99: {latency_stats['p99']:.2f}ms")
        print(f"  Throughput: {throughput_stats['predictions_per_day']:.2f} predictions/day")

        print("\n🔍 Detecting Drift...")
        drift_results = {}
        history = self.historical_tracker.load_history()
        if history and len(history) > 0 and 'metrics.latency_ms' in df.columns:
            baseline_latency = np.array([h.get('performance', {}).get('mean', 0) for h in history[-10:]])
            current_latency = df['metrics.latency_ms'].dropna().values
            if len(baseline_latency) > 0 and len(current_latency) > 0:
                ks_result = self.drift_detector.ks_test(baseline_latency, current_latency)
                psi_score = self.drift_detector.calculate_psi(baseline_latency, current_latency)
                drift_results['latency_ks'] = ks_result
                drift_results['psi'] = psi_score
                print(f"  KS Test: p-value={ks_result['pvalue']:.4f}, Drift={ks_result['drift_detected']}")
                print(f"  PSI Score: {psi_score:.4f} {'⚠️ DRIFT DETECTED' if psi_score > THRESHOLDS['drift_psi'] else '✅ No drift'}")
        else:
            print("  ℹ️ Insufficient historical data for drift detection")

        print("\n🔬 Checking Data Quality...")
        quality_results = self.quality_checker.check_quality(df)
        for check, result in quality_results.items():
            print(f"  {check}: {result}")

        print("\n🎯 Evaluating Retraining Triggers...")
        retraining_decision = self.retraining_trigger.should_retrain(drift_results, quality_results, latency_stats)
        if retraining_decision['retrain_recommended']:
            print(f"  🚨 RETRAINING RECOMMENDED (Severity: {retraining_decision['severity']})")
            for trigger in retraining_decision['triggers']:
                print(f"    - {trigger}")
        else:
            print("  ✅ No retraining needed at this time")

        print("\n📈 Generating Dashboard...")
        self.dashboard_gen.create_dashboard(df, latency_stats, drift_results, quality_results, retraining_decision)

        print("\n💾 Saving to History...")
        self.historical_tracker.save_metrics({
            "performance": latency_stats,
            "throughput": throughput_stats,
            "drift": drift_results,
            "quality": quality_results,
            "retraining": retraining_decision
        })

        print("\n" + "=" * 70)
        print("✅ Monitoring Complete!")
        print(f"📊 Dashboard: {DASHBOARD_OUTPUT}")
        print(f"💾 History: {METRICS_HISTORY_FILE}")
        print("=" * 70)

if __name__ == "__main__":
    if not setup_dagshub_mlflow():
        print("❌ Failed to connect to DagsHub MLflow. Check your environment variables.")
        exit(1)
    monitor = ModelMonitor()
    monitor.run_monitoring(days_back=7)
