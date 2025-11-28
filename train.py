import os
import torch
import psutil
from ultralytics import YOLO
from typing import Optional, Dict, Any
import mlflow
import json
from datetime import datetime
import shutil

from ultralytics import settings
settings.update({'mlflow': False})
settings.reset()

# --- Configuration ---
# You MUST update these paths to your actual project structure before running the script.

DATA_YAML_PATH = r"H:\Startups\Autonomus Car Detection DEPI\data.yaml"
PROJECT_DIR = r"H:\Startups\Autonomus Car Detection DEPI"
MODEL_NAME = "tuned_yolov11s.pt"
MODEL_WEIGHTS = r"H:\Startups\Autonomus Car Detection DEPI\best68.pt"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "YOLOv11s_Autonomous_Driving_OD"

# --- Utility Functions ---

def get_hardware_config(default_batch: int = 8, default_imgsz: int = 640, default_freeze: int = 4):
    """
    Checks available hardware (VRAM/RAM) and suggests training configuration.
    """
    print("=" * 70)
    print(" Checking Environment...")
    print("=" * 70)

    vram_gb = 0
    device = "cpu"
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" ✅ GPU Detected: {gpu_name}")
        print(f" ✅ VRAM: {vram_gb:.2f} GB")
        print(f" ✅ CUDA Version: {torch.version.cuda}")
        print(f" ✅ cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Verify GPU is actually working
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            device = 0  # Use the first GPU
            print(f" ✅ GPU verification: PASSED")
        except Exception as e:
            print(f" ⚠️ GPU verification failed: {e}")
            print(f" ⚠️ Falling back to CPU")
            device = "cpu"
    else:
        print(" ❌ CUDA not available. Training will run on CPU.")
        print("    Install CUDA-enabled PyTorch for GPU acceleration:")
        print("    https://pytorch.org/get-started/locally/")
        device = "cpu"
    
    ram_gb = psutil.virtual_memory().total / 1024**3
    available_ram_gb = psutil.virtual_memory().available / 1024**3
    print(f" System RAM: {ram_gb:.1f} GB (Available: {available_ram_gb:.1f} GB)")

    # Dynamic configuration logic - optimized for speed
    if device == "cpu":
        # CPU training - use smaller batch
        batch_size = 4
        imgsz = 640
        freeze_layers = 10
        use_cache = False
        workers = 0  # Windows multiprocessing issues
    elif vram_gb <= 4:
        batch_size = default_batch
        imgsz = default_imgsz
        freeze_layers = default_freeze
        use_cache = available_ram_gb > 8  # Only cache if enough RAM
        workers = 4 if os.name != 'nt' else 0  # Windows: use 0, Linux: use 4
    elif vram_gb <= 6:
        batch_size = 12
        imgsz = 640
        freeze_layers = 4
        use_cache = available_ram_gb > 10
        workers = 6 if os.name != 'nt' else 0
    else:
        batch_size = 16
        imgsz = 800
        freeze_layers = 2
        use_cache = available_ram_gb > 12
        workers = 8 if os.name != 'nt' else 0

    print(f"\nTraining Configuration:")
    print(f"   → Batch Size: {batch_size}")
    print(f"   → Image Size: {imgsz}")
    print(f"   → Freeze Layers: {freeze_layers}")
    print(f"   → Device: {device} {'(GPU)' if device != 'cpu' else '(CPU - SLOW!)'}")
    print(f"   → Workers: {workers}")
    print(f"   → Cache Images: {use_cache}")
    if device == "cpu":
        print(f"\n ⚠️ WARNING: Training on CPU will be VERY SLOW!")
        print(f"    Expected time: 5-10+ hours")
        print(f"    Consider using GPU for faster training")
    print("=" * 70)
    
    return batch_size, imgsz, freeze_layers, device, workers, use_cache

def log_all_metrics(results, mlflow_run):
    """
    Logs all available metrics from YOLO training results to MLflow.
    """
    try:
        # Extract all metrics from results
        metrics_dict = results.metrics if hasattr(results, 'metrics') else {}
        
        # Log all available metrics
        metric_keys = [
            'metrics/mAP50', 'metrics/mAP50-95', 'metrics/mAP@0.5', 'metrics/mAP@0.5:0.95',
            'metrics/precision', 'metrics/recall',
            'metrics/box_loss', 'metrics/cls_loss', 'metrics/dfl_loss',
            'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
        ]
        
        logged_metrics = {}
        for key in metric_keys:
            # Try different key variations
            for variant in [key, key.replace('/', '_'), key.replace('metrics/', '')]:
                if variant in metrics_dict:
                    value = metrics_dict[variant]
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(variant, value)
                        logged_metrics[variant] = value
                        break
        
        # Also try to get metrics from results object directly
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)) and 'loss' in key.lower() or 'map' in key.lower():
                    mlflow.log_metric(key, value)
                    logged_metrics[key] = value
        
        print(f"✅ Logged {len(logged_metrics)} metrics to MLflow")
        return logged_metrics
        
    except Exception as e:
        print(f"⚠️ Warning: Could not log all metrics: {e}")
        return {}

def log_training_artifacts(results, mlflow_run):
    """
    Logs training artifacts (plots, configs, models) to MLflow.
    """
    artifacts_logged = []
    
    try:
        save_dir = results.save_dir if hasattr(results, 'save_dir') else None
        
        if save_dir and os.path.exists(save_dir):
            # Log best model
            best_model_path = os.path.join(save_dir, "weights", "best.pt")
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path, "model_weights")
                artifacts_logged.append("best.pt")
                print("✅ Logged best.pt model")
            
            # Log last model
            last_model_path = os.path.join(save_dir, "weights", "last.pt")
            if os.path.exists(last_model_path):
                mlflow.log_artifact(last_model_path, "model_weights")
                artifacts_logged.append("last.pt")
            
            # Log training plots
            plots_dir = os.path.join(save_dir, "plots")
            if os.path.exists(plots_dir):
                mlflow.log_artifacts(plots_dir, "training_plots")
                artifacts_logged.append("training_plots")
                print("✅ Logged training plots")
            
            # Log confusion matrix
            confusion_matrix = os.path.join(plots_dir, "confusion_matrix.png")
            if os.path.exists(confusion_matrix):
                mlflow.log_artifact(confusion_matrix, "plots")
                artifacts_logged.append("confusion_matrix.png")
            
            # Log results CSV if exists
            results_csv = os.path.join(save_dir, "results.csv")
            if os.path.exists(results_csv):
                mlflow.log_artifact(results_csv, "metrics")
                artifacts_logged.append("results.csv")
            
            # Log args.yaml (training configuration)
            args_yaml = os.path.join(save_dir, "args.yaml")
            if os.path.exists(args_yaml):
                mlflow.log_artifact(args_yaml, "config")
                artifacts_logged.append("args.yaml")
        
        print(f"✅ Logged {len(artifacts_logged)} artifact groups to MLflow")
        return artifacts_logged
        
    except Exception as e:
        print(f"⚠️ Warning: Could not log all artifacts: {e}")
        return artifacts_logged

def train_model(
    data_yaml_path: str,
    model_weights: str,
    project_dir: str,
    name: str,
    epochs: int = 1,
    batch_size: Optional[int] = None,
    imgsz: Optional[int] = None,
    freeze_layers: Optional[int] = None,
    device: Optional[str] = None,
    workers: Optional[int] = None,
    use_cache: Optional[bool] = None
):
    """
    Initializes and trains the YOLO model with specified parameters.
    """
    
    # Get dynamic configuration if not provided
    if batch_size is None or imgsz is None or freeze_layers is None or device is None:
        batch_size, imgsz, freeze_layers, device, workers, use_cache = get_hardware_config()
    
    # Force GPU if available
    if device != "cpu" and torch.cuda.is_available():
        # Ensure device is integer for GPU
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        elif device == "cpu":
            device = 0  # Force GPU
        # Verify GPU one more time
        try:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            print(f"\n ✅ GPU {device} is ready for training")
        except Exception as e:
            print(f"\n ⚠️ GPU setup failed: {e}")
            print(f"    Falling back to CPU (will be slow)")
            device = "cpu"
    
    print(f"\n Loading {model_weights} model...")
    model = YOLO(model_weights)
    
    # Move model to GPU explicitly if available
    if device != "cpu" and torch.cuda.is_available():
        try:
            # YOLO handles device internally, but we can verify
            print(f" ✅ Model will use GPU: {torch.cuda.get_device_name(device)}")
        except:
            pass

    print(f"\n Starting Fine-Tuned Training...")
    print(f" Device: {device}, Workers: {workers}, Cache: {use_cache}")
    
    # Optimize training parameters for speed
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,  # Explicitly set device
        workers=workers if workers is not None else (0 if os.name == 'nt' else 4),  # Windows: 0, Linux: 4
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        patience=5,
        amp=True if device != "cpu" else False,  # AMP only on GPU
        cache=use_cache if use_cache is not None else False,  # Cache based on RAM
        freeze=freeze_layers,
        dropout=0.05,
        box=7.0, cls=1.0, dfl=1.5,
        project=project_dir,
        name=name,
        exist_ok=False,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        seed=42,
        verbose=True,
        # Additional speed optimizations
        close_mosaic=10,  # Disable mosaic in last 10 epochs for faster training
        resume=False,  # Don't resume from checkpoint
        fraction=1.0,  # Use full dataset
    )
    
    print("\n Training completed.")
    return results


if __name__ == "__main__":
    
    # === 1. Setup MLflow Tracking ===
    print("\n" + "=" * 70)
    print(" MLflow Configuration")
    print("=" * 70)
    
    # Set tracking URI (defaults to local if not set)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f" MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            print(f" ✅ Created new experiment: {EXPERIMENT_NAME}")
        else:
            experiment_id = experiment.experiment_id
            print(f" ✅ Using existing experiment: {EXPERIMENT_NAME}")
    except Exception as e:
        print(f" ⚠️ Warning: {e}")
        experiment_id = None
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # =================================
    
    try:
        # Get hardware configuration
        batch, size, freeze, dev, workers, use_cache = get_hardware_config()
        
        # Create run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"YOLOv11s_Train_Batch_{batch}_Size_{size}_{timestamp}"
        
        # === 2. Start MLflow Run ===
        with mlflow.start_run(run_name=run_name) as run:
            
            print(f"\n MLflow Run ID: {run.info.run_id}")
            print(f" Run Name: {run_name}")
            
            # === 3. Log Parameters ===
            print("\n Logging parameters...")
            
            # Training parameters
            mlflow.log_param("epochs", 1)
            mlflow.log_param("optimizer", "AdamW")
            mlflow.log_param("lr0", 0.0008)
            mlflow.log_param("lrf", 0.01)
            mlflow.log_param("batch_size", batch)
            mlflow.log_param("img_size", size)
            mlflow.log_param("freeze_layers", freeze)
            mlflow.log_param("patience", 5)
            mlflow.log_param("dropout", 0.05)
            mlflow.log_param("box_loss_weight", 7.0)
            mlflow.log_param("cls_loss_weight", 1.0)
            mlflow.log_param("dfl_loss_weight", 1.5)
            mlflow.log_param("amp", True)
            mlflow.log_param("seed", 42)
            
            # Hardware info
            if torch.cuda.is_available():
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
                mlflow.log_param("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
            mlflow.log_param("ram_gb", round(psutil.virtual_memory().total / 1024**3, 1))
            mlflow.log_param("device", str(dev))
            
            # Data paths
            mlflow.log_param("data_yaml_path", DATA_YAML_PATH)
            mlflow.log_param("model_weights", MODEL_WEIGHTS)
            
            # === 4. Log Tags ===
            mlflow.set_tag("model_type", "YOLOv11s")
            mlflow.set_tag("task", "object_detection")
            mlflow.set_tag("domain", "autonomous_driving")
            mlflow.set_tag("framework", "ultralytics")
            mlflow.set_tag("training_type", "fine_tuning")
            
            # === 5. Run Training ===
            print("\n Starting Fine-Tuned Training...")
            training_start_time = datetime.now()
            
            results = train_model(
                data_yaml_path=DATA_YAML_PATH,
                model_weights=MODEL_WEIGHTS,
                project_dir=PROJECT_DIR,
                name=MODEL_NAME,
                epochs=1,
                batch_size=batch,
                imgsz=size,
                freeze_layers=freeze,
                device=dev,
                workers=workers,
                use_cache=use_cache
            )
            
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            mlflow.log_metric("training_duration_seconds", training_duration)
            
            # === 6. Log Metrics ===
            print("\n Logging metrics...")
            logged_metrics = log_all_metrics(results, run)
            
            # === 7. Log Artifacts ===
            print("\n Logging artifacts...")
            logged_artifacts = log_training_artifacts(results, run)
            
            # === 8. Log Model as MLflow Model (optional) ===
            try:
                best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
                if os.path.exists(best_model_path):
                    # Log model with metadata
                    mlflow.log_artifact(best_model_path, "models")
                    
                    # Register model (optional - uncomment if you want to use model registry)
                    # mlflow.register_model(f"runs:/{run.info.run_id}/models/best.pt", "YOLOv11s_Autonomous_Driving")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not register model: {e}")
            
            # === 9. Summary ===
            print("\n" + "=" * 70)
            print(" MLflow Tracking Summary")
            print("=" * 70)
            print(f" ✅ Run ID: {run.info.run_id}")
            print(f" ✅ Experiment: {EXPERIMENT_NAME}")
            print(f" ✅ Metrics logged: {len(logged_metrics)}")
            print(f" ✅ Artifacts logged: {len(logged_artifacts)}")
            print(f" ✅ Training duration: {training_duration:.2f} seconds")
            print(f"\n View results at: {mlflow.get_tracking_uri()}")
            print("=" * 70)
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error to MLflow if run is active
        try:
            mlflow.log_param("error", str(e))
            mlflow.set_tag("status", "failed")
        except:
            pass
