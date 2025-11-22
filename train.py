import os
import torch
import psutil
from ultralytics import YOLO
from typing import Optional

# --- Configuration ---
# You MUST update these paths to your actual project structure before running the script.

DATA_YAML_PATH = r"E:\test_final_split_v2\data.yaml"
PROJECT_DIR = r"E:\python\depi\runs\v11s\runs/train"
MODEL_NAME = "yolo11s"
MODEL_WEIGHTS = "yolo11s.pt"

# --- Utility Functions ---

def get_hardware_config(default_batch: int = 8, default_imgsz: int = 640, default_freeze: int = 4):
    """
    Checks available hardware (VRAM/RAM) and suggests training configuration.
    """
    print("=" * 70)
    print(" Checking Environment...")
    print("=" * 70)

    vram_gb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" GPU Detected: {gpu_name}")
        print(f" VRAM: {vram_gb:.2f} GB")
        device = 0 # Use the first GPU
    else:
        print("CUDA not available. Training will run on CPU.")
        device = "cpu"
    
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f" System RAM: {ram_gb:.1f} GB")

    # Dynamic configuration logic
    if vram_gb <= 4:
        batch_size = default_batch
        imgsz = default_imgsz
        freeze_layers = default_freeze
    elif vram_gb <= 6:
        batch_size = 12
        imgsz = 640
        freeze_layers = 4
    else:
        batch_size = 16
        imgsz = 800
        freeze_layers = 2

    print(f"\nTraining Configuration:")
    print(f"   → Batch Size: {batch_size}")
    print(f"   → Image Size: {imgsz}")
    print(f"   → Freeze Layers: {freeze_layers}")
    print(f"   → Device: {device}")
    print("=" * 70)
    
    return batch_size, imgsz, freeze_layers, device

def train_model(
    data_yaml_path: str,
    model_weights: str,
    project_dir: str,
    name: str,
    epochs: int = 80,
    batch_size: Optional[int] = None,
    imgsz: Optional[int] = None,
    freeze_layers: Optional[int] = None,
    device: Optional[str] = None
):
    """
    Initializes and trains the YOLO model with specified parameters.
    """
    
    # Get dynamic configuration if not provided
    if batch_size is None or imgsz is None or freeze_layers is None or device is None:
        batch_size, imgsz, freeze_layers, device = get_hardware_config()
    
    print(f"\n Loading {model_weights} model...")
    model = YOLO(model_weights)

    print("\n Starting Fine-Tuned Training...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=2,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        patience=5,
        amp=True,
        cache=True,
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
        verbose=True
    )
    
    print("\n Training completed.")
    return results


if __name__ == "__main__":
    try:
        # Get configuration based on available hardware
        batch, size, freeze, dev = get_hardware_config()
        
        # Start training
        train_model(
            data_yaml_path=DATA_YAML_PATH,
            model_weights=MODEL_WEIGHTS,
            project_dir=PROJECT_DIR,
            name=MODEL_NAME,
            epochs=80,
            batch_size=batch,
            imgsz=size,
            freeze_layers=freeze,
            device=dev
        )
        
        print("\n--- Model training script finished. ---")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure all dependencies are installed and the paths in the configuration section are correct.")
