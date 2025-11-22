import os
import argparse
from ultralytics import YOLO

# --- Configuration ---
# NOTE: Update these paths to your actual project structure.
DEFAULT_MODEL_PATH = "models/best.pt" 
DEFAULT_SOURCE_PATH = "data/test_images/" # Path to a folder of images or a single image
DEFAULT_OUTPUT_PATH = "runs/predict/"

def run_inference(
    model_path: str,
    source_path: str,
    output_path: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7
):
    """
    Loads a trained YOLO model and runs inference on the specified source.
    
    Args:
        model_path (str): Path to the trained model weights ('runs/train/yolo11s/weights/best.pt').
        source_path (str): Path to the image, video, or directory for inference.
        output_path (str): Directory to save the prediction results.
        imgsz (int): Image size for inference.
        conf (float): Object confidence threshold.
        iou (float): Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
    """
    print("--- Starting Model Inference ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Please ensure you have trained the model or updated the DEFAULT_MODEL_PATH.")
        return

    try:
        # Load the trained model
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        
        # Run inference
        results = model.predict(
            source=source_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=True,
            project=os.path.dirname(output_path),
            name=os.path.basename(output_path.rstrip('/\\')),
            exist_ok=True,
            verbose=True
        )
        
        print(f"\n✅ Inference complete. Results saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        print("Please check your model path, source path, and ensure all dependencies are installed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO model inference.")
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_PATH, 
        help="Path to the trained model weights (e.g., runs/train/yolo11s/weights/best.pt)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default=DEFAULT_SOURCE_PATH, 
        help="Path to the image, video, or directory for inference"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_OUTPUT_PATH, 
        help="Directory to save the prediction results"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Image size for inference"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Object confidence threshold"
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.7, 
        help="IoU threshold for Non-Maximum Suppression (NMS)"
    )
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        source_path=args.source,
        output_path=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou
    )
