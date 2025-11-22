# Egypt-Aware Object Detection for Autonomous Driving

## 1. Project Overview

This project focuses on developing a robust object detection model specifically tailored for the unique and often chaotic road environments of Egypt. By enhancing the BDD100K dataset with local Egyptian road images, we fine-tuned a YOLOv11s model to reliably detect objects, including local anomalies like tuk-tuks and visual noise such as banners, which are critical for autonomous driving and traffic monitoring systems in the region.

## 2. Team Information

| Name | ID |
| :--- | :--- |
| Mohamed Ragab Abdelhamid | 21000437 |
| Mohsen Ibrahim Hasan | 21112997 |
| Fatima Waleed Mostafa | 21072476 |
| Ebram Magdy Adolf Ibrahim | 21000974 |
| Ramy Mohsen Abdelmoneim | 21058422 |
| Ahmed Mohamed Abdelmoneim | 21072750 |

## 3. Problem Definition

Most existing autonomous driving datasets fail to capture the **chaotic and unique nature of Egyptian roads**. This includes:
*   The unexpected presence of **tuk-tuks**.
*   **Pedestrians** crossing randomly.
*   Frequent **lane violations**.
*   **Visual noise** such as banners and non-standard signage.

This lack of representation leads to poor performance and reliability of off-the-shelf object detection models when deployed in Egypt.

## 4. Proposed Solution

We developed a custom **YOLOv11s** model trained on the BDD100K dataset, which was significantly enhanced with locally-captured Egyptian road images. This enhancement specifically targeted underrepresented classes and scenarios, including tuk-tuks, banners, and realistic lane-violation situations. The resulting model demonstrates strong performance and superior generalization in real Egyptian environments.

## 5. Dataset and Preprocessing

*   **Base Dataset**: BDD100K.
*   **Enhancement**: Extended with locally-captured Egyptian images to improve detection of local objects (e.g., tuk-tuks, banners).
*   **Preprocessing Steps**:
    *   Removal of unlabeled and duplicate images.
    *   Ensuring synchronization between images and labels.
    *   Application of class-specific bounding-box thresholding to remove tiny noisy boxes.
    *   Use of class-weighting to handle class imbalance.
    *   Application of only physically-realistic augmentation techniques.

## 6. Model Selection and Final Results

We evaluated several models (YOLOv8n, YOLOv8n-frozen, YOLOv11n, and YOLOv11s) on a sample subset. **YOLOv11s with a frozen backbone** was selected as it showed the highest accuracy and generalization capability.

| Metric | Value |
| :--- | :--- |
| Final Epoch | 80 |
| Final mAP50 | 0.87149 |
| Final mAP50-95 | 0.63542 |
| Final Precision | 0.84072 |
| Final Recall | 0.80044 |

The model was tested on real-time webcam feeds and recorded driving scenarios, achieving stable and reliable detection performance after fine-tuning with local images.

## 7. Project Structure and Usage

The repository is structured to separate data handling, training, and inference logic.

```
/
├── pre.py             # Data preprocessing and cleaning script.
├── train.py           # Model training script (YOLO).
├── predict.py         # Model inference/prediction script.
├── requirements.txt   # List of Python dependencies.
├── .gitignore         # Files and directories to exclude from Git.
├── data/              # Directory for raw and processed datasets (excluded from Git).
├── models/            # Directory for trained model weights (e.g., best.pt).
└── vis/               # Directory for visualizations and analysis.
```

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Scripts

**NOTE:** Before running, you must update the hardcoded absolute paths (e.g., `D:\...`, `E:\...`) in the configuration sections of `pre.py` and `train.py` to match your local data structure.

*   **Data Preprocessing:**
    ```bash
    python pre.py
    ```

*   **Model Training:**
    ```bash
    python train.py
    ```

*   **Model Prediction (Inference):**
    ```bash
    python predict.py --model models/best.pt --source data/test_images/
    ```

## 8. Deployment and Next Steps

The final model was deployed using **Gradio and Hugging Face Spaces** with API support.

**Future Work:**
1.  Integrate the model into larger autonomous driving or traffic monitoring systems.
2.  Expand the dataset with more diverse Egyptian road environments.
3.  Improve production readiness with lightweight MLOps for monitoring and retraining.

---
**Project Drive:** [Drive Link]
**GitHub Repository:** [Repo Link]
**Dataset:** BDD100K
