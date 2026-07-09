# TACO-InstanceSegmentation

Instance segmentation on the **TACO (Trash Annotations in Context)** dataset, comparing **YOLOv8** and **Mask R-CNN** — including a custom class-balanced subset, hyperparameter tuning, and a full performance comparison against previously published results.

> Course project for *Image and Video Understanding* — a two-person collaboration by **Soroush Karami** and **Saeed Khosravi**.

## Problem

TACO is a real-world litter dataset (~1,500 images, ~4,800 annotations, 60 classes) captured in streets, parks, beaches, and forests. It is a genuinely hard segmentation benchmark because of severe class imbalance, small objects, high intra-class variability, occlusions, and visual clutter. Published instance-segmentation results on the full dataset are correspondingly low (Mask R-CNN ≈ 12.7–16.3%, DetectoRS ≈ 16.7% mAP@50), which sets realistic expectations for this task.

## Dataset Preparation

The original class distribution is highly long-tailed — a few classes (plastic bottle, drink can, aluminium foil) have 200+ instances while many have fewer than 10. To make training and evaluation fairer, we built a **custom subset of the 20 most frequent classes**:

- 1,350 images
- Split 70% train / 20% validation / 10% test

## Models & Configuration

Two instance-segmentation approaches were compared. Both used 640×640 or 960×960 images and batch size 16. (Mask2Former and DETR were also explored but not fully tuned.)

**YOLOv8** (`yolov8l-seg.pt`)
- Automated hyperparameter tuning via Ultralytics over a defined search space (learning rate, momentum, weight decay, HSV color augmentation, scale, box-loss gain)
- 50 tuning iterations × 20 epochs each, then final training for 200 epochs

**Mask R-CNN** (`mask_rcnn_R_101_FPN_3x`, Detectron2)
- Tuned learning rate, NMS threshold, and gamma (LR-decay)
- Best config: LR = 0.001, gamma = 0.1, NMS = 0.6; trained for 6,000 iterations

## Results

**Final comparison (best configuration of each model):**

| Metric | YOLOv8 (Box) | Mask R-CNN (Box) | YOLOv8 (Seg) | Mask R-CNN (Seg) |
|---|---|---|---|---|
| Precision | **32.1%** | 27.2% | **30.1%** | 25.0% |
| Recall | 27.2% | 25.0% | 28.9% | 25.0% |
| mAP@50 | **26.5%** | 23.9% | **25.4%** | 23.7% |

**Key findings:**
- **YOLOv8 outperformed Mask R-CNN** across precision, recall, and mAP@50 on both bounding-box and segmentation tasks.
- YOLOv8's segmentation mAP@50 of 25.4% is competitive with — and in places exceeds — previously published Mask R-CNN / DetectoRS results on TACO instance segmentation.
- A notable practical finding: on YOLOv8, the *default* configuration at image size 960 actually beat the *tuned* configuration on recall and mAP@50 — but tuning was constrained to image size 640 due to GPU memory limits (an A100 request was redirected to an L4), so the comparison isn't fully like-for-like. This is documented honestly in the report rather than glossed over.

## Tech Stack

- Python, PyTorch
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — training, tuning, segmentation
- [Detectron2](https://github.com/facebookresearch/detectron2) — Mask R-CNN
- NumPy, Matplotlib — data handling and visualization

## Repository Contents

- Training / evaluation code for both models
- Full project report (PDF) with methodology, tuning tables, confusion matrices, and qualitative comparisons
- Presentation slides

---

*Computer-vision project by Soroush Karami (with Saeed Khosravi), MSc Computer Science (AI & Data Engineering), Ca' Foscari University of Venice.*
