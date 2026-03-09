# CNN Image Classifier with Grad-CAM

Custom CNN for CIFAR-10 classification with Grad-CAM explainability visualization.

## Architecture
- 3 conv blocks with BatchNorm, GELU, MaxPool, Dropout
- Global Average Pooling
- MLP classifier head
- Grad-CAM hooks on final conv layer

## Features
- Data augmentation: crop, flip, rotation, color jitter
- AdamW optimizer + cosine annealing
- Label smoothing
- Grad-CAM heatmaps to visualize what the CNN looks at

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `training_history.png` — accuracy curves
- `gradcam.png` — original/heatmap/overlay for 6 test images
- `best_cnn_model.pth` — saved weights
