# GrainVue System Architecture

Comprehensive technical documentation of the three-stage classification pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Stage 1: Seed Detection](#stage-1-seed-detection)
3. [Stage 2: Metric Learning Filter](#stage-2-metric-learning-filter)
4. [Stage 3: Fine-Grained Classification](#stage-3-fine-grained-classification)
5. [Quality Filters](#quality-filters)
6. [Model Architectures](#model-architectures)
7. [Data Flow](#data-flow)
8. [Performance Optimization](#performance-optimization)

---

## Overview

GrainVue uses a **three-stage cascade pipeline** to achieve 98.14% accuracy:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 1: DETECTION                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Faster R-CNN (ResNet50 backbone)                          │  │
│  │ • Detects individual seeds                                │  │
│  │ • Confidence threshold: 0.5                               │  │
│  │ • NMS threshold: 0.5                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                         │                                         │
│                         ▼                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Quality Filters (5 filters)                               │  │
│  │ • Blur detection (Laplacian variance)                     │  │
│  │ • Multi-seed detection (contour analysis)                 │  │
│  │ • Elongation check (aspect ratio)                         │  │
│  │ • Valley split detection (Sobel edges)                    │  │
│  │ • Watershed analysis                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                         │                                         │
│                    Individual crops                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: METRIC LEARNING FILTER                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ EfficientNet-B3 + ArcFace Loss                            │  │
│  │ • 512-dimensional embeddings                              │  │
│  │ • Distance to Shree101 prototype                          │  │
│  │ • Threshold: 0.76                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                         │                                         │
│              ┌──────────┴──────────┐                              │
│              ▼                     ▼                              │
│      Distance ≥ 0.76         Distance < 0.76                     │
│      [REJECT: Other]         [Pass to Stage 3]                   │
│      (60-70% rejected)       (30-40% suspicious)                 │
└────────────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 3: FINE-GRAINED CLASSIFICATION                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Dual Backbone CNN                                         │  │
│  │ • EfficientNet-B3 + ResNet50                              │  │
│  │ • Feature concatenation → 512-dim                         │  │
│  │ • Binary classifier (Shree101 vs Others)                  │  │
│  │ • Weighted BCE loss (2x penalty for FN)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                         │                                         │
│              ┌──────────┼──────────┐                              │
│              ▼          ▼          ▼                              │
│       P(good) > 0.8  0.4-0.8   P(good) < 0.4                     │
│       [SHREE101]   [UNCERTAIN]  [OTHER]                          │
│         (pass)    (manual review) (reject)                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
1. **Early rejection** - Filter obvious "Others" in Stage 2 (saves computation)
2. **Uncertainty handling** - Flag ambiguous cases for human review
3. **Cascade efficiency** - Only 30-40% of seeds reach Stage 3
4. **Quality first** - Multiple filters ensure clean inputs

---

## Stage 1: Seed Detection

### Model: Faster R-CNN

**Architecture:**
- Backbone: ResNet50 (pre-trained on ImageNet)
- RPN (Region Proposal Network): 9 anchor boxes
- ROI Pooling: 7x7 feature maps
- Classification head: 2 classes (seed, background)
- Regression head: Bounding box refinement

**Training Details:**
```python
Optimizer: SGD (momentum=0.9, weight_decay=0.0005)
Learning Rate: 0.005 (step decay)
Batch Size: 4
Epochs: 50
Augmentation: Rotation (±180°), flip, brightness/contrast
Dataset: 300+ annotated images
```

**Inference Parameters:**
```yaml
confidence_threshold: 0.5
nms_threshold: 0.5
max_detections_per_image: 50
```

**Performance:**
- Precision: ~95%
- Recall: ~90%
- Speed: 1-2 seconds per image (GPU)

### Quality Filters

Applied after detection to ensure crop quality:

#### 1. Blur Detection
```python
Method: Laplacian variance
Threshold: 0.5
Rejects: Out-of-focus crops
```

#### 2. Multi-Seed Detection
```python
Method: Contour analysis
Threshold: >1 significant contours
Rejects: Overlapping seeds
```

#### 3. Elongation Check
```python
Method: Aspect ratio
Threshold: >4.2
Rejects: Heavily overlapping seeds (elongated bounding boxes)
```

#### 4. Valley Split Detection
```python
Method: Sobel edge detection
Threshold: 0.25 (edge strength)
Action: Detects touching seeds, attempts split
```

#### 5. Watershed Analysis
```python
Method: Distance transform + watershed
Threshold: 0.9
Action: Separates touching seeds
```

**Combined Impact:** 60% reduction in false positives

---

## Stage 2: Metric Learning Filter

### Purpose
Create an embedding space where Shree101 forms a tight cluster, enabling fast rejection of obvious "Others".

### Model: EfficientNet-B3 + ArcFace

**Architecture:**
```python
Input: 224x224x3 RGB image
    ↓
EfficientNet-B3 backbone (pretrained)
    ↓
Global Average Pooling
    ↓
512-dimensional embedding
    ↓
L2 Normalization
    ↓
ArcFace Loss (margin=0.5, scale=30)
```

**Loss Function: ArcFace**
```
L = -log(exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_j≠yi exp(s·cos(θ_j))))

Where:
- s = 30 (scale)
- m = 0.5 (angular margin)
- θ_yi = angle between feature and ground truth class center
```

**Training:**
- Dataset: 14 varieties + Shree101 (all training data)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Batch Size: 64
- Epochs: 100 (early stopping)
- Augmentation: Minimal (rotation, flip only)

**Inference:**
```python
# Compute Shree101 prototype (mean embedding of all Shree101 training samples)
prototype = mean(all_shree101_embeddings)  # 512-dim vector

# For each test image
embedding = model(image)  # 512-dim
distance = cosine_distance(embedding, prototype)

if distance < 0.76:
    pass_to_stage_3()
else:
    reject_as_other()
```

**Performance:**
- ROC AUC: 0.95+
- Optimal threshold: 0.76
- Rejection rate: 60-70% of "Others"
- Speed: 50ms per seed (GPU)

**t-SNE Visualization:**
```
    Shree101 ●●●●●●
              ●●●●●   (tight cluster)
              ●●●●
                      
  Others      ○  ○
           ○     ○    (scattered)
         ○   ○      ○
```

---

## Stage 3: Fine-Grained Classification

### Model: Dual Backbone CNN

**Architecture:**
```python
Input: 224x224x3 RGB image
         │
    ┌────┴────┐
    ▼         ▼
EfficientNet-B3  ResNet50
(pretrained)     (pretrained)
    │              │
Feature Map    Feature Map
(1536-dim)     (2048-dim)
    │              │
    └────┬─────────┘
         │
    Concatenate (3584-dim)
         │
         ▼
    ┌─────────────┐
    │  Classifier  │
    ├─────────────┤
    │ Dropout(0.5) │
    │ Linear(512)  │
    │ BatchNorm1d  │
    │ ReLU         │
    │ Dropout(0.5) │
    │ Linear(256)  │
    │ BatchNorm1d  │
    │ ReLU         │
    │ Dropout(0.4) │
    │ Linear(128)  │
    │ BatchNorm1d  │
    │ ReLU         │
    │ Dropout(0.3) │
    │ Linear(2)    │
    │ Sigmoid      │
    └─────────────┘
         │
         ▼
   P(Shree101), P(Other)
```

**Training:**
```python
Dataset: 7 carefully curated varieties
  - Shree101: 3,514 samples
  - Others: RNR, JSR, Elito, Kaveri7155, Chintu, ARV-511, YSR

Loss: Weighted Binary Cross-Entropy
  - Class 0 (Shree101): weight = 2.0 (penalize false negatives)
  - Class 1 (Others): weight = 1.0

Optimizer: AdamW
  - Learning rate: 5e-5
  - Weight decay: 5e-4
  
Training strategy:
  - Minimal augmentation (preserve fine details)
  - Moderate dropout (0.3-0.5)
  - Large validation set (20%)
  - Early stopping (patience=8)
  
Batch Size: 32
Epochs: 50 (stopped at ~30 due to early stopping)
```

**Decision Thresholds:**
```python
if P(Shree101) > 0.8:
    return "SHREE101"  # High confidence pass
elif P(Shree101) < 0.4:
    return "OTHER"  # High confidence reject
else:
    return "UNCERTAIN"  # Manual review needed
```

**Performance:**
- Validation Accuracy: 98.76%
- Test Accuracy: 98.14%
- Uncertain rate: 2.3%
- False positives: 1.1%
- False negatives: 1.1%

---

## Model Architectures

### Model Files

| Model | Size | Architecture | Purpose |
|-------|------|--------------|---------|
| `seed_frcnn_2.pth` | 165 MB | Faster R-CNN + ResNet50 | Detection |
| `final_metric_model.pth` | 50 MB | EfficientNet-B3 + ArcFace | Metric filter |
| `improved_fine_grained_model.pth` | 436 MB | EfficientNet-B3 + ResNet50 | Classification |

### Total Pipeline Size: **651 MB**

---

## Data Flow

### Complete Pipeline Code Flow

```python
def classify_image(image_path):
    # Stage 1: Detection
    image = load_image(image_path)
    detections = detection_model(image)
    
    crops = []
    for bbox in detections:
        crop = extract_crop(image, bbox)
        
        # Quality filters
        if is_blurry(crop):
            continue
        if has_multiple_seeds(crop):
            split_crops = attempt_split(crop)
            crops.extend(split_crops)
        else:
            crops.append(crop)
    
    results = []
    for crop in crops:
        # Stage 2: Metric Learning
        embedding = metric_model(crop)
        distance = cosine_distance(embedding, shree101_prototype)
        
        if distance >= 0.76:
            results.append({"label": "OTHER", "stage": 2})
            continue
        
        # Stage 3: Fine-Grained
        logits = fine_grained_model(crop)
        prob_shree101 = sigmoid(logits[0])
        
        if prob_shree101 > 0.8:
            label = "SHREE101"
        elif prob_shree101 < 0.4:
            label = "OTHER"
        else:
            label = "UNCERTAIN"
        
        results.append({
            "label": label,
            "confidence": float(prob_shree101),
            "stage": 3
        })
    
    return results
```

---

## Performance Optimization

### Speed Optimizations

1. **Batch Processing**
   - Stage 2: Process 32 crops simultaneously
   - Stage 3: Process 16 crops simultaneously
   
2. **Early Rejection**
   - 60-70% rejected in Stage 2 (faster metric learning)
   - Only 30-40% reach Stage 3 (slower dual backbone)

3. **Model Optimization**
   - TorchScript compilation (20% faster)
   - Mixed precision (FP16 inference)
   - ONNX export support

4. **GPU Utilization**
   - Multi-stream processing
   - Asynchronous data loading
   - Pinned memory

### Memory Optimizations

1. **Lazy Loading**
   - Load models only when needed
   - Unload unused models

2. **Crop Caching**
   - Cache quality-filtered crops
   - Avoid reprocessing

3. **Efficient Data Types**
   - FP16 where possible
   - Uint8 for images

### Benchmark (Tesla T4 GPU)

| Stage | Time | Throughput |
|-------|------|------------|
| Detection (per image) | 1.5s | 0.67 img/s |
| Metric Filter (per crop) | 50ms | 20 crops/s |
| Fine-Grained (per crop) | 150ms | 6.7 crops/s |
| **Total Pipeline** | **2-3s/image** | **~0.4 img/s** |

---

## Configuration

See `config.yaml` for all configurable parameters:

```yaml
detection:
  model_path: "models/seed_frcnn_2.pth"
  confidence_threshold: 0.5
  nms_threshold: 0.5

metric_learning:
  model_path: "models/final_metric_model.pth"
  distance_threshold: 0.76

fine_grained:
  model_path: "models/improved_fine_grained_model.pth"
  pass_threshold: 0.8
  reject_threshold: 0.4

quality_filters:
  blur_threshold: 0.5
  aspect_ratio_max: 4.2
  multi_seed_enabled: true
```

---

## Future Improvements

1. **Model Compression**
   - Quantization (INT8)
   - Knowledge distillation
   - Pruning

2. **Architecture**
   - Replace dual backbone with single efficient model
   - Attention mechanisms
   - Vision Transformers

3. **Pipeline**
   - End-to-end training
   - Differentiable quality filters
   - Active learning for uncertain samples

---

**Last Updated:** December 2025  
**Author:** Srushti Tathe
