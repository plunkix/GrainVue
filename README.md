# GrainVue  
### End-to-End Computer Vision System for Rice Seed Purity Testing

GrainVue is a **production-ready computer vision system** built to automate **Other Differential Variety (ODV)** testing for rice seeds.  
It replaces slow, expert-driven manual inspection with a **robust ML pipeline** that distinguishes **Shree101 rice seeds from 18+ visually similar varieties**, while remaining reliable under real-world, open-set conditions.

This is a ML project — it is a **shipped, full-stack CV system solving a real agricultural problem**.

---

##  Problem Statement

Seed certification labs rely on visual inspection to verify genetic purity.  
For **Shree101 rice seeds**, this task is extremely difficult because:

- Visual differences from other varieties are **subtle**
- Length-to-width ratios differ by **<5%**
- Color and texture cues are inconsistent
- **Unseen varieties** regularly appear

This is **not** a standard multi-class classification problem.

The real question is binary and open-set:

> **Is this seed Shree101 — or not?**

False negatives are costly, so the system must prioritize **decision reliability**, not just accuracy.

---

## Solution Overview

GrainVue uses a **three-stage computer vision pipeline** designed for robustness, speed, and safe decision-making.

### 1️ Seed Detection  
- **Model:** Faster R-CNN (ResNet50 backbone)  
- Detects and isolates individual seeds from raw images  
- Filters overlapping, low-quality, and noisy samples  

### 2️ Metric Learning Filter  
- **Model:** EfficientNet-B3 with ArcFace loss  
- Learns an embedding space where Shree101 forms a tight cluster  
- Rejects **60–70%** of non-target varieties early using distance thresholds  

### 3️ Fine-Grained Classification  
- **Dual-backbone CNN:** EfficientNet-B3 + ResNet50  
- Applied only to borderline cases  
- Produces final classification with uncertainty handling  

This staged approach balances **accuracy, inference cost, and risk control**.

---

##  Results

- **Final accuracy:** 98.14%  
- **False negatives:** 1.1% (within certification tolerance)  
- **Uncertain cases:** ~2.3% (flagged for manual review)  
- **Inference speed:**  
  - 2–3s per seed (classification only)  
  - 5–10s per image (full detection pipeline)

The system generalizes well to **unseen rice varieties**, which is critical for real-world deployment.

---

##  Why This Works

Key design decisions:

- Binary framing instead of brittle multi-class prediction  
- Metric learning to handle open-set conditions  
- Early rejection to reduce compute cost  
- Quality filtering to minimize false positives  
- Explicit uncertainty handling for safety  

The goal was **reliable decisions**, not leaderboard metrics.

---

##  Tech Stack

**Machine Learning / CV**
- PyTorch
- Faster R-CNN
- EfficientNet, ResNet50
- ArcFace (metric learning)

**Backend**
- FastAPI
- Modular inference pipeline
- SQLite for result tracking

**Frontend / Interfaces**
- Gradio (web UI)
- PyQt6 (desktop prototype)
- Electron + React (production desktop app)

---

##  Deployment & Interfaces

GrainVue was deployed across multiple user-facing platforms:

- Web-based classifier
- Full detection + classification pipeline
- Desktop application with batch processing
- API-first backend for reuse across platforms

Designed for **non-technical users** in seed quality laboratories.

---
## Demo Video

A short walkthrough demonstrating:
- Seed detection
- Classification output
- Uncertainty handling
- End-to-end application flow

><>< Demo video link: <PASTE LINK HERE>




