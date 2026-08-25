# Intelligent Network Traffic Classification

Multi-class classification of encrypted/VPN network traffic using flow-level features, built as coursework for CS301M.

## Overview

Classifies network traffic into 14 categories (BROWSING, CHAT, FT, MAIL, P2P, STREAMING, VOIP, and their VPN-tunneled counterparts) using statistical features derived from packet flow data — inter-arrival times, byte/packet rates, active/idle durations.

## Dataset

~60,000 labeled flow records with 23 numeric features per flow (duration, forward/backward inter-arrival time stats, flow packet/byte rates, active/idle time stats).

## Approach

- **Preprocessing** (`src/data_preprocessing.py`): strips malformed headers, replaces infinities with NaN and drops corrupted rows, identifies the target label column.
- **Model** (`src/model_training.py`): Random Forest classifier (100 estimators, balanced class weights) on standardized features, with an 80/20 stratified train-test split.
- **Evaluation**: confusion matrix and top-10 feature importance plots generated automatically.

## Results

- Strong diagonal separation across most classes; the model reliably distinguishes VOIP, P2P, and mail traffic.
- Main confusion occurs between VPN-tunneled and non-VPN variants of the same application (e.g., BROWSING ↔ VPN-BROWSING, FT ↔ VPN-P2P), which makes sense since VPN encapsulation partially obscures but doesn't eliminate underlying flow timing patterns.
- Most important features: `flowBytesPerSecond`, inter-arrival time statistics (`min_flowiat`, `max_flowiat`, `mean_flowiat`), and flow duration — confirming that timing/rate features carry more signal than payload-derived stats for this task.

![Confusion Matrix](results/confusion_matrix.png)
![Feature Importance](results/feature_importance.png)

## Usage

```bash
pip install -r requirements.txt
python src/model_training.py
```

Run from the repo root — the script looks for a CSV inside `data/`.

## Tech Stack

Python, scikit-learn, pandas, numpy, matplotlib, seaborn
