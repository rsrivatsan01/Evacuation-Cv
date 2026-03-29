# 🚀 10-Day GitHub Publishing Plan

This guide outlines a strategic, day-by-day roadmap for pushing your **AI Evacuation System** to GitHub. 

By releasing portions of the project incrementally over 10 days rather than a single massive dump, it demonstrates to recruiters and contributors that the project was built deliberately with structured, logical commits.

---

### 📅 Day 1: Project Skeleton & Architecture
**Goal:** Initialize the repository, document the vision, and set up shared configurations.
* **Files to Commit:**
  * `.gitignore` (ignore `venv/`, `__pycache__/`, `.env`)
  * `README.md` (Without the final screenshots yet)
  * `requirements.txt`
  * `config.py`
* **Commit Message:** `chore: initialize project architecture and base configuration`

### 📅 Day 2: Computer Vision Core
**Goal:** Introduce the object detection and tracking logic that parses real-time video.
* **Files to Commit:**
  * `src/detection.py`
  * `src/tracking.py`
  * `src/zones.py`
* **Commit Message:** `feat: implement YOLOv8 detection and ByteTrack multi-person tracking`

### 📅 Day 3: Feature Engineering Pipeline
**Goal:** Add the data extraction tools that analyze raw movement into actionable analytics.
* **Files to Commit:**
  * `src/features.py`
* **Commit Message:** `feat: build feature extractor for crowd density and velocity calculation`

### 📅 Day 4: Training Setup & Weights
**Goal:** Push the Jupyter Notebooks used for training, as well as the trained model weights.
* **Files to Commit:**
  * `models/lstm_scaler.pkl`
  * `models/lstm_congestion.pt`
  * `notebooks/colab_setup.ipynb`
  * `notebooks/01_yolov8_finetune.ipynb`
  * `notebooks/02_lstm_training.ipynb`
* **Commit Message:** `ai: add trained LSTM models and cloud training notebooks`

### 📅 Day 5: Congestion Engine
**Goal:** Integrate the LSTM Neural Network into the Python backend for live inference.
* **Files to Commit:**
  * `src/congestion.py`
  * `diag_lstm.py` (if applicable)
* **Commit Message:** `feat: intergrate PyTorch LSTM for real-time corridor congestion prediction`

### 📅 Day 6: Pathfinding Routing
**Goal:** Introduce the logic for assigning crowds to the safest unblocked exits.
* **Files to Commit:**
  * `src/pathplanning.py`
  * `data/venue_graph.json` 
  * `data/zones.json` (Default fallback settings)
* **Commit Message:** `feat: implement NetworkX Dijkstra algorithm for dynamic evacuation routing`

### 📅 Day 7: Pipeline Threading & Utilities
**Goal:** Add the master controller that runs the ML models safely in a background thread.
* **Files to Commit:**
  * `src/pipeline.py`
  * `utils/logger.py`
  * `utils/visualizer.py`
* **Commit Message:** `core: add background processing pipeline and visualizer utilities`

### 📅 Day 8: Flask Backend APIs
**Goal:** Set up the web server that routes the video streams and JSON status data to the frontend.
* **Files to Commit:**
  * `app.py`
* **Commit Message:** `feat: build Flask REST APIs and MJPEG video streaming endpoints`

### 📅 Day 9: Live Dashboard UI
**Goal:** Construct the real-time front-end dashboard that operators will view.
* **Files to Commit:**
  * `templates/dashboard.html`
  * `static/css/dashboard.css`
  * `static/js/dashboard.js`
* **Commit Message:** `ui: develop responsive web dashboard for live monitoring and alerts`

### 📅 Day 10: Analytics & Release v1.0
**Goal:** Polish the system by introducing historical analytics and releasing version 1.0!
* **Files to Commit:**
  * `templates/analytics.html`
  * `README.md` (Add final screenshots and badges!)
  * `data/maps/` (Sample default maps)
* **Commit Message:** `feat: launch interactive analytics page and release v1.0.0`
