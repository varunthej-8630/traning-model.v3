# Custos AI Model Builder (v3.0)

**Custom Dataset & Model Builder for the CUSTOS Guardian Intelligence System**  
*Developed by Varunthej Parimi*

---

## 🚀 What's New in v3.0

Version 3.0 is a massive architectural overhaul of the CUSTOS AI Trainer, transforming it from a prototype into a highly stable, enterprise-scale model pipeline.

* **Extreme Scale Memory Engine**: You can now upload **lakhs (100,000+) of images** directly into the browser. The system bypasses heavy visual DOM elements, instantly mathematically extracting features into RAM, preventing browser OOM (Out of Memory) crashes.
* **Anti-Freeze Queues**: A built-in thread-yielding algorithm ensures that the browser interface never freezes (no "Page Unresponsive" errors) during massive batch media ingestion. 
* **Enterprise Neural Network**: Upgraded to use **MobileNet V2** embeddings fed into a much wider, 512-neuron dense classifier. 
* **Long-Term Training & Early Stopping**: Supports up to **1000 Epochs** with automated `Early Stopping`. The system will safely auto-halt training if the validation accuracy stops improving, preventing model overfitting.
* **Clean, Accessible UI**: A completely rebuilt "Slate/White" UI with standard, friendly terminology allowing non-technical teammates to quickly build and understand AI classifiers without getting confused by technical jargon.
* **Native File System**: Bypasses the default browser "Downloads" folder and invokes the native Windows "Save As" API for direct folder management of JSON/CSV exports.

---

## 📌 Core Capabilities

### 1. Dataset Generation
Collect structured training samples in real-time or via massive batch upload:
* **Live Webcam Capture** (Burst record or single snapshots)
* **Image Uploads** (.jpg, .png, etc.)
* **Video Parsing** (Automatically scrubs through .mp4 files and extracts optimal frames)
* **Animated GIFs** 

### 2. High-Capacity Model Training
Train a neural network entirely within Google Chrome using **TensorFlow.js**:
* MobileNet V2 Feature Extraction
* 512-Dense Classification Head (Adam Optimizer, Categorical Crossentropy)
* 100% Local processing (Zero server dependencies, completely private)

### 3. Immediate Export & Integration
Seamlessly export everything required to plug into your wider CUSTOS infrastructure:
* `custos-guardian-model.json/bin`: The fully trained weights and architecture
* `custos-dataset-export.json`: High-density raw extracted array data
* `custos-meta.json`: Class configurations and mappings
* CSV Dataset support for external editing and validation

---

## 🛠️ Technology Stack

* **HTML/CSS**: Deeply optimized, lightweight clean user interface.
* **Vanilla JavaScript (ES6)**: State management, media ingestion queues, and file system APIs.
* **TensorFlow.js (v4.17)**: High-performance WebGL-accelerated neural network backend.
* **MobileNet (v2.1.0)**: Pre-trained convolutional base for ultra-fast, high-accuracy feature embeddings.
* **Chart.js**: Real-time performance graphing (Loss / Accuracy metrics).

---

## 💻 How To Run Locally

Modern browsers block webcam access and advanced file APIs when opening files via `file://`. A local server is strictly required.

**Step 1 — Install Node.js**  
Download and install the LTS version from [nodejs.org](https://nodejs.org).

**Step 2 — Install Live Server**  
Open your terminal and run:
```bash
npm install -g live-server
```

**Step 3 — Start the Environment**  
Navigate to your project folder and start the engine:
```bash
cd path/to/My-trainer-main
live-server
```
*(Alternatively, you can run `python -m http.server 8080`, and open `http://localhost:8080/` in Chrome).*
