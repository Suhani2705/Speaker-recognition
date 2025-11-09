## 🧠 Overview  
The system performs **automatic speaker recognition** by analyzing speech audio clips and classifying which known speaker produced the utterance.  
It uses **MFCC features** (Mel-Frequency Cepstral Coefficients) and their first and second derivatives as input to a **1D CNN + LSTM network**, combining spatial and temporal feature learning.  

### 🔑 Key Capabilities  
- End-to-end feature extraction and training pipeline  
- Extensive audio augmentation (noise, pitch shift, time-stretching)  
- Robust speaker classification with ROC-AUC and confusion matrix analysis  
- Visualization of MFCCs, ROC curves, and prediction confidences  

---

## 🎧 Dataset  
- **Dataset Path:** `/kaggle/input/speaker-recognition-dataset/16000_pcm_speeches/`  
- **Speakers Used:**
  - Nelson Mandela  
  - Benjamin Netanyahu  
  - Margaret Thatcher  
  - Jens Stoltenberg  
  - Julia Gillard  

Each speaker has multiple `.wav` files recorded at a **16 kHz** sampling rate.  
An optional `_background_noise_` folder is used for **noise-based data augmentation**.

---

## 🧩 Features and Techniques  

### 🎵 Audio Preprocessing & Augmentation  
- **Time-stretching** (phase vocoder)  
- **Pitch shifting**  
- **Background noise mixing**  
- **SpecAugment** (random time/frequency masking)  
- **Standardization** with `StandardScaler`  

### 🪶 Feature Extraction  
- **MFCCs** (13 coefficients)  
- **Delta** (first derivative)  
- **Delta-Delta** (second derivative)  
- **Padding/Truncation** to a fixed number of time steps (`MAX_TIME_STEPS = 100`)  

---

## 🧱 Model Architecture  
The model is a **hybrid CNN–LSTM network** designed for temporal and spectral pattern learning:

| Layer | Type | Description |
|:------:|:-----|:------------|
| 1 | Conv1D(64) + BatchNorm | Local feature extraction |
| 2 | Conv1D(64) + MaxPooling | Dimensionality reduction |
| 3 | Conv1D(128) + BatchNorm + MaxPooling | Deeper spectral features |
| 4 | LSTM(64) | Temporal pattern learning |
| 5 | Dense(128, relu) | Fully connected |
| 6 | Dense(num_classes, softmax) | Speaker classification |

**Regularization:** Dropout (0.3)  
**Optimizer:** Adam  
**Loss:** Sparse Categorical Crossentropy  

---

## 📊 Results and Metrics  
After training, the model produces:

- **Accuracy and loss curves** over epochs  
- **Classification report** with per-class Precision, Recall, and F1-score  
- **Confusion Matrix**  
- **ROC-AUC** (overall and per class)  
