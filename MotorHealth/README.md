# VibraSense AI: Multimodal Edge-AI Motor Health Monitoring

**VibraSense AI** is a fully functional, low-cost, speed-aware condition-monitoring prototype for electric motors (demonstrated on a guarded stand fan). Built on the ESP32-S3, this system performs real-time, on-device (Edge AI) anomaly detection by fusing vibration, acoustic, rotational speed, and temperature data.

This repository contains the complete working prototype, including the firmware, the offline Machine Learning training pipeline, and the Python data acquisition scripts.

---

## 🏆 Current Status: Working Prototype

The system is successfully deployed and operating at a demonstrable level. It features:

*   **Real-Time Multimodal Feature Extraction:** Computes 13 features every second, including 128-point FFT (Frequency Domain) for vibration and Zero Crossing Rate (ZCR) for audio.
*   **On-Device Inference:** Runs a robust Median/MAD Z-score model directly on the ESP32-S3.
*   **Multimodal Fusion & False-Alarm Rejection:** Accurately ignores background talking while detecting mechanical faults by requiring multimodal agreement (Vibration × 0.6 + Audio × 0.4).
*   **Live User Interface:** A 3-tier LED system (Green/Yellow/Red) and an SH1106 OLED displaying live status (`HEALTHY`, `WARNING`, `ANOMALY`), fused scores, and individual sensor Z-scores.

---

## 🧠 System Architecture & Edge AI Pipeline

The AI operates on a 1-second sliding window.

### 1. The 13-Feature Vector
Every second, the ESP32-S3 extracts:
*   **Vibration (ADXL345):** X/Y/Z RMS, X/Y/Z Peak, and the dominant **FFT Peak Frequency** (Hz).
*   **Audio (INMP441):** RMS, Peak, and **Zero Crossing Rate (ZCR)** to detect high-frequency transients (like paper slashing).
*   **Context:** Hall-Effect RPM, DS18B20 Temperature, and ADXL FIFO overruns.

### 2. Machine Learning (Train Offline, Deploy on Edge)
*   **Training (Python):** Baseline data is collected across multiple speeds. An Isolation Forest algorithm (using `scikit-learn`) learns the normal operating envelope.
*   **Deployment (C++):** The baseline parameters (Median and robust Scale) are exported to `VibraSense_EdgeModel.h`. The ESP32 computes the Z-score for each feature.
*   **Fusion Logic:**
    *   `Z_Vib` = Max Z-score of vibration features
    *   `Z_Aud` = Max Z-score of audio features
    *   `Fused Score` = `(Z_Vib * 0.6) + (Z_Aud * 0.4)`
*   **3-Tier Decision:**
    *   🟢 **HEALTHY:** Fused score < 2.5
    *   🟡 **WARNING:** Fused score > 2.5
    *   🔴 **ANOMALY:** Fused score > 4.5, OR if both `Z_Vib > 4.0` and `Z_Aud > 4.0` (Instant trigger).

---

##  Hardware Configuration & Pin Mapping

| Component | Interface | ESP32-S3 GPIO | Notes |
| :--- | :--- | :--- | :--- |
| **ADXL345** | I²C (0x53) | SDA=8, SCL=9 | 3.3V Power |
| **SH1106 OLED** | I²C (0x3C) | SDA=8, SCL=9 | Shared I²C bus |
| **INMP441 Mic** | I²S | BCLK=4, WS=5, SD=6 | L/R tied to GND |
| **A3144 Hall** | GPIO Interrupt | 7 | Requires 10kΩ pull-up to 3.3V |
| **DS18B20** | 1-Wire | 15 | Requires 4.7kΩ pull-up to 3.3V |
| **Red LED** | GPIO | 21 | Anomaly Indicator (via 220Ω) |
| **Yellow LED** | GPIO | 48 | Warning Indicator (via 220Ω) |
| **Green LED** | GPIO | 47 | Healthy Indicator (via 220Ω) |

*Note: All components must share a common ground (GND).*

---

## 📂 Repository Structure

*   **`Motor_Health_Detection_Using_Edge_ai.ino`** — The main firmware. Handles I2S/I2C acquisition, FFT computation, ZCR, and runs the Edge AI inference loop.
*   **`VibraSense_EdgeModel.h`** — The generated C++ header containing the trained model's `MEDIAN` and `SCALE` arrays.
*   **`vibrasense_integrated_logger.py`** — The PC logger. Connects via COM port and saves the 13-feature vectors into structured CSV files.
*   **`train_vibrasense_model.py`** — The ML training pipeline. Reads the CSVs, trains the Isolation Forest, calculates the robust Z-scores, and exports the `.h` file.
*   **`vibrasense_data/`** — Directory containing the recorded baseline and demo CSV files.

---

## Procedure for Replication (From Scratch)

If you are setting up this project from a fresh environment, follow these steps.

### 1. Python Environment Setup
```bash
pip install pyserial pandas numpy scikit-learn joblib
