# 🏥 GlucoBreath – Non-Invasive Glucose Prediction System

## 🌟 Project Overview
**GlucoBreath** is a conceptual non-invasive glucose monitoring system that simulates breath-based glucose prediction using synthetic IoT sensor data and machine learning. In real-world applications, breath acetone is a known biomarker for blood glucose levels, particularly in diabetic patients. This project demonstrates how sensor data (Acetone, Temperature, and Humidity) can be leveraged to estimate blood sugar levels.

## 📊 Dataset Generation
The project utilizes the **Pima Indians Diabetes Dataset** and enriches it with synthetic sensor parameters:

*   **Acetone (Biomarker for diabetes):** Correlated with blood glucose using a random multiplier (0.8x to 1.5x) to simulate typical physiological variation.
*   **Temperature (IoT Sensor):** Simulated environmental/breath temperature (25°C to 37°C).
*   **Humidity (IoT Sensor):** Simulated breath humidity (40% to 80%).

The final processed dataset is saved as `updated_dataset.csv`.

## 🤖 Machine Learning Model
The system uses a **Random Forest Regressor** as the primary predictive model, with **Linear Regression** as a performance baseline.

*   **Features:** Acetone, Temperature, Humidity
*   **Target:** Blood Glucose level (mg/dL)
*   **Splits:** 80% Training, 20% Testing
*   **Performance Metric:** Mean Absolute Error (MAE)

## 📁 Project Structure
```
project/
│
├── diabetes.csv           (Original Pima dataset)
├── updated_dataset.csv     (Processed with IoT features)
├── dataset_prepare.py      (Data engineering script)
├── model_train.py          (ML training & visualization)
├── predict.py              (CLI-based prediction tool)
├── gluco_model.pkl         (Saved model artifact)
├── correlation_matrix.png  (Visual report)
├── actual_vs_predicted.png (Visual report)
└── README.md               (Project documentation)
```

## 🚀 How to Run the Project

### 1. 🛠️ Prepare the Dataset
Generates the synthetic sensor features.
```bash
python dataset_prepare.py
```

### 2. 🧪 Train the Machine Learning Model
Trains the regression models and saves the best one.
```bash
python model_train.py
```

### 3. 🤔 Predict Glucose Level
Enter sensor readings to get a glucose prediction.
```bash
python predict.py
```

## 📈 Future Scope
*   **Hardware Integration:** Connecting real acetone (MQ-138) and DHT sensors to an ESP32 or Arduino.
*   **Real-time Dashboard:** Developing a web or mobile interface for continuous monitoring.
*   **Explainable AI:** Using SHAP or LIME to explain why a certain prediction was made based on sensor input.

---
**Disclaimer:** This is a simulation project and should not be used for medical diagnosis.
