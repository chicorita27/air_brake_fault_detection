# 🚂 Air Brake Fault Detection System

A web-based application built with **Streamlit** that detects faults in train air brake systems using a **Random Forest Classifier**. Users can input sensor readings to determine whether the brakes are in a **Normal** or **Faulty** condition.

---

## 🔗 Live Demo

👉 [Open the App](https://airbrakefaultdetection-qi96clytxq68okk73nardq.streamlit.app/)

---

## 🧠 Key Features

- Predicts air brake faults based on sensor inputs using a **machine learning model**.
- Displays:
  - 🚨 **Fault Detected** (with probability)
  - ✅ **Normal Condition** (with confidence score)
- Clean, multi-column input layout using **Streamlit**.
- Real-time results with probability scoring.

---

## 🔍 Machine Learning Model

- **Model Type**: Random Forest Classifier  
- **Training Library**: scikit-learn  
- **Model Saved Using**: `pickle`

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **scikit-learn**
- **NumPy**
- **Pickle**

---

## 🧪 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/air-brake-fault-detection.git
   cd air-brake-fault-detection
