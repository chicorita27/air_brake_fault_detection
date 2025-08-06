# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 13:47:05 2025
@author: 91701
"""

import pickle
import numpy as np
import streamlit as st

# Load the trained model
air_brake_model = pickle.load(open('C:/Users/91701/Downloads/air_brake_model.sav', 'rb'))

# Prediction function
def fault_prediction(my_reading):
    input_array = np.array(my_reading).reshape(1, -1)
    
    pred_label = air_brake_model.predict(input_array)[0]
    pred_prob = air_brake_model.predict_proba(input_array)[0][1]

    if pred_label == 1:
        return f"🚨 Faulty Detected! (Confidence: {round(pred_prob * 100, 2)}%)"
    else:
        return f"✅ Normal Brake Condition (Confidence: {round((1 - pred_prob) * 100, 2)}%)"

# Main app
def main():
    st.title("🚂 Air Brake Fault Detection System")

    st.markdown("### Enter Sensor Readings:")

    # Divide inputs into 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        timestamp = st.text_input('Time Instance')
        air_leak_rate = st.text_input('Air Leak Rate')

    with col2:
        brake_pressure = st.text_input('Brake Pressure')
        vibration_level = st.text_input('Vibration Level')

    with col3:
        ambient_temperature = st.text_input('Ambient Temperature')
        brake_response_time = st.text_input('Brake Response Time')

    with col4:
        brake_temperature = st.text_input('Brake Temperature')
        train_speed = st.text_input('Train Speed')

    fault = ""

    if st.button('Run Fault Detection'):
        try:
            input_data = list(map(float, [
                timestamp,
                brake_pressure,
                ambient_temperature,
                brake_temperature,
                air_leak_rate,
                vibration_level,
                brake_response_time,
                train_speed
            ]))
            fault = fault_prediction(input_data)
        except ValueError:
            fault = "⚠️ Please enter all values correctly (numbers only)."

    if fault:
        st.success(fault)

if __name__ == '__main__':
    main()
