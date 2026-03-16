import streamlit as st

def probability_chart(probabilities, class_names):
    # Menerima list probabilitas dan render grafik balok / Bar Chart sederhana
    # menggunakan Streamlit agar pengguna bisa memvisualisasikan persebaran prediksi.
    prob_dict = {k: float(v) for k,v in zip(class_names, probabilities)}
    st.subheader("Probability Distribution")
    st.bar_chart(prob_dict)