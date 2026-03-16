import streamlit as st

def show_result(prediction, confidence):
    # Komponen UI untuk menampilkan teks hasil prediksi jenis tumor dan nilai kepercayaannya,
    # sekaligus membuat progress bar (barchart) dengan Streamlit.
    st.subheader("Prediction Result")

    st.metric(
        "Tumor Type",
        prediction
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.progress(float(confidence))