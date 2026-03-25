import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

from utils.preprocessing import preprocess_image
from utils.pdf_handler import pdf_to_image
from utils.inference import predict
from utils.gradcam import generate_gradcam_grayscale
from utils.report import create_report

from components.evaluation import render_evaluation_page
from components.uploader import upload_mri
from components.result_panel import show_result
from components.chart import probability_chart

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Brain Tumor AI",
    layout="wide"
)

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Tumor Analysis", "Model Evaluation"])

st.sidebar.header("Model Information")
st.sidebar.write("Architecture: DenseNet121 (RGB)")
st.sidebar.write("Dataset: MRI Brain Tumor Dataset")
st.sidebar.write("Accuracy: ~95%")

if page == "Tumor Analysis":
    # ---------------------------------
    # Header
    # ---------------------------------
    st.title("🧠 Brain Tumor AI Detection")
    st.markdown("""
    ### AI-powered MRI Analysis
    Upload a brain MRI scan to detect tumor type using deep learning.
    Supported formats:
    - JPG
    - PNG
    - PDF
    """)

    # ---------------------------------
    # Load Model
    # ---------------------------------
    @st.cache_resource
    def get_model():
        # Memuat model Deep Learning (.keras) ke dalam memori aplikasi.
        # Dilengkapi dengan *decorator* `@st.cache_resource` dari Streamlit agar 
        # model hanya dimuat satu kali saat aplikasi pertama kali dijalankan, 
        # sehingga menghemat waktu komputasi pada pemanggilan inferensi berikutnya.
        model = keras.models.load_model("model/best_model_densenet.keras")
        return model

    model = get_model()


    # ---------------------------------
    # Layout Columns (Upload & Result)
    # ---------------------------------
    st.divider()
    col_upload, col_result = st.columns([1,1])

    with col_upload:
        uploaded_file = upload_mri()

    # ---------------------------------
    # File Handling & Analysis Flow
    # ---------------------------------
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            image = pdf_to_image(uploaded_file)
            if image is None:
                st.error("Gagal membaca file PDF atau file kosong.")
                st.stop()
        else:
            image = Image.open(uploaded_file).convert("L")  # grayscale

        # Preview di kolom upload
        with col_upload:
            st.image(image, caption="MRI Preview", width=350)

        # Inisialisasi session state untuk menyimpan hasil analisis dan nama file
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = None
        if "last_file" not in st.session_state:
            st.session_state.last_file = None

        # Reset jika file yang diunggah berubah
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.analysis_results = None
            st.session_state.last_file = uploaded_file.name

        # Tombol jalankan analisis
        if col_upload.button("Run AI Analysis"):
            with st.spinner("Running model inference..."):
                # Preprocess
                img_rgb = preprocess_image(image)  # (1,224,224,1)
                
                # Predict
                result, confidence, probs = predict(model, img_rgb)
                
                # GradCAM
                try:
                    heatmap = generate_gradcam_grayscale(model, img_rgb, np.array(image))
                except Exception as e:
                    heatmap = None
                    st.warning(f"GradCAM error: {e}")

                # Simpan hasil ke session state
                st.session_state.analysis_results = {
                    "result": result,
                    "confidence": confidence,
                    "probs": probs,
                    "heatmap": heatmap
                }

        # Jika hasil analisis sudah ada di session state, tampilkan
        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            result = res["result"]
            confidence = res["confidence"]
            probs = res["probs"]
            heatmap = res["heatmap"]

            # ---------------------------------
            # Display Results (Top Right)
            # ---------------------------------
            class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            # Jika inference.py mengembalikan integer, ubah ke string
            tumor_type = class_names[result] if isinstance(result, (int, np.integer)) else result

            with col_result:
                show_result(prediction=tumor_type, confidence=confidence)

                # Probability chart
                probability_chart(probabilities=probs, class_names=class_names)

            st.divider()

            # ---------------------------------
            # Explainability Section
            # ---------------------------------
            st.markdown("### Explainability")
            
            col_img, col_cam = st.columns(2)
            
            with col_img:
                st.subheader("Original MRI")
                st.image(image, use_container_width=True)
                
            with col_cam:
                st.subheader("Grad-CAM Overlay")
                if heatmap is not None:
                    st.image(heatmap, use_container_width=True)
                else:
                    st.info("Heatmap tidak tersedia untuk gambar ini.")

            st.divider()

            st.info(
                f"Model memprediksi **{tumor_type}** karena model melihat pada area-area berikut pada heatmap Grad-CAM:\n\n"
                "- 🔴 **Area Merah:** Menunjukkan area dengan pengaruh paling tinggi (sangat penting) dalam proses prediksi model.\n"
                "- 🟡/🟢 **Area Kuning & Hijau:** Menunjukkan area dengan pengaruh menengah.\n"
                "- 🔵 **Area Biru:** Menunjukkan area dengan pengaruh paling rendah (kurang penting) terhadap hasil prediksi."
            )

            # ---------------------------------
            # Report Generation
            # ---------------------------------
            
            try:
                report_path = create_report(
                    image=image,
                    heatmap=heatmap,
                    prediction=tumor_type,
                    confidence=float(confidence),
                    model_name="DenseNet121 (RGB)"
                )
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="Brain_Tumor_MRI_Analysis_Report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Gagal memuat laporan: {e}")

            st.markdown("---")
            st.warning("⚠️ Disclaimer: This system is not a medical diagnostic tool. Please consult with a healthcare professional.")

elif page == "Model Evaluation":
    render_evaluation_page()