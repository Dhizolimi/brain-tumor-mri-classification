import streamlit as st

def upload_mri():
    # Merender area input upload file (drag & drop) Streamlit khusus untuk berkas citra maupun PDF.
    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg","png","jpeg","pdf"]
    )

    return uploaded_file