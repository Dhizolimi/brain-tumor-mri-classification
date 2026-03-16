from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from PIL import Image
import tempfile
import numpy as np

def create_report(image, heatmap, prediction, confidence, model_name="DenseNet121 (RGB)"):
    # Dinamis membuat dokumen laporan analitik format PDF yang merangkum hasil prediksi 
    # AI, tingkat kepercayaan (confidence), nama model, dan visualisasi perbandingan 
    # (MRI Asli vs *Heatmap* Grad-CAM). File disimpan secara temporer.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Brain Tumor MRI Analysis")

    # Image (Original)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 90, "Original MRI:")
    if image is not None:
        try:
            if not isinstance(image, Image.Image):
                img_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            else:
                img_pil = image.convert("RGB")
            c.drawImage(ImageReader(img_pil), 50, height - 280, width=180, height=180)
        except Exception as e:
            c.drawString(50, height - 110, f"[Error: {e}]")

    # Image (GradCAM)
    c.drawString(300, height - 90, "Grad-CAM Heatmap:")
    if heatmap is not None:
        try:
            if not isinstance(heatmap, Image.Image):
                heat_pil = Image.fromarray(np.uint8(heatmap)).convert("RGB")
            else:
                heat_pil = heatmap.convert("RGB")
            c.drawImage(ImageReader(heat_pil), 300, height - 280, width=180, height=180)
        except Exception as e:
            c.drawString(300, height - 110, f"[Error: {e}]")

    # Prediction Info
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 320, f"Prediction: {prediction}")
    c.drawString(50, height - 350, f"Confidence: {confidence * 100:.2f}%")
    c.drawString(50, height - 380, f"Model Used: {model_name}")

    c.save()
    return tmp.name