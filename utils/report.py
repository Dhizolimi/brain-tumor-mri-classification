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

    # Prediction Info
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 90, f"Prediction: {prediction}")
    c.drawString(50, height - 120, f"Confidence: {confidence * 100:.2f}%")
    c.drawString(50, height - 150, f"Model Used: {model_name}")

    # Image (Original)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 200, "Original MRI:")
    if image is not None:
        try:
            if not isinstance(image, Image.Image):
                img_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            else:
                img_pil = image.convert("RGB")
            c.drawImage(ImageReader(img_pil), 50, height - 390, width=180, height=180)
        except Exception as e:
            c.drawString(50, height - 220, f"[Error: {e}]")

    # Image (GradCAM)
    c.drawString(300, height - 200, "Grad-CAM Heatmap:")
    if heatmap is not None:
        try:
            if not isinstance(heatmap, Image.Image):
                heat_pil = Image.fromarray(np.uint8(heatmap)).convert("RGB")
            else:
                heat_pil = heatmap.convert("RGB")
            c.drawImage(ImageReader(heat_pil), 300, height - 390, width=180, height=180)
        except Exception as e:
            c.drawString(300, height - 220, f"[Error: {e}]")

    # Grad-CAM Explanation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 430, "Penjelasan Heatmap Grad-CAM:")

    c.setFont("Helvetica", 11)
    explanation_lines = [
        f"Model memprediksi {prediction} karena model melihat pada area-area berikut",
        "pada heatmap Grad-CAM:",
        "",
        "- Area Merah: Menunjukkan area dengan pengaruh paling tinggi (sangat penting)",
        "  dalam proses prediksi model.",
        "- Area Kuning & Hijau: Menunjukkan area dengan pengaruh menengah.",
        "- Area Biru: Menunjukkan area dengan pengaruh paling rendah (kurang penting)",
        "  terhadap hasil prediksi."
    ]
    
    text_y = height - 450
    for line in explanation_lines:
        c.drawString(50, text_y, line)
        text_y -= 15

    c.save()
    return tmp.name