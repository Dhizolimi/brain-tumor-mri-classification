import numpy as np

def predict(model, img_rgb):
    # Memprediksi kelas tumor dari citra MRI yang diberikan dan mengembalikan hasil probabilitasnya.
    # Fungsi ini mengambil model terlatih dan gambar yang telah diproses (img_rgb), 
    # kemudian melakukan inferensi untuk menentukan jenis tumor berdasarkan probabilitas tertinggi.
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    preds = model.predict(img_rgb)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0, class_idx]
    
    tumor_type = class_names[class_idx]
    return tumor_type, confidence, preds[0]