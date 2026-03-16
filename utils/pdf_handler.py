from pdf2image import convert_from_bytes
from PIL import Image

def pdf_to_image(uploaded_file):
    # Membaca berkas PDF yang diunggah oleh pengguna dan mengonversi halaman pertamanya 
    # menjadi gambar berformat PIL RGB agar kompatibel dengan alur prapemrosesan model.
    # Pastikan file dibaca dari awal
    uploaded_file.seek(0)
    bytes_data = uploaded_file.read()
    
    images = convert_from_bytes(bytes_data)
    
    if images and len(images) > 0:
        image = images[0]           # ambil halaman pertama
        image = image.convert("RGB") # pastikan 3 channel
        return image
    
    return None