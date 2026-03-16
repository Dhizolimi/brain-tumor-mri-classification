import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam_grayscale(model, img, original_image):
    # Menghasilkan *Heatmap* Grad-CAM (Gradient-weighted Class Activation Mapping) 
    # untuk memberikan penjelasan visual mengenai area fitur mana pada MRI yang paling 
    # mempengaruhi keputusan klasifikasi model (Explainability).
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # Jika input masih grayscale (1 channel), duplikasi menjadi 3 channel
    if img_tensor.shape[-1] == 1:
        img_tensor = tf.concat([img_tensor, img_tensor, img_tensor], axis=-1)

    # Temukan index densenet121
    densenet_index = -1
    for i, layer in enumerate(model.layers):
        if layer.name == "densenet121":
            densenet_index = i
            break
    
    if densenet_index == -1:
        raise ValueError("Layer 'densenet121' tidak ditemukan pada model.")

    base_model = model.layers[densenet_index]

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_tensor)
        tape.watch(conv_outputs)
        
        x = conv_outputs
        # Teruskan forward pass ke layer klasifikasi
        for layer in model.layers[densenet_index + 1:]:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x)

        predictions = x
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)+1e-8

    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    if len(original_image.shape)==2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255*heatmap),
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image.astype(np.uint8),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay