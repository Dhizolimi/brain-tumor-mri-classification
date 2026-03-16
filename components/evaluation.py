import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score, cohen_kappa_score
import os

def show_confusion_matrix(y_true, y_pred, class_names):
    # Menampilkan visualisasi Confusion Matrix menggunakan Seaborn Heatmap.
    # Ini membantu dalam mengidentifikasi kelas mana yang sering disalahartikan oleh model.
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def show_classification_report(y_true, y_pred, class_names):
    # Menghitung dan mengembalikan metrik evaluasi klasifikasi (Precision, Recall, F1-Score,
    # dan Support) per kelas, serta merendernya dalam bentuk tabel Streamlit (Dataframe).
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    # Create a DataFrame for better display in Streamlit
    report_dict = {
        "Class": class_names,
        "Precision": np.round(precision, 4),
        "Recall": np.round(recall, 4),
        "F1-Score": np.round(f1, 4),
        "Support": support
    }
    df_report = pd.DataFrame(report_dict)
    
    st.write("### Per-Class Metrics")
    st.dataframe(df_report, use_container_width=True)

def show_roc_curve(y_true, y_score, class_names):
    # Menampilkan kurva Receiver Operating Characteristic (ROC) dengan strategi One-vs-Rest, 
    # dan menghitung area under curve (AUC) untuk setiap kelas.
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        # We treat class i as the positive class, all others as negative
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_score_class = np.array(y_score)[:, i]
        
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_class)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (One-vs-Rest)')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def render_evaluation_page():
    # Fungsi utama yang merender halaman "Model Evaluation" di Streamlit. 
    # Membaca data performa prediksi aktual `densenet_evaluation_results.csv` dan merender 
    # berbagai matrik beserta visualisasinya ke dalam antarmuka UI.
    st.title("📊 Model Evaluation Metrics")
    st.markdown("Halaman ini menampilkan metrik evaluasi aktual dari model DenseNet121 pada *test set*.")
    
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    
    csv_path = "model/densenet_evaluation_results.csv"
    if not os.path.exists(csv_path):
         st.error(f"File hasil evaluasi tidak ditemukan di: `{csv_path}`")
         return
         
    # Load actual evaluation data
    df = pd.read_csv(csv_path)
    
    # Berdasarkan kolom yang diinfokan:
    # y_true_index, y_pred_index (0-3)
    y_true = df['y_true_index'].values
    y_pred = df['y_pred_index'].values
    
    # Probabilitas per kelas
    y_score = df[['prob_glioma', 'prob_meningioma', 'prob_notumor', 'prob_pituitary']].values
    
    # ---------------------------
    # Global Metrics Calculation
    # ---------------------------
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # Tampilkan High-level Metrics
    st.write("### Overall Metrics")
    colA, colB, colC = st.columns(3)
    colA.metric("Accuracy", f"{acc*100:.2f}%")
    colB.metric("Macro F1-Score", f"{f1_macro:.4f}")
    colC.metric("Cohen's Kappa", f"{kappa:.4f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        show_confusion_matrix(y_true, y_pred, class_names)
        
    with col2:
        st.subheader("ROC-AUC")
        show_roc_curve(y_true, y_score, class_names)
        
    st.divider()
    show_classification_report(y_true, y_pred, class_names)