# Pastikan semua pustaka yang diperlukan diinstal
# Jika Anda menjalankan ini secara lokal, gunakan terminal untuk menginstal pustaka berikut:
# pip install streamlit pandas numpy scikit-learn matplotlib seaborn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Fungsi untuk memuat data
def load_data():
    try:
        data = pd.read_csv('spam.csv')
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Pastikan file 'spam.csv' tersedia di direktori.")
        return None
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


# Fungsi untuk membagi data
def split(df):
    y = df['yesno']  # Ganti sesuai kolom target Anda
    x = df.drop(columns=['yesno'])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


# Fungsi untuk menampilkan berbagai metrik
def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        fig, ax = plt.subplots()
        disp.plot(cmap='Blues', ax=ax)
        st.pyplot(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)

        fig, ax = plt.subplots()
        roc_disp.plot(ax=ax)
        st.pyplot(fig)

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)

        fig, ax = plt.subplots()
        pr_disp.plot(ax=ax)
        st.pyplot(fig)


# Fungsi utama aplikasi
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Classify your data! 🚀")
    st.sidebar.markdown("Classify your data! 🚀")

    # Muat data
    df = load_data()
    if df is None:
        return  # Hentikan jika data tidak dimuat

    class_names = ['Not Spam', 'Spam']
    x_train, x_test, y_train, y_test = split(df)

    # Tampilkan heatmap korelasi
    if st.sidebar.checkbox("Show Feature Correlation Heatmap", False):
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Pilih model di sidebar
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
    )

    # Konfigurasi SVM
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')
        )

        if st.sidebar.button("Classify", key='classify_svm'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))
            st.write("F1-Score:", f1_score(y_test, y_pred))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # Konfigurasi Logistic Regression
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')
        )

        if st.sidebar.button("Classify", key='classify_lr'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))
            st.write("F1-Score:", f1_score(y_test, y_pred))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # Konfigurasi Random Forest
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')
        )

        if st.sidebar.button("Classify", key='classify_rf'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=(bootstrap == 'True'))
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))
            st.write("F1-Score:", f1_score(y_test, y_pred))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # Tampilkan data mentah
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Dataset")
        st.write(df)


if __name__ == '__main__':
    main()
