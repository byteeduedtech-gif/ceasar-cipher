import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="Caesar Cipher + Cybersecurity ML (SVM)",
    layout="centered",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
body 
{ 
  background-color: #ffffff; 
  color: #1E1E2F; 
}
h1 
{ 
  color: #0056b3; 
  text-align: center; 
}
h2, h3 
{ 
 color: #ff6600; 
}
[data-testid="stSidebar"] 
{ 
  background-color: grey; 
  border-right: 2px solid #0056b3; 
}
div.stButton > button 
{
    background-color: #0056b3; 
    color: white;
    border-radius: 10px; 
    border: none; transition: 0.3s;
}
div.stButton > button:hover 
{ 
  background-color: #ff6600; 
  color: white; 
}
[data-testid="stRadio"] label, .stSlider 
 { 
  color: #0056b3; 
 }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


def caesar_encrypt(text, shift):
    result = ''
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result

def caesar_decrypt(text, shift):
    return caesar_encrypt(text, -shift)

def text_to_features(text):
    """Convert text into frequency vector of letters"""
    letters = string.ascii_lowercase
    freq = np.zeros(26)
    total = 0
    for char in text.lower():
        if char in letters:
            freq[ord(char) - ord('a')] += 1
            total += 1
    if total > 0:
        freq = freq / total
    return freq

def generate_dataset(n_samples=800):
    """Generate synthetic Caesar cipher dataset"""
    words = [
        "machine learning is fun",
        "python streamlit app",
        "data science project",
        "cryptography example",
        "college assignment work",
        "artificial intelligence",
        "frequency analysis",
        "linear regression model"
    ]
    X, y = [], []
    for _ in range(n_samples):
        text = random.choice(words)
        shift = random.randint(0, 25)
        encrypted = caesar_encrypt(text, shift)
        X.append(text_to_features(encrypted))
        y.append(shift)
    return np.array(X), np.array(y)







def generate_synthetic_cyber_dataset():
    """Generate fake CICIDS + ODIDS dataset for testing"""
    np.random.seed(42)
    num_features = 6
    cicids_attacks = [f"CICIDS_Attack_{i}" for i in range(1, 16)]
    odids_attacks = [f"ODIDS_Attack_{i}" for i in range(1, 30)]

    cicids_data = pd.DataFrame(
        np.random.rand(100, num_features),
        columns=[f"feature_{i}" for i in range(num_features)]
    )
    cicids_data["label"] = np.random.choice(cicids_attacks, 100)

    odids_data = pd.DataFrame(
        np.random.rand(100, num_features),
        columns=[f"feature_{i}" for i in range(num_features)]
    )
    odids_data["label"] = np.random.choice(odids_attacks, 100)

    return cicids_data, odids_data

def load_cyber_dataset():
    """Load real datasets if available, else generate synthetic ones"""
    if os.path.exists("cicids.csv") and os.path.exists("odids2022.csv"):
        cicids = pd.read_csv("cicids.csv")
        odids = pd.read_csv("odids2022.csv")
    else:
        st.info("Real datasets not found. Using synthetic CICIDS + ODIDS data.")
        cicids, odids = generate_synthetic_cyber_dataset()

    df = pd.concat([cicids, odids], axis=0, ignore_index=True)

    if "label" not in df.columns:
        st.error(" Dataset must have a 'label' column.")
        return None, None, None, None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0)
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, le, df["label"].value_counts()


def plot_frequency(text):
    letters = string.ascii_lowercase
    freq = {ch: 0 for ch in letters}
    for char in text.lower():
        if char in freq:
            freq[char] += 1
    fig, ax = plt.subplots()
    ax.bar(freq.keys(), freq.values(), color="#0056b3")
    ax.set_title("Letter Frequency Distribution", color="#ff6600")
    st.pyplot(fig)

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    ax.set_title("Confusion Matrix (Predicted vs Actual)", color="#ff6600")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def plot_accuracy(train_acc, test_acc):
    fig, ax = plt.subplots()
    ax.bar(["Train", "Test"], [train_acc, test_acc], color=["#0056b3", "#ff6600"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Test Accuracy", color="#ff6600")
    st.pyplot(fig)


st.title("Caesar Cipher + Cybersecurity ML (SVM)")
st.caption("Built by Group-44 | VIT Batch 2024")

dataset_mode = st.sidebar.radio(
    "Choose Dataset",
    ["Caesar Cipher (Synthetic)", "Cybersecurity (CICIDS + ODIDS2022)"]
)


if dataset_mode == "Caesar Cipher (Synthetic)":
    X, y = generate_dataset()
    label_encoder = None
    dataset_info = None
    st.info("Using synthetic Caesar cipher dataset for training.")
else:
    X, y, label_encoder, dataset_info = load_cyber_dataset()
    if X is None:
        st.stop()

    st.success("Loaded CICIDS + ODIDS2022 Dataset")
    st.write("### Sample of Dataset")
    st.dataframe(pd.DataFrame(X).head())
    
    st.write("### Attack Distribution")
    st.bar_chart(dataset_info)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


mode = st.radio("Choose Mode", ["Encrypt", "Decrypt", "Crack with ML (SVM)", "Model Performance"])
text = st.text_area("Enter your text:")
shift = st.slider("Select Shift (for Encrypt/Decrypt)", 0, 25, 3)

if st.button("Run"):
    if mode == "Encrypt":
        output = caesar_encrypt(text, shift)
        st.subheader("Encrypted Text:")
        st.write(output)
        plot_frequency(output)

    elif mode == "Decrypt":
        output = caesar_decrypt(text, shift)
        st.subheader("Decrypted Text:")
        st.write(output)
        plot_frequency(output)

    elif mode == "Crack with ML (SVM)":
        if dataset_mode == "Caesar Cipher (Synthetic)":
            if text.strip() == "":
                st.warning("Please enter some ciphertext to crack.")
            else:
                features = text_to_features(text).reshape(1, -1)
                predicted_shift = model.predict(features)[0]
                decrypted = caesar_decrypt(text, predicted_shift)

                st.subheader("ML Predicted Shift (SVM):")
                st.write(predicted_shift)
                st.subheader("Decrypted Text (ML):")
                st.write(decrypted)
        else:
            st.subheader("ML Prediction (Cybersecurity Dataset)")
            pred = model.predict(X_test[:1])[0]
            label_name = label_encoder.inverse_transform([pred])[0] if label_encoder else pred
            st.write(f"Predicted Attack Type: **{label_name}**")
            st.info("Dataset used: CICIDS (15 attacks) + ODIDS2022 (29 attacks)")

        st.subheader("Model Test Accuracy (SVM):")
        st.write(f"{test_acc * 100:.2f}%")

    else:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(cm)

        st.subheader("Accuracy Comparison")
        plot_accuracy(train_acc, test_acc)

        if dataset_info is not None:
            st.subheader("Dataset Class Distribution")
            st.dataframe(dataset_info)
