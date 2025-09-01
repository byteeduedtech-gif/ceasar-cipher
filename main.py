import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Caesar Cipher ML by Group-44",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------
# Custom CSS Styling
# -------------------
custom_css = """
<style>
/* Background */
body {
    background-color: #ffffff;
    color: #1E1E2F;
}

/* Title */
h1 {
    color: #0056b3; /* Blue */
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #ff6600; /* Orange */
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f5f9ff;
    border-right: 2px solid #0056b3;
}

/* Buttons */
div.stButton > button {
    background-color: #0056b3;
    color: white;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #ff6600;
    color: white;
}

/* Radio and Slider labels */
[data-testid="stRadio"] label, .stSlider {
    color: #0056b3;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------
# Caesar Cipher Logic
# -------------------
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

# -------------------
# Frequency Features
# -------------------
def text_to_features(text):
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

# -------------------
# Dataset Generation
# -------------------
def generate_dataset(n_samples=800):
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

# -------------------
# Train ML Model
# -------------------
X, y = generate_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -------------------
# Frequency Plot
# -------------------
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

# -------------------
# Confusion Matrix Plot
# -------------------
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    ax.set_title("Confusion Matrix (Predicted vs Actual Shifts)", color="#ff6600")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -------------------
# Accuracy Comparison Plot
# -------------------
def plot_accuracy(train_acc, test_acc):
    fig, ax = plt.subplots()
    ax.bar(["Train", "Test"], [train_acc, test_acc], color=["#0056b3", "#ff6600"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Test Accuracy", color="#ff6600")
    st.pyplot(fig)

# -------------------
# Streamlit UI
# -------------------
st.title("Caesar Cipher with ML") 
st.caption("Built by Group-44 | VIT Batch 2024")

# st.sidebar.subheader("📌 About the Project")
# st.sidebar.info("Encrypt/Decrypt text using Caesar Cipher and break it with ML. "
#                 "\n\nMade with Python, Streamlit, Scikit-learn, and Matplotlib.")

mode = st.radio("Choose Mode", ["Encrypt", "Decrypt", "Crack with ML", "Model Performance"])
text = st.text_area("Enter your text:")
shift = st.slider("Select Shift (for Encrypt/Decrypt)", 0, 25, 3)

if st.button("Run"):
    if mode == "Encrypt":
        output = caesar_encrypt(text, shift)
        st.subheader(" Encrypted Text:")
        st.write(output)
        plot_frequency(output)

    elif mode == "Decrypt":
        output = caesar_decrypt(text, shift)
        st.subheader(" Decrypted Text:")
        st.write(output)
        plot_frequency(output)

    elif mode == "Crack with ML":
        if text.strip() == "":
            st.warning(" Please enter some ciphertext to crack.")
        else:
            features = text_to_features(text).reshape(1, -1)
            predicted_shift = model.predict(features)[0]
            decrypted = caesar_decrypt(text, predicted_shift)

            st.subheader(" ML Predicted Shift:")
            st.write(predicted_shift)
            st.subheader("Decrypted Text (ML):")
            st.write(decrypted)

            st.subheader("Model Test Accuracy:")
            st.write(f"{test_acc * 100:.2f}%")

    else:  # Model Performance
        st.subheader("📊 Confusion Matrix")
        plot_confusion_matrix(cm)

        st.subheader("📈 Accuracy Comparison")
        plot_accuracy(train_acc, test_acc)
