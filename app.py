import streamlit as st
import torch
from torchvision.transforms import ToPILImage
import random
from train_generator import Generator

st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #2c3e50;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = Generator()
model.load_state_dict(torch.load("models/generator.pth", map_location="cpu"))
model.eval()

st.markdown("<h1 style='text-align: center;'>🖋️ Handwritten Digit Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Choose a digit (0–9) and generate 5 synthetic MNIST-style images.</p>", unsafe_allow_html=True)

digit = st.number_input("Digit to generate (0–9)", min_value=0, max_value=9, value=0)

if st.button("Generate Samples"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        outputs = model(z, labels)
        outputs = (outputs + 1) / 2  # Scale to [0,1]

    st.subheader(f"Generated Samples for '{digit}'")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        img = ToPILImage()(outputs[i])
        col.image(img.resize((140, 140)), caption=f"Sample {i+1}", use_column_width=True)

st.markdown("""
    <style>
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)