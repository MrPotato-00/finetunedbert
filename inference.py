import torch
import streamlit as st
import numpy as np
import pickle 
from transformers import AutoTokenizer

torch.classes.__path__ = []
id2label= {0: "Safe", 1:"Not Safe"}

@st.cache_data
def model_tokenizer():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    tokenizer= AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, device

st.title("Phising Link Detector")

model, tokenizer, device= model_tokenizer()
def pred(text):
    
    tokenized_text= tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs= model(**tokenized_text)
        logits= outputs.logits.argmax(-1).item()

    predicted_class= id2label[logits]
    
    return predicted_class

try: 
    
    text= st.text_input("Enter website", key="placeholder")   
    if text is not None:
        if st.button("Pred"):
            predicted_class= pred(text)
            st.write(f"The predicted label for {text}: {predicted_class}")

except Exception as e:
    print(e)
