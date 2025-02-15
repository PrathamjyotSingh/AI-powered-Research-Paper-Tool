import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./gemma_finetuned"  # Adjust path if necessary

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n\n"
    return text

# Streamlit UI
st.title("📄 AI Research Paper Summarizer")
st.write("Upload a PDF research paper, and get a concise AI-generated summary!")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Truncate text if too long
    max_chars = 25000  # Adjust based on model's token limit
    pdf_text = pdf_text[:max_chars]
    
    st.subheader("Extracted Text Preview:")
    st.text_area("", pdf_text[:500], height=150)
    
    if st.button("Generate Summary ✨"):
        with st.spinner("Generating summary..."):
            inputs = tokenizer(pdf_text, return_tensors="pt", truncation=True, max_length=512)
            output = model.generate(**inputs, max_length=2000, do_sample=True, temperature=0.7)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
        
        st.subheader("📌 AI-Generated Summary:")
        st.write(summary)
