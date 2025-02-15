# AI-Powered Research Paper Summarizer

## Overview
This project develops an **LLM-based research paper summarization system** using **Google’s GEMMA model**, fine-tuned with **LoRA** for efficiency. The model processes academic papers and generates concise, context-aware summaries.

## Features
✅ **Fine-tuned GEMMA model** for improved summarization.  
✅ **LoRA-based parameter-efficient fine-tuning** for reduced computational cost.  
✅ **Quantization (4-bit & 8-bit)** using **BitsAndBytes (bnb)** for memory optimization.  
✅ **PyMuPDF integration** for accurate **PDF text extraction**.  
✅ **Real-time summarization UI** built with **Streamlit**.  

## Project Structure
- **`app.py`** – Streamlit-based web interface for summarization.  
- **`Finetuning.ipynb`** – Jupyter Notebook for fine-tuning GEMMA with LoRA.  

## Installation
1. Clone the repository:  
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run app.py
```
