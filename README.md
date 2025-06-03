# 💸 Finetuned LLM for Financial Sentiment Analysis using Gemma-2B + LoRA

> 🚀 An end-to-end project where a 2-billion-parameter language model (Gemma-2B) is fine-tuned on financial sentiment data using LoRA and deployed on Hugging Face Spaces with an intuitive Gradio UI.

![HF badge](https://img.shields.io/badge/model-Gemma--2B-green) ![Gradio badge](https://img.shields.io/badge/UI-Gradio-blue) ![Huggingface badge](https://img.shields.io/badge/hosted%20on-HuggingFace-orange)

---

## 🧠 Introduction

In financial markets, **sentiment matters** — from central bank statements to earnings announcements, the tone of the language can move markets. This project builds a **language model that reads financial news and classifies the sentiment** into **positive, neutral, or negative**.

Built with Hugging Face’s `transformers`, Google's `Gemma`, and parameter-efficient fine-tuning (LoRA), this model is designed for **cost-effective training and real-time inference**.

---

## 📚 Background

### 📌 What is Gemma?

> **Gemma-2B** is a lightweight yet powerful large language model (LLM) from Google optimized for conversational tasks and reasoning. With 2 billion parameters, it's ideal for fine-tuning and on-device inference.

### 🛠 What We Did

- Used **LoRA (Low-Rank Adaptation)** to fine-tune only a fraction of model parameters.
- Trained on custom sentiment-labelled financial text using a prompt-based format.
- Merged adapter weights and pushed the full model to [🤗 Hugging Face Hub](https://huggingface.co).
- Deployed an interactive Gradio app on Hugging Face Spaces.

---

## 🧰 Tools & Technologies Used

| Tool / Platform            | Purpose                              |
|---------------------------|--------------------------------------|
| 🐍 Python                 | Core programming language            |
| 🧠 Google Gemma-2B Model | Base LLM                             |
| 💻 Vertex AI Workbench    | Training & compute                   |
| 📓 Google Colab           | Data exploration, pretraining setup  |
| 🤗 Hugging Face Hub       | Model hosting and sharing            |
| 🎛 Gradio                 | Web UI for text input and sentiment  |

---

## 🖼 Deployed UI Snapshot

![App Screenshot](https://huggingface.co/spaces/dushyant22/Financial_Sentiment_Analysis/resolve/main/demo_ui.png) <!-- Replace this with actual hosted screenshot if needed -->

🔗 **Live Demo**: [Click here to try the app](https://huggingface.co/spaces/dushyant22/Financial_Sentiment_Analysis)

---

## 📈 Project Workflow

```mermaid
flowchart TD
    A[Raw Financial Text] --> B[Instruction-Response Dataset]
    B --> C[LoRA Fine-tuning on Gemma-2B]
    C --> D[Model Evaluation]
    D --> E[Merge and Push to HF]
    E --> F[Deploy with Gradio in Spaces]
    F --> G[User Enters Financial News]
    G --> H[Model Predicts Sentiment]
