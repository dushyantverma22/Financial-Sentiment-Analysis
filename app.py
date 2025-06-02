# prompt: create a code using gardio liberary, write in the format of app.py for financial analysis that i can directly use in the hugging space

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Define the directory where you saved the model and tokenizer
# This should be the directory you created when saving the model in Colab
SAVE_DIRECTORY = "./finetuned_gemma" # Adjust this path if you saved it elsewhere

# Load the fine-tuned model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIRECTORY)
    # Ensure model is loaded onto the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIRECTORY).to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model and tokenizer loaded successfully from {SAVE_DIRECTORY}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    tokenizer = None
    model = None
    device = "cpu" # Set device to cpu if loading fails

def analyze_sentiment_gradio(text):
    """
    Analyzes the sentiment of the given text using the loaded model
    for Gradio interface.
    """
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded. Please ensure the model directory exists and contains the necessary files."

    if not text:
        return "Please enter some financial text to analyze."

    # Format input as a chat or prompt template
    input_text = f"### Financial News:\n{text}\n### Sentiment:"

    try:
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50, # You can adjust this
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # deterministic output
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the sentiment
        if "### Sentiment:" in generated_text:
            return generated_text.split("### Sentiment:")[-1].strip()
        else:
            # Fallback if the expected format is not found
            # You might want to refine this based on your model's output
            return generated_text.strip()

    except Exception as e:
        return f"An error occurred during analysis: {e}"

# Create the Gradio interface
if model is not None and tokenizer is not None:
    gr.Interface(
        fn=analyze_sentiment_gradio,
        inputs=gr.Textbox(lines=5, label="Enter Financial News Text"),
        outputs=gr.Textbox(label="Sentiment"),
        title="Financial News Sentiment Analysis",
        description="Analyze the sentiment of financial news headlines and sentences using a fine-tuned language model."
    ).launch()
else:
    print("Gradio interface could not be launched because the model or tokenizer failed to load.")

