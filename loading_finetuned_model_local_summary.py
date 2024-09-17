from transformers import BartTokenizer, BartForConditionalGeneration

# Path to the directory where the model is saved
model_path = 'bart_finetuned (2)'  # Replace with the model path you download from huggingface link provided

# Load the model and tokenizer
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Function to summarize text
def summarize_text_bart_finetuned(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example dummy text
dummy_text = """
The BART model, developed by Facebook AI, is a powerful model for sequence-to-sequence tasks. 
It is based on a transformer architecture and combines bidirectional and autoregressive transformers. 
BART is particularly effective for text generation tasks, such as summarization and translation, 
by leveraging both the encoder and decoder components.
"""

# Summarize the dummy text
summary = summarize_text(dummy_text)
print("Original Text:")
print(dummy_text)
print("\nSummary:")
print(summary)
