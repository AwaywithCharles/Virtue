"""
Author: Charles Bostwick
Website: www.AwaywithCharles.com
GitHub: https://github.com/AwaywithCharles
License: MIT
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Read in the preprocessed text data from the file
with open("preprocessed_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encode the text using the GPT-2 tokenizer
input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

# Generate text using the GPT-2 model
output = model.generate(input_ids, max_length=50, num_return_sequences=5, no_repeat_ngram_size=2, early_stopping=True)

# Decode the generated text using the GPT-2 tokenizer
generated_text = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(output))]

# Print the generated text
print(generated_text)

