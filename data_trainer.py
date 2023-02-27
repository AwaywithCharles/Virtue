"""
Author: Charles Bostwick
Website: www.AwaywithCharles.com
GitHub: https://github.com/AwaywithCharles
License: MIT
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Read in the preprocessed text data from the file
with open("preprocessed_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encode the text using the GPT-2 tokenizer
encoded_text = tokenizer.encode(text)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Define the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_text
)

# Train the model
trainer.train()
