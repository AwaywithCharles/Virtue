"""
Author: Charles Bostwick
Website: www.AwaywithCharles.com
GitHub: https://github.com/AwaywithCharles
License: MIT
"""

import string

# Read in the text data from the file
with open("python_wikipedia.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Remove punctuation and convert all characters to lowercase
text = text.translate(str.maketrans("", "", string.punctuation))
text = text.lower()

# Split the text into individual tokens
tokens = text.split()

# Print the first 10 tokens
print(tokens[:10])
