"""
Author: Charles Bostwick
Website: www.AwaywithCharles.com
GitHub: https://github.com/AwaywithCharles
License: MIT
"""

import requests
from bs4 import BeautifulSoup

# URL of the web page to scrape
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

# Send a GET request to the URL
response = requests.get(url)

# Use BeautifulSoup to parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Find all the paragraph tags on the page and extract their text content
paragraphs = [p.get_text() for p in soup.find_all("p")]

# Save the text content to a file
with open("python_wikipedia.txt", "w", encoding="utf-8") as f:
    for paragraph in paragraphs:
        f.write(paragraph + "\n")
