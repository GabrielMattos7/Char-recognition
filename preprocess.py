import requests
from bs4 import BeautifulSoup
import os
import random

def get_random_wikipedia_article():
    response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
    return response.url

def extract_main_text(url, num_words):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.select('#mw-content-text p')
    text = ' '.join([p.get_text() for p in paragraphs])
    words = text.split()[:num_words]
    return ' '.join(words)

def save_to_file(text, index):
    os.makedirs('text', exist_ok=True)
    filename = f"text/{index:03d}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def main(num_pages, words_per_page):
    for i in range(1, num_pages + 1):
        url = get_random_wikipedia_article()
        text = extract_main_text(url, words_per_page)
        save_to_file(text, i)
        print(f"Saved file {i:03d}.txt")

if __name__ == "__main__":
    num_pages = 5  # Specify the number of Wikipedia pages to scrape
    words_per_page = 100  # Specify the number of words to extract from each page
    main(num_pages, words_per_page)

