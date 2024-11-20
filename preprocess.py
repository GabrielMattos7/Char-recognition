import requests
from bs4 import BeautifulSoup
import os
import random
import re
import unicodedata

def remove_accents(text):

    normalized = unicodedata.normalize('NFD', text)

    without_accents = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')

    return without_accents

def filter_non_english(text):
    # Keep only English letters (A-Z, a-z), spaces, and optionally punctuation
    return re.sub(r'[^A-Za-z0-9 .,!?\'\"\-()]', '', text)

def get_random_wikipedia_article():
    response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
    return response.url

def extract_main_text(url, num_words):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.select('#mw-content-text p')
    text = ' '.join([p.get_text() for p in paragraphs])
    text = ''.join(char for char in text if char.isprintable())  # Remove non-printable characters
    text = filter_non_english(text)  # Remove accents and other alfabets letters
    words = text.split()[:num_words]
    text = ' '.join(words) #now we are going to upper cases to 
    # text = re.sub(r'[^a-z ]', '', text).lower()
    return text

def save_to_file(text, index):
    os.makedirs('text', exist_ok=True)
    filename = f"text/{index:03d}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def generate_strings(N, L, Chars):
    strings = []
    for i in range(1, N + 1):
        length = random.randint(50, L)
        string = ''.join(random.choice(Chars) if i % 20 != 0 else ''.join('\n') for i in range(1,length+1,1))
        string+="\n"
        save_to_file(string, i) 
        strings.append(string)
    return strings

def main(num_pages, words_per_page):
    for i in range(1, num_pages + 1):
        url = get_random_wikipedia_article()
        text = extract_main_text(url, words_per_page)
        save_to_file(text, i)
        print(f"Saved file {i:03d}.txt")


classes = f" \"\ ` ' ! ? / ; . , ( ) A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ")

if __name__ == "__main__":
    num_pages = 10
    words_per_page = 100 
    main(num_pages, words_per_page)
    # print(generate_strings(50, 150, classes))
