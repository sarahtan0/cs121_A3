import json
from bs4 import BeautifulSoup
import re
from pathlib import Path
from nltk.stem import PorterStemmer

invertedIndex: [str, tuple[int, float]]= {}
doc_count = 0

#TODO implement stemming

#tokenize all the words in the page
#add all words to inverted index and posting(docID, tf)
def remove_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def parse(file):
    text = ""
    with open(file, 'r') as file:
        data = json.load(file)

    if "content" in data:
        text = remove_html(data["content"])
    
    return text

def index(json):
    global doc_count 
    global invertedIndex
    tf_vals = {}
    freq = {}

    doc_count += 1
    doc_id = doc_count
    text = parse(json)
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    length = len(tokens)

    for token in tokens:
        if len(token) > 1:
            freq[token] = freq.get(token, 0) + 1
    
    for word, count in freq.items():
        tf_vals[word] = count/length
        
    for word, tf in tf_vals.items():
        invertedIndex[word] = (doc_id, tf)

def buildIndex(dir):
    directory = Path(dir)
    # json_files = list(directory.rglob("*.json"))  # Get all JSON files
    global links
    for json_file in directory.rglob("*.json"):
        print(json_file)

def main():
    dir = "/home/tans9/121_assignment3/cs121_A3/DEV"
    buildIndex(dir)
    for word, vals in invertedIndex.items():
        print(word, vals)

if __name__ == "__main__":
    main()
    
