import json
from bs4 import BeautifulSoup
import re
from pathlib import Path
from nltk.stem import PorterStemmer

invertedIndex: [str, tuple[int, float]]= {}
doc_count = 0
ps = PorterStemmer()

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
        global ps
        #found in https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
        stem = ps.stem(token)
        freq[stem] = freq.get(stem, 0) + 1
    
    for word, count in freq.items():
        tf_vals[word] = count/length
        
    for word, tf in tf_vals.items():
        invertedIndex[word] = (doc_id, tf)
    
    return invertedIndex

def buildIndex(dir):
    directory = Path(dir)
    global links
    # for json_file in directory.rglob("*.json"):
    #     print(json_file)

def main():
    unique_token_count = 0

    dir = "/home/tans9/121_assignment3/cs121_A3/8ef6d99d9f9264fc84514cdd2e680d35843785310331e1db4bbd06dd2b8eda9b.json"
    inverted = index(dir)
    for word, vals in inverted.items():
        unique_token_count += 1
        print(word, vals)
    print("Unique token count: {unique_token_count}")

if __name__ == "__main__":
    main()
    
