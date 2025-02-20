import json
from bs4 import BeautifulSoup
import re

#inverted index contains 
invertedIndex: [str, tuple[int, float]]= {}

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

# def main():
#     parse("8ef6d99d9f9264fc84514cdd2e680d35843785310331e1db4bbd06dd2b8eda9b.json")

# if __name__ == "__main__":
#     main()
    
