import json
from bs4 import BeautifulSoup
import re
from pathlib import Path
from nltk.stem import PorterStemmer

invertedIndex: [str, tuple[int, float]]= {}
doc_count = 0
ps = PorterStemmer()
indexed_docs = 0

#tokenize all the words in the page
#add all words to inverted index and posting(docID, tf)

def parse(file):
    text = ""
    with open(file, 'r') as file:
        data = json.load(file)

    if "content" in data:
        #excludes xml
        text = BeautifulSoup(data["content"], "html.parser").get_text()
        
    return text

def index(json):
    global doc_count 
    global invertedIndex
    tf_vals = {}
    freq = {}

    doc_count += 1
    doc_id = doc_count
    text = parse(json)
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    length = len(tokens)

    for token in tokens:
        global ps
        #found in https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
        stem = ps.stem(token)
        freq[stem] = freq.get(stem, 0) + 1
    
    for word, count in freq.items():
        tf_vals[word] = count/length
        
    for word, tf in tf_vals.items():
        if word not in invertedIndex:
            invertedIndex[word] = []
        invertedIndex[word].append((doc_id, tf))
    
    return invertedIndex

def buildIndex(dir):
    global links
    global indexed_docs
    directory = Path(dir)
    for json_file in directory.rglob("*.json"):
        index(json_file)
        indexed_docs+=1
        # print(f"{indexed_docs}/55393")
    return invertedIndex

def save_index_to_disk(index, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for token in index.keys():
            doc_id, tf = index[token]
            f.write(f"{token}: {doc_id}:{tf}\n")
    print(f"Saved index to {filename}")

def merge_indexes(partial_indexes, output_filename):
    merged_index = {}
    
    for filename in partial_indexes:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split into token and postings part
                try:
                    token, postings_str = line.split(": ", 1)
                except ValueError:
                    continue  # skip malformed lines
                # Each posting is separated by ", " if multiple exist
                postings = postings_str.split(", ")
                for posting in postings:
                    try:
                        doc_id_str, tf_str = posting.split(":", 1)
                        doc_id = int(doc_id_str)
                        tf = float(tf_str)
                    except ValueError:
                        continue  # skip malformed postings
                    if token not in merged_index:
                        merged_index[token] = []
                    merged_index[token].append((doc_id, tf))
    
    # Write the merged index to the output file in sorted token order.
    with open(output_filename, 'w', encoding='utf-8') as f:
        for token in sorted(merged_index.keys()):
            postings_str = ", ".join(f"{doc_id}:{tf}" for doc_id, tf in merged_index[token])
            f.write(f"{token}: {postings_str}\n")
    print(f"Merged index saved to {output_filename}")

def main():
    unique_token_count = 0

    dir = "/home/tans9/121_assignment3/cs121_A3/DEV"
    inverted = buildIndex(dir)
    for word, vals in inverted.items():
        unique_token_count += 1
    # print(f'Unique count: {unique_token_count}')
    # print(f'Total document count: {doc_count}')

if __name__ == "__main__":
    main()
