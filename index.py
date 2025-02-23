import json
from bs4 import BeautifulSoup
import re
from pathlib import Path
from nltk.stem import PorterStemmer
import os

doc_count = 0
ps = PorterStemmer()

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

def index(json, invertedIndex):
    global doc_count 
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

def buildIndex(dir):
    SAVE_THRESHOLD = 1000
    invertedIndex: [str, tuple[int, float]]= {}
    file_counter = 0
    directory = Path(dir)
    partial_indexes = []

    for json_file in directory.rglob("*.json"):
        print("indexing",json_file)
        index(json_file, invertedIndex)
        file_counter+=1

        if file_counter % SAVE_THRESHOLD == 0:
            print("----------------------- WRITING TO DISK ---------------------------")
            partial_filename = f"partial_index_{len(partial_indexes)}.txt"
            save_index_to_disk(invertedIndex, partial_filename)
            partial_indexes.append(partial_filename)
            invertedIndex.clear()  # removes all elements from a dictionary
            print("------------------------- CLEARING ------------------------------")
    # Save any remaining documents if the last batch didn't reach the threshold.
    if invertedIndex:
        partial_filename = f"partial_index_{len(partial_indexes)}.txt"
        save_index_to_disk(invertedIndex, partial_filename)
        partial_indexes.append(partial_filename)
        invertedIndex.clear()
    print(f"Total files processed: {file_counter}")
    return partial_indexes

def save_index_to_disk(index, filename):
    new_token = ""
    with open(filename, 'w', encoding='utf-8') as f:
        for token in index.keys():
            f.write(f"{token}: ")
            postings = index[token]
            for posting in postings:
                f.write(f"{posting[0]}:{posting[1]}, ")
            f.write("\n")
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
    print(f"Unique tokens: {len(merged_index)}")
    print(f"Merged index saved to {output_filename}")

def main():

    # large file: mondego_ics_uci_edu/7e7ab052f410de3ff187976df4a61e51d50faea14edba3e6d24c15496832dcb7.json

    # """
    # SIMPLE MAIN IMPLEMENTATIN
    # """
    # dir = "/home/tans9/121_assignment3/cs121_A3/DEV"
    # inverted = buildIndex(dir)
    # print(f'Unique count: {len(inverted)}')
    # print(f'Total document count: {doc_count}')

    """
    NEW IMPLEMENTATION OF MAIN
    """
    directory = Path("/home/tans9/121_assignment3/cs121_A3/DEV")

    # Process each json file in the directory
    partial_indexes = buildIndex(directory)    
    # Merge all the partial indexes into the final index
    merge_indexes(partial_indexes, "final_index.txt")

    # Prints final index size using os module
    final_index_size = os.path.getsize("final_index.txt")
    print(f"Final index size: {final_index_size} bytes")

if __name__ == "__main__":
    main()
