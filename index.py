import json
from bs4 import BeautifulSoup
import re
from pathlib import Path
from nltk.stem import SnowballStemmer
import os
import heapq
import math

doc_count = 0
ps = SnowballStemmer("english")
doc_id_to_url = {}

def parse(file):
    important_text = set()
    text = ""
    url = ""
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract the url from the JSON
    url = data.get("url", "")
    if "content" in data:
        soup = BeautifulSoup(data["content"], "html.parser")
        # Extract words from important tags
        important_text.update(extract_words("title", soup))
        important_text.update(extract_words("h1", soup))
        important_text.update(extract_words("h2", soup))
        important_text.update(extract_words("h3", soup))
        important_text.update(extract_words("b", soup))
        text = soup.get_text()
    return (text, important_text, url)

def extract_words(tag_type, soup):
    words = set()
    tag_lines = soup.find_all(tag_type)
    for tag in tag_lines:
        for word in tag.text.split():
            words.add(ps.stem(word))
    return words

def index(json_file, invertedIndex):
    global doc_count 
    IMPORTANT_MULTIPLIER = 5
    tf_vals = {}
    freq = {}
    important_text = set()

    doc_count += 1
    doc_id = doc_count
    text, important_text, url = parse(json_file)
    # Save the URL in the mapping.
    doc_id_to_url[doc_id] = url
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    length = len(tokens)

    for token in tokens:
        stem = ps.stem(token)
        freq[stem] = freq.get(stem, 0) + 1

    for word, count in freq.items():
        tf_vals[word] = count / length
        if word in important_text:
            tf_vals[word] *= IMPORTANT_MULTIPLIER
        
    for word, tf in tf_vals.items():
        if word not in invertedIndex:
            invertedIndex[word] = []
        invertedIndex[word].append((doc_id, tf))

def buildIndex(dir):
    """
    Returns a list of all the partial indexes created that contain a threshold amount of documents each.
    """
    SAVE_THRESHOLD = 1000
    max_test_threshold = 4

    invertedIndex = {}
    file_counter = 0
    directory = Path(dir)
    partial_indexes = []

    for json_file in directory.rglob("*.json"):
        # TESTING
        # if file_counter == max_test_threshold:
        #     break
        #TESTING
        print("Indexing", json_file)
        index(json_file, invertedIndex)
        file_counter += 1

        if file_counter % SAVE_THRESHOLD == 0:
            print("----------------------- WRITING TO DISK ---------------------------")
            partial_filename = f"indexes/partial_index_{len(partial_indexes)}.txt"
            save_index_to_disk(invertedIndex, partial_filename)
            partial_indexes.append(partial_filename)
            invertedIndex.clear()
            print("------------------------- CLEARING ------------------------------")
    if invertedIndex:
        partial_filename = f"indexes/partial_index_{len(partial_indexes)}.txt"
        save_index_to_disk(invertedIndex, partial_filename)
        partial_indexes.append(partial_filename)
        invertedIndex.clear()
    print(f"Total files processed: {file_counter}")
    return partial_indexes

def save_index_to_disk(index, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for token in sorted(index.keys()):
            postings = index[token]
            postings_str = ", ".join(f"{doc_id}:{tf}" for doc_id, tf in postings)
            f.write(f"{token}: {postings_str}\n")
    print(f"Saved index to {filename}")

def merge_indexes(partial_indexes, output_filename, offset_filename):
    files = []
    for filename in partial_indexes:
        f = open(filename, 'r', encoding='utf-8')
        files.append(f)
    
    heap = []
    for i, f in enumerate(files):
        line = f.readline()
        if line:
            token, postings_str = line.split(": ", 1)
            heapq.heappush(heap, (token, postings_str, i))
    
    with open(output_filename, 'w', encoding='utf-8') as fout:
        while heap:
            current_token, postings_str, file_idx = heapq.heappop(heap)
            merged_postings = []
            
            def parse_postings(p_str):
                postings = [p for p in p_str.split(", ") if p]
                result = []
                for posting in postings:
                    try:
                        doc_id, tf = posting.split(":", 1)
                        result.append((doc_id, tf))
                    except ValueError:
                        continue
                return result

            merged_postings.extend(parse_postings(postings_str))
            
            while heap and heap[0][0] == current_token:
                _, next_postings_str, next_file_idx = heapq.heappop(heap)
                merged_postings.extend(parse_postings(next_postings_str))
                next_line = files[next_file_idx].readline()
                if next_line:
                    next_token, next_postings_str = next_line.split(": ", 1)
                    heapq.heappush(heap, (next_token, next_postings_str, next_file_idx))

            merged_postings_str = ", ".join(f"{doc_id}:{tf}" for doc_id, tf in merged_postings)
            fout.write(f"{current_token}: {merged_postings_str}\n")
            
            next_line = files[file_idx].readline()
            if next_line:
                next_line = next_line.strip()
                next_token, next_postings_str = next_line.split(": ", 1)
                heapq.heappush(heap, (next_token, next_postings_str, file_idx))

    for f in files:
        f.close()

    print(f"Merged index saved to {output_filename}")

def calc_idfs(final_index_filename):
    global doc_count  
    total_docs = doc_count 
    temp_path = final_index_filename + ".tmp"
    
    with open(final_index_filename, 'r', encoding='utf-8') as fin, \
         open(temp_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            try:
                token, postings_str = line.split(": ", 1)
            except ValueError:
                continue
            postings = [p for p in postings_str.split(", ") if p]
            df = len(postings)
            idf = math.log(total_docs / df)
            fout.write(f"{token}: {postings_str} ; {idf}\n")
    
    os.replace(temp_path, final_index_filename)
    print(f"Updated final index with IDFs in {final_index_filename}")

def create_offset_index(final_index_filename, offset_filename):
    offset_index = {}
    with open(final_index_filename, 'r', encoding='utf-8') as fin:
        while True:
            offset = fin.tell()
            line = fin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                token, _ = line.split(": ", 1)
            except ValueError:
                continue
            offset_index[token] = offset
    with open(offset_filename, 'w', encoding='utf-8') as f_offset:
        json.dump(offset_index, f_offset)
    print(f"Offset index saved to {offset_filename}")

def main():
    # directory = Path("/Users/sarah/Downloads/DEV")
    directory = Path("/home/ssha2/cs121/cs121_A3/DEV")
    partial_indexes = buildIndex(directory)
    
    merge_indexes(partial_indexes, "final_index.txt", "final_offset.json")
    calc_idfs("final_index.txt")
    create_offset_index("final_index.txt", "final_offset.json")
    
    with open('doc_mapping.json', 'w', encoding='utf-8') as map_file:
        json.dump(doc_id_to_url, map_file)
    
    final_index_size = os.path.getsize("final_index.txt")
    print(f"Final index size: {final_index_size} bytes")

if __name__ == "__main__":
    main()
