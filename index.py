import json
from bs4 import BeautifulSoup
import re
from pathlib import Path
from nltk.stem import PorterStemmer
import os
import heapq

doc_count = 0
ps = PorterStemmer()

#tokenize all the words in the page
#add all words to inverted index and posting(docID, tf)

def parse(file):
    important_text = set()
    text = ""
    with open(file, 'r') as file:
        data = json.load(file)

    if "content" in data:
        #excludes xml
        soup = BeautifulSoup(data["content"], "html.parser")
        important_text.update(extract_words("title", soup))
        important_text.update(extract_words("h1", soup))
        important_text.update(extract_words("h2", soup))
        important_text.update(extract_words("h3", soup))
        text = soup.get_text()
        print("IMPORTANT WORDS:",important_text)
    return (text, important_text) #returns all text and a set of important words

def extract_words(type, soup):
    global ps
    words = set()
    tag_lines = soup.find_all(type)
    for tag in tag_lines:
        #tag is a list of lines with the tags INCLUDING tags
        for word in tag.text.split():
            words.add(ps.stem(word))
    return words

def index(json, invertedIndex):
    global doc_count 
    #value that frequency will be multiplied by for important words
    IMPORTANT_MULTIPLIER = 3 
    
    tf_vals = {}
    freq = {}
    important_text = set()

    doc_count += 1
    doc_id = doc_count
    text, important_text = parse(json)
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    length = len(tokens)

    for token in tokens:
        global ps
        #found in https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
        stem = ps.stem(token)
        freq[stem] = freq.get(stem, 0) + 1
    
    for word, count in freq.items():
        tf_vals[word] = count/length
        if word in important_text:
            print(f"{word} found in important text")
            tf_vals[word] *= IMPORTANT_MULTIPLIER
        
    for word, tf in tf_vals.items():
        if word not in invertedIndex:
            invertedIndex[word] = []
        invertedIndex[word].append((doc_id, tf))

def buildIndex(dir):
    """
    Returns a list of all the partial indexes made that contains the save threshold amount of documents worth of tokens each
    """
    SAVE_THRESHOLD = 1000

    max_test_threshold = 4

    invertedIndex: [str, tuple[int, float]]= {}
    file_counter = 0
    directory = Path(dir)
    partial_indexes = []

    for json_file in directory.rglob("*.json"):
        # TESTING
        # if file_counter == max_test_threshold:
        #     break
        #TESTING
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
        for token in sorted(index.keys()):
            f.write(f"{token}: ")
            postings = index[token]
            for posting in postings:
                f.write(f"{posting[0]}:{posting[1]}, ")
            f.write("\n")
    print(f"Saved index to {filename}")

# Referenced https://www.geeksforgeeks.org/merge-k-sorted-arrays-set-2-different-sized-arrays/
def merge_indexes(partial_indexes, output_filename, offset_filename):
    # Open all partial index files
    files = []
    for filename in partial_indexes:
        f = open(filename, 'r', encoding='utf-8')
        files.append(f)
    
    # Each heap element is a tuple: (token, postings_str, file_index)
    heap = []
    for i, f in enumerate(files):
        line = f.readline()
        if line:
            token, postings_str = line.split(": ", 1)
            heapq.heappush(heap, (token, postings_str, i))

    offset_index = {}
    
    with open(output_filename, 'w', encoding='utf-8') as fout:
        while heap:
            # Pop the smallest token from the heap.
            current_token, postings_str, file_idx = heapq.heappop(heap)
            merged_postings = []
            
            # Helper: parse a postings string into a list of (doc_id, tf) tuples.
            def parse_postings(p_str):
                postings = [p for p in p_str.split(", ") if p]  # filter out empty strings
                result = []
                for posting in postings:
                    try:
                        doc_id, tf = posting.split(":", 1)
                        result.append((doc_id, tf))
                    except ValueError:
                        continue  # skip malformed postings
                return result

            # Parse postings from the current popped line.
            merged_postings.extend(parse_postings(postings_str))
            
            # If following elements in the heap have the same token, merge them too.
            while heap and heap[0][0] == current_token:
                _, next_postings_str, next_file_idx = heapq.heappop(heap)
                merged_postings.extend(parse_postings(next_postings_str))
                # After consuming a line from a file, read its next line.
                next_line = files[next_file_idx].readline()
                if next_line:
                    next_token, next_postings_str = next_line.split(": ", 1)
                    heapq.heappush(heap, (next_token, next_postings_str, next_file_idx))

            offset = fout.tell()
            offset_index[current_token] = offset

            # Write the merged postings for the current token to the output file.
            merged_postings_str = ", ".join(f"{doc_id}:{tf}" for doc_id, tf in merged_postings)
            fout.write(f"{current_token}: {merged_postings_str}\n")
            
            # Read the next line from the file where the current token came from.
            next_line = files[file_idx].readline()
            if next_line:
                next_line = next_line.strip()
                next_token, next_postings_str = next_line.split(": ", 1)
                heapq.heappush(heap, (next_token, next_postings_str, file_idx))

    # Close all files.
    for f in files:
        f.close()

    with open(offset_filename, 'w', encoding='utf-8') as f_offset:
        json.dump(offset_index, f_offset)
    
    print(f"Merged index saved to {output_filename}")
    print(f"Offset index saved to {offset_filename}")
def main():

    # large file: mondego_ics_uci_edu/7e7ab052f410de3ff187976df4a61e51d50faea14edba3e6d24c15496832dcb7.json

    """
    NEW IMPLEMENTATION OF MAIN
    """
    # directory = Path("/home/tans9/121_assignment3/cs121_A3/DEV")
    directory = Path("/home/tans9/121_assignment3/cs121_A3/DEV")

    # Process each json file in the directory
    partial_indexes = buildIndex(directory)    
    # Merge all the partial indexes into the final index
    merge_indexes(partial_indexes, "final_new.txt", "final_offset.json")

    # Prints final index size using os module
    final_index_size = os.path.getsize("final_index.txt")
    print(f"Final index size: {final_index_size} bytes")

if __name__ == "__main__":
    main()
