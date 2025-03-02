import json
import mmap
import time
import re
import math
from collections import Counter
from nltk.stem import PorterStemmer

# Load the offset index and the document mapping (which now contains URLs),
# and create a memory mapped file for the final index.
with open('final_offset.json', 'r') as o_file:
    offset_index = json.load(o_file)

with open('doc_mapping.json', 'r') as d_file:
    doc_mapping = json.load(d_file)

with open('final_index.txt', 'r') as i_file:
    mem_map = mmap.mmap(i_file.fileno(), 0, access=mmap.ACCESS_READ)

def tokenize_query(query):
    ps = PorterStemmer()   
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    return [ps.stem(token) for token in tokens]

def retrieve(word):
    if word not in offset_index:
        return ""
    word_offset = int(offset_index[word])
    mem_map.seek(word_offset)
    line = mem_map.readline().decode('utf-8')  # decode bytes to str
    return line

def parse_index_line(line):
    line = line.strip()
    if not line:
        return {}, 0.0
    try:
        token_postings, idf_str = line.split(" ; ")
        # Remove token header; we only need postings.
        _, postings_str = token_postings.split(": ", 1)
    except ValueError:
        return {}, 0.0

    postings = {}
    if postings_str:
        posting_items = postings_str.split(", ")
        for item in posting_items:
            if item:
                try:
                    doc_id_str, tf_str = item.split(":")
                    doc_id = int(doc_id_str)
                    tf_val = float(tf_str)
                    postings[doc_id] = tf_val
                except ValueError:
                    continue
    try:
        idf = float(idf_str)
    except ValueError:
        idf = 0.0
    return postings, idf

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors represented as dictionaries.
    """
    dot_product = sum(vec1.get(token, 0) * vec2.get(token, 0)
                      for token in set(vec1.keys()).intersection(vec2.keys()))
    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def compute_doc_scores(query_vector, doc_vectors):
    """
    Given a query vector and a dictionary of document vectors,
    compute the cosine similarity for each document.
    Returns a dictionary mapping doc_id to cosine similarity score.
    """
    scores = {}
    for doc_id, doc_vector in doc_vectors.items():
        scores[doc_id] = cosine_similarity(query_vector, doc_vector)
    return scores

def retrieve_original_document(doc_id):
    """
    Given a document ID, look up its URL using the doc_mapping and return it.
    """
    # The keys in the mapping might be strings, so convert the doc_id to a string.
    return doc_mapping.get(str(doc_id))

if __name__ == "__main__":
    query = input("Type your query and press Enter: ")
    start_time = time.perf_counter()
    query_tokens = tokenize_query(query)
    query_counts = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    total_query_terms = sum(query_counts.values())
    
    # Build the query vector and the document vectors.
    query_vector = {}
    doc_vectors = {}  
    
    for token, count in query_counts.items():
        line = retrieve(token)
        if not line:
            continue  # Token not found in index.
        postings, idf = parse_index_line(line)
        # Compute normalized query term frequency and its weight.
        query_tf = count / total_query_terms
        query_weight = query_tf * idf
        query_vector[token] = query_weight
        
        # For each document that has this token, update its vector.
        for doc_id, tf in postings.items():
            doc_weight = tf * idf
            if doc_id not in doc_vectors:
                doc_vectors[doc_id] = {}
            doc_vectors[doc_id][token] = doc_weight

    # Compute cosine similarity scores.
    doc_scores = compute_doc_scores(query_vector, doc_vectors)

    # Retrieve the top 5 documents based on cosine similarity.
    top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for doc_id, score in top_docs:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
        url = retrieve_original_document(doc_id)
        if url:
            print("URL:", url)
        else:
            print("URL not found for doc_id", doc_id)
        print("-" * 50)
    
    elapsed = (time.perf_counter() - start_time) * 1000  # in milliseconds
    print(f"Elapsed time: {elapsed:.2f} ms")
