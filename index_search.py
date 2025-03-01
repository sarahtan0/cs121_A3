import json
import mmap
import time
import re
import math
from collections import Counter
from nltk.stem import PorterStemmer

# Load offset index and create memory mapped file for final index.
with open('final_offset.json', 'r') as o_file:
    offset_index = json.load(o_file)

with open('final_index.txt', 'r') as i_file:
    mem_map = mmap.mmap(i_file.fileno(), 0, access=mmap.ACCESS_READ)

def tokenize_query(query):
    ps = PorterStemmer()   
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens

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
        # remove token. only need postings
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
    # Dot product for common tokens.
    dot_product = sum(vec1.get(token, 0) * vec2.get(token, 0) for token in set(vec1.keys()).intersection(vec2.keys()))
    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def compute_doc_scores(query_vector, doc_vectors):
    """
    calculates cosine similarity based on query vector and document vectors
    """
    scores = {}
    for doc_id, doc_vector in doc_vectors.items():
        scores[doc_id] = cosine_similarity(query_vector, doc_vector)
    return scores

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    # tokenize query.
    query = "masters of software engineering"
    query_tokens = tokenize_query(query)
    query_counts = Counter(query_tokens)
    total_query_terms = sum(query_counts.values())
    
    # Build the query vector and the document vectors.
    # query_vector: token -> (query tf * idf)
    # doc_vectors: doc_id -> { token: (doc tf * idf) }
    query_vector = {}
    doc_vectors = {}  
    
    for token, count in query_counts.items():
        line = retrieve(token)
        if not line:
            continue  # token not in index
        postings, idf = parse_index_line(line)
        # Compute normalized query term frequency and its weight.
        query_tf = count / total_query_terms
        query_weight = query_tf * idf
        query_vector[token] = query_weight
        
        # Update each document's vector with the token weight.
        for doc_id, tf in postings.items():
            doc_weight = tf * idf
            if doc_id not in doc_vectors:
                doc_vectors[doc_id] = {}
            doc_vectors[doc_id][token] = doc_weight

    # calc all cosine similarity scores
    doc_scores = compute_doc_scores(query_vector, doc_vectors)

    # Retrieve the top x documents based on cosine similarity.
    top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10] # <-- how many docs to retreive, no limit runs too slow ~900ms
    
    for doc_id, score in top_docs:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
    
    elapsed = (time.perf_counter() - start_time) * 1000  # in milliseconds
    print(f"Elapsed time: {elapsed:.2f} ms")
