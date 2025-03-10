import json
import mmap
import time
import re
import math
from collections import Counter
from nltk.stem import SnowballStemmer

# Load the offset index and the document mapping (which now contains URLs),
# and create a memory mapped file for the final index.
with open('final_offset.json', 'r') as o_file:
    offset_index = json.load(o_file)

with open('doc_mapping.json', 'r') as d_file:
    doc_mapping = json.load(d_file)

with open('final_index.txt', 'r') as i_file:
    mem_map = mmap.mmap(i_file.fileno(), 0, access=mmap.ACCESS_READ)

def tokenize_query(query):
    ps = SnowballStemmer("english")   # Using SnowballStemmer for English
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    return [ps.stem(token) for token in tokens]

def retrieve(word):
    if word not in offset_index:
        return ""
    word_offset = int(offset_index[word])
    # print(f"{word} found in offset index at byte {word_offset}")
    mem_map.seek(word_offset)
    line = mem_map.readline().decode('utf-8')  # decode bytes to str
    return line

def parse_index_line(line):
    '''
    Turns the given line into a set of postings with the idf
    '''
    line = line.strip()
    if not line:
        return {}, 0.0
    try:
        token_postings, idf_str = line.split(" ; ")
        # Remove token header; we only need postings.
        _, postings_str = token_postings.split(":", 1)
    except ValueError as e:
        return {}, 0.0

    postings = {}
    if postings_str:
        posting_items = postings_str.split(", ")
        for item in posting_items:
            if item:
                try:
                    doc_id_str, info = item.split(":", 1)
                    doc_id = int(doc_id_str)
                    tf_str, pos_str = info.split(":")
                    tf_val = float(tf_str)
                    positions = set(map(int, pos_str.split("|")))
                    postings[doc_id] = (tf_val, positions)
                except ValueError:
                    continue
    try:
        idf = float(idf_str)
    except ValueError:
        idf = 0.0
    return postings, idf

def compute_doc_scores(query_vector, doc_vectors):
    """
    calc cosine similarity by document
    """
    scores = {}
    # Calculate the norm of the query vector.
    query_norm = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
    for doc_id, doc_vector in doc_vectors.items():
        # Compute dot product of query and document vectors.
        dot_product = sum(query_vector.get(token, 0) * doc_vector.get(token, 0)
                          for token in query_vector)
        # Compute norm of the document vector.
        doc_norm = math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))
        if query_norm == 0 or doc_norm == 0:
            scores[doc_id] = 0.0
        else:
            scores[doc_id] = dot_product / (query_norm * doc_norm)
    return scores

def retrieve_original_document(doc_id):
    """
    Given a document ID, look up its URL using the doc_mapping and return it.
    """
    return doc_mapping.get(str(doc_id))

def search(query):
    results = []
    start_time = time.perf_counter()
    query_tokens = tokenize_query(query)
    query_counts = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    query_positions = {}
    
    # Build the query vector and the document vectors.
    query_vector = {}
    doc_vectors = {}  
    
    #calculate vectors for each query word
    for token, count in query_counts.items():
        line = retrieve(token)
        if not line:
            continue  # Token not found in index.
        postings, idf = parse_index_line(line)

        # For Boolean search, give each query token a weight equal to its idf.
        query_weight = idf
        query_vector[token] = query_weight
        
        # For each document that contains this token, update its vector.
        for doc_id, tf in postings.items():
            doc_weight = tf[0] * idf
            positions = tf[1]
            if doc_id not in doc_vectors:
                doc_vectors[doc_id] = {}
            doc_vectors[doc_id][token] = doc_weight

            if doc_id not in query_positions:
                query_positions[doc_id] = {}
            if token not in query_positions[doc_id]:
                query_positions[doc_id][token] = set()
            query_positions[doc_id][token].update(positions)
            # print(f"POSITIONS OF QUERY {token} IN DOC {doc_id}: {query_positions[doc_id][token]}")

    # Enforce Boolean AND: only keep documents that contain all query tokens.
    common_docs = set(doc_vectors.keys())
    prev_token = None

    for token in query_vector:
        docs_with_token = set({doc_id for doc_id, vec in doc_vectors.items() if token in vec})
        # print(f"CURR TOKEN: {token}")

        #11894

        if prev_token is not None:
            filtered_docs = set()
            # print(F"PREV TOKEN: {prev_token}")
            #loops through docs that have the current token, should check if the previous token exists and THEN if there exists one that's 1 pos before
            for doc_id in common_docs:
                if prev_token in query_positions[doc_id] and token in query_positions[doc_id]:
                    prev_positions = query_positions[doc_id][prev_token]
                    curr_positions = query_positions[doc_id][token]

                    shorter, longer = (prev_positions, curr_positions) if len(prev_positions) < len(curr_positions) else (curr_positions, prev_positions)
                    if (shorter == prev_positions and any(pos + 1 in longer for pos in shorter)) or (shorter == curr_positions and any(pos - 1 in longer for pos in shorter)):
                        # print(f"IN DOC {doc_id}: {token} POS = {curr_positions}\n{prev_token} POS = {prev_positions}")
                        filtered_docs.add(doc_id)
            # print(f"ADDING {filtered_docs} TO COMMON DOCS")
            common_docs &= filtered_docs
        else:
            common_docs &= docs_with_token
        prev_token = token
        # print(f"DOCS WITH {prev_token} AND {token}: {common_docs}")
    doc_vectors = {doc_id: vec for doc_id, vec in doc_vectors.items() if doc_id in common_docs}
    
    # Compute cosine similarity scores.
    doc_scores = compute_doc_scores(query_vector, doc_vectors)

    # Retrieve the top 5 documents based on the score.
    top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for doc_id, score in top_docs:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
        url = retrieve_original_document(doc_id)
        print("URL:", url)
        print("-" * 50)
        results.append(url)
    
    elapsed = (time.perf_counter() - start_time) * 1000  # in milliseconds
    print(f"Elapsed time: {elapsed:.2f} ms")
    # summaries = generate_all_summaries(top_docs)
    # return [results, summaries]
    return results

if __name__ == "__main__":
    query = input("Type your query and press Enter: ")
    search(query)