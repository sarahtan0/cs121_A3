import json
import mmap
import time

with open('final_offset.json', 'r') as o_file:
    offset_index = json.load(o_file)

with open('final_index.txt', 'r') as i_file:
    mem_map = mmap.mmap(i_file.fileno(), 0, access=mmap.ACCESS_READ)

def retrieve(word):
    if word not in offset_index:
        return ""
    word_offset = int(offset_index[word])
    
    mem_map.seek(word_offset)
    line = mem_map.readline().decode('utf-8')  # decode bytes to str
    return line

if __name__ == "__main__":
    start_time = time.perf_counter()
    result = retrieve("ace")
    elapsed = (time.perf_counter() - start_time) * 1000  # in ms
    print(result)
    print(f"Elapsed time: {elapsed:.2f} ms")
