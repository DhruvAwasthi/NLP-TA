import chromadb
import csv
chromadb_client = chromadb.Client()
chromadb_collection = chromadb_client.get_or_create_collection("data_sources")
openai_key = ""
from openai import OpenAI
openai_client = OpenAI(api_key=openai_key)
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        start_index = i
        end_index = i + chunk_size + overlap
        if end_index < len(text):
            while end_index > i and text[end_index] != ' ':
                end_index -= 1
        while start_index < len(text) and start_index < end_index and text[start_index] != ' ':
            start_index += 1
        chunks.append(text[start_index:end_index].strip())
        i = end_index - overlap
    return chunks
def read_csv(file_path):
    data_array = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            sno, chapter_title, chapter_url, chapter_content = row
            chunks = chunk_text(chapter_content)
            data_array.append({
                'sno': sno,
                'chapter_title': chapter_title,
                'chapter_url': chapter_url,
                'chunks': chunks
            })
    return data_array
file_path = 'naval(Building RAG based AI Apps).csv'
result_array = read_csv(file_path)
for entry in result_array:
  for index, chunk in enumerate(entry['chunks']):
    chromadb_collection.add(ids=[f"{entry['sno']}_{index}"], documents=[chunk], metadatas=[{ **{'url': entry['chapter_url'],
                                                                               'title':entry['chapter_title']}}])
query = "How to make money. give 5 points"
def get_top_k_context(query, n_results):
  top_k_context = chromadb_collection.query(
    query_texts = [query],
    n_results = n_results)
  converted_data = {}
  for i, doc_id in enumerate(top_k_context["ids"][0]):
      metadata = top_k_context["metadatas"][0][i]
      document = top_k_context["documents"][0][i]
      converted_data[doc_id] = {
          "chapter_name": metadata["title"],
          "url": metadata["url"],
          "snippet": document
      }
  top_k_context = converted_data
  return top_k_context
top_k_context = get_top_k_context(query,5)
def generate_response(query, top_k_context):
    prompt = f"Given is a query {query}. Answer the query from snippets from a book  {top_k_context}. also include reference like url, title etc. In your answer dont mention word snippets, just answer like you have read the book and providing the expert answer"
    messages = [
        {"role": "system", "content": "You are a helpful assistant who help other find and synthesise insights and able to answer user questions from the given book snippets."},
        {"role": "user", "content": prompt}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",  #gpt-4-turbo
        messages=messages,
        max_tokens=1024,
        n=1,
        stop=None
    )
    return response.choices[0].message.content
generate_response(query,top_k_context)