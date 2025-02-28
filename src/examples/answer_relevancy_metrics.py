from src.utils import (
    initialize,
    load_file,
    insert_or_fetch_embeddings,
    chunk_data,
    jarvis,
    create_analysis_document,
    write_to_file,
)
from metrics.llmtest import LLMMetrics

# loads the keys for openai and pinecone
initialize()

file = r"data\input_files\2024_us_memo.pdf"
data = load_file(file)

# chunk data recursive text splitter
chunks = chunk_data(data)

print(f"chunks length: {len(chunks)}")


# pinecone index name
index_name = "myfirstrag"

# insert embeddings into vector db pinecone
vector_store = insert_or_fetch_embeddings(index_name, chunks)


query = "How does the document characterize the Democrats' approach to immigration and border security?"
# query = "What are the proposed changes to the sports system by the toxicology Party?"
model = "gpt-3.5-turbo"
workflow = "rag"
# qury openai for answer
answer = jarvis(query, model, workflow, vector_store)
actual_output = answer["result"]
print(f"\n\n{'=' * 25}LLM RESPONSE{'=' * 25}")
print(actual_output)


metrics = LLMMetrics(model)
metrics.answer_relevancy_metric(query, answer)
print(metrics.metrics_data["answer_relevancy"])
write_to_file(
    actual_output,
    query,
    metrics.metrics_data["answer_relevancy"],
    r"data\output_files\answer_relevancy.json",
)
