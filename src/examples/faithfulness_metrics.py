from src.utils import (
    initialize,
    load_file,
    insert_or_fetch_embeddings,
    chunk_data,
    jarvis,
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


query = "What are the Republican Party's commitments to increasing federal funding for public education?"
model = "gpt-3.5-turbo"
workflow = "rag"
# qury openai for answer
answer = jarvis(query, model, workflow, vector_store)
actual_output = answer["result"]
print(f"\n\n{'=' * 25}LLM RESPONSE{'=' * 25}")
print(actual_output)


metrics = LLMMetrics(model)
metrics.faithfulness_metric(
    query, answer, [source.page_content for source in answer["source_documents"]]
)
print(metrics.metrics_data["faith"])
write_to_file(
    actual_output,
    query,
    metrics.metrics_data["faith"],
    r"data\output_files\faith_metric.json",
)
