from metrics.llmtest import LLMMetrics
from src.utils import (
    initialize,
    load_file,
    insert_or_fetch_embeddings,
    check_embedding_cost,
    chunk_data,
    jarvis,
)


def start(file, query, model, workflow, pdf_data=None):
    # loads the keys for openai and pinecone
    initialize()

    if workflow == "rag":

        # loads the the file docx and pdf are supported
        data = load_file(file)

        # chunk data recursive text splitter
        chunks = chunk_data(data)

        print(f"chunks length: {len(chunks)}")

        # embedding cost
        check_embedding_cost(chunks)

        # pinecone index name
        index_name = "myfirstrag"

        # insert embeddings into vector db pinecone
        vector_store = insert_or_fetch_embeddings(index_name, chunks)

        # qury openai for answer
        answer = jarvis(query, model, workflow, vector_store)
    else:
        print(f"{'=' * 25}Chat Workflow:{'=' * 25}")
        answer = jarvis(query, model, workflow, context=pdf_data)
        result = answer

    print(f"{'=' * 25}LLM Response:{'=' * 25}")
    print(answer)
    return answer


file = r"data\input_files\2024_us_memo.pdf"

query = "How does the document portray the impact of the Biden administration on the American economy and society?"
model = "gpt-3.5-turbo"
rag_workflow = "rag"
chat_workflow = "chat"


answer = start(file, query, model, rag_workflow)
print(answer)
# metrics = LLMMetrics(model)
# metrics.g_eval_metrics()
# metrics.bias_metric(query, answer)
# print(metrics.metrics_data["bias_metric"])
