import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.openai_functions import create_citation_fuzzy_match_runnable
from langchain_pinecone import PineconeVectorStore
import fitz
import os, json


def initialize():
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)


def get_pdf_data(pdf_file: str):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    print(len(text))
    # print(text)
    return text


def create_analysis_document(answer, query, score, reason, opinions, verdicts):
    result = answer["result"]
    source_documents = []
    for source in answer["source_documents"]:
        page = source.metadata["page"]
        file = source.metadata["source"]
        content = source.page_content
        source_documents.append({"page": page, "file": file, "content": content})
    data = {
        "query": query,
        "response": result,
        "citations": source_documents,
        "score": score,
        "reason": reason,
        "opinions": opinions,
        "verdicts": verdicts,
    }
    with open(
        r"data\output_files\metadata1.json",
        "r",
    ) as f:
        existing_data = json.load(f)
    existing_data.append(data)
    with open(
        r"\data\output_files\metadata1.json",
        "w",
    ) as f:
        json.dump(existing_data, f)


def load_file(file):

    name, extension = os.path.splitext(file)
    if extension == ".pdf":

        print(f"file: {file} loading................")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader

        print(f"file: {file} loading................")
        loader = Docx2txtLoader(file)
    else:
        print(f"file: {file} format {extension} not supported")
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256):

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


def check_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost in USD: {total_tokens / 1000 * 0.0004:.6f}")


def insert_or_fetch_embeddings(index_name, chunks):

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    if index_name in pc.list_indexes().names():
        print(f"Index: {index_name} already exists, Loading embeddings.....", end="")
        vector_store = PineconeVectorStore(
            pc.Index(index_name), embeddings, text_key="text"
        )
        print("Done!")
        return vector_store
    else:
        print(f"creating index {index_name} and embeddings....")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=pc.ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        vector_store = pc.from_documents(chunks, embeddings, index_name=index_name)
        print("Done!")
        return vector_store


def get_prompt(query, workflow, context=None):
    if workflow == "rag":
        prompt = f"""
        you are an expert in english and analyzing textual data.
        you are given a query to answer.
        Use only the information present with you, do not use external information.
        
        Query: {query}
        """
        return prompt
    else:
        prompt = f"""
        you are an expert in english and analyzing textual data.
        you are given a query to answer.
        Use the context below to answer the query.
        
        Query: {query}
        
        Context:
        donald trump is dumb
        """
        return prompt


def jarvis(query, model, workflow, vector_store=None, context=None):

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=3000)

    if workflow == "rag":

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        prompt = get_prompt(query, workflow)
        answer = chain.invoke(prompt)
    else:
        prompt = get_prompt(query, workflow, context)
        print("prompt:", prompt)
        answer = llm.invoke(query)

    return answer


def write_to_file(actial_output: str, query: str, metrics_data, file_name: str):
    data = {"query": query, "response": actial_output, "metrics_data": metrics_data}
    with open(
        file_name,
        "w",
    ) as f:
        json.dump(data, f)
