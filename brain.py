import os
import logging
import hashlib
from functools import lru_cache
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4",
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=300,
        request_timeout=30
    )


def get_pdf_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdfs_from_backend(pdf_folder=r"pdfs"):
    pdf_files, pdf_filenames = [], []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            pdf_files.append(filepath)
            pdf_filenames.append(filename)
    return pdf_files, pdf_filenames


def process_pdf(args):
    filepath, filename, text_splitter = args
    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split(text_splitter)
        for page in pages:
            page.metadata["filename"] = filename
        return pages
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return []


def get_index_for_pdf(pdf_filepaths: List[str], pdf_filenames: List[str], api_key: str) -> FAISS:
    embeddings = get_embeddings(api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=0,
        length_function=len
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_pdf, (filepath, filename, text_splitter))
            for filepath, filename in zip(pdf_filepaths, pdf_filenames)
        ]
        docs = []
        for future in concurrent.futures.as_completed(futures):
            docs.extend(future.result())

    vectordb = FAISS.from_documents(docs, embeddings)
    os.makedirs("index", exist_ok=True)
    vectordb.save_local("index")
    return vectordb


def load_faiss_index(api_key: str) -> FAISS:
    embeddings = get_embeddings(api_key)
    return FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)


def generate_response(query: str, api_key: str, prompt_template: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    # Initialize the LLM and load the FAISS index
    llm = get_llm(api_key)
    vectordb = load_faiss_index(api_key)
    
    # Perform similarity search to find relevant documents
    docs = vectordb.similarity_search(query, k=3)

    # If no relevant documents are found, return a default message
    if not docs:
        return "Mashay la, that question is beyond my scope. Please ask questions based on RSEB only."

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template + "\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    # Create the document chain
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )

    # Invoke the chain to generate a response
    response = chain.invoke({"question": query, "context": docs})

    # Format the response as a numbered or bulleted list if it contains steps or processes
    response = format_response(response)

    return response

def format_response(response: str) -> str:
    """
    Formats the response as a numbered or bulleted list if it contains steps or processes.
    """
    # Check if the response contains keywords indicating a list is needed
    list_keywords = ["steps", "process", "follow these", "instructions", "guide"]
    if any(keyword in response.lower() for keyword in list_keywords):
        # Split the response into lines
        lines = response.split('\n')
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        # Format as a numbered list
        formatted_response = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
        return formatted_response
    
    # If no list is needed, return the response as is
    return response


def generate_streaming_response(message: str, api_key: str, prompt_template: str, conversation_history: List):
    vectordb = load_faiss_index(api_key)
    docs = vectordb.similarity_search(message, k=3)

    if not docs:
        yield "data: Mashay la, that question is beyond my scope. Please ask questions based on RSEB only.\n\n"
        return

    llm = get_llm(api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{question}")
    ])

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )

    for chunk in chain.stream({"question": message, "context": docs}):
        yield f"data: {chunk}\n\n"


def update_conversation_history(history: List[Dict[str, str]], user_message: str, assistant_response: str) -> List[Dict[str, str]]:
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_response})
    return history
