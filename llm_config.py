from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama  # Updated import
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Default model setting
DEFAULT_MODEL = "llama"  # Options: "llama" or "gpt"

def get_llm(temperature=0, model=DEFAULT_MODEL):
    """Get LLM instance with specified temperature"""
    if model == "llama":
        return ChatOllama(
            model="llama3.1",  # or any other model you have in Ollama
            temperature=temperature
        )
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)


def get_rag_chain(retriever, prompt):
    """Create a RAG chain with given prompt and retriever"""
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    llm = get_llm()
    chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain
