from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def get_llm(temperature=0):
    """Get LLM instance with specified temperature"""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)


def get_rag_chain(retriever, prompt):
    """Create a RAG chain with given prompt and retriever"""
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    llm = get_llm()
    chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain
