from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import Dict, Any
from llm_config import get_llm, get_rag_chain

def load_and_process_url(url: str):
    """Load and process website content"""
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    return vectorstore.as_retriever()

def _categorizer_agent(retriever) -> str:
    """Categorize the AI tool"""
    prompt = ChatPromptTemplate.from_template(
        """
    Based on the provided context, categorize this AI tool:
    1. Main category (e.g., Text-to-Speech, Image Generation, etc.)
    2. Subcategories if applicable
    3. Primary use cases
    
    Context: {context}
    
    Provide a concise categorization.
    """
    )
    response = get_rag_chain(retriever, prompt).invoke({"input": "categorize this tool"})
    return response["answer"]

def _analyzer_agent(retriever) -> str:
    """Analyze top features"""
    prompt = ChatPromptTemplate.from_template(
        """
    Based on the provided context, identify and explain the top 3 features of this AI tool.
    For each feature, include its URL if available in the context.
    Format your response exactly as follows:

    1. [Feature Name](URL if available)
    - What makes it powerful/unique
    - What problems it solves
    - Key capabilities

    2. [Feature Name](URL if available)
    - What makes it powerful/unique
    - What problems it solves
    - Key capabilities

    3. [Feature Name](URL if available)
    - What makes it powerful/unique
    - What problems it solves
    - Key capabilities

    Important:
    - Only include URLs if they directly relate to the feature
    - URLs must be complete (starting with http:// or https://)
    - If no URL exists for a feature, just list the feature name
    
    Context: {context}
    """
    )
    response = get_rag_chain(retriever, prompt).invoke({"input": "analyze features with urls"})
    return response["answer"]

def _details_agent(retriever) -> str:
    """Extract additional details"""
    prompt = ChatPromptTemplate.from_template(
        """
    You are a detail-oriented researcher. Based on the provided context about this AI tool, 
    extract EXACT URLs and information for each category. Format your response exactly as follows:

    1. Pricing Information:
    - List each pricing tier and cost
    - Pricing page URL: [extract exact URL from context]

    2. Demo/Trial Access:
    - Trial availability details
    - Demo/Trial URL: [extract exact URL from context]

    3. Documentation:
    - API Documentation URL: [extract exact URL from context]
    - Getting Started URL: [extract exact URL from context]
    - Developer Docs URL: [extract exact URL from context]

    4. Additional Resources:
    - Tutorial URL: [extract exact URL from context]
    - Community/Support URL: [extract exact URL from context]
    - Integration Guide URL: [extract exact URL from context]

    Important:
    - Include COMPLETE URLs (starting with http:// or https://)
    - Place each URL on a new line after its description
    - If a URL is not found in the context, skip that item
    - Do not make up or modify URLs, only use exact URLs found in the context

    Context: {context}
    """
    )
    response = get_rag_chain(retriever, prompt).invoke({"input": "extract details and links"})
    return response["answer"]

def _final_reviewer_agent(url: str, category: str, features: str, details: str) -> str:
    """Create final review"""
    prompt = ChatPromptTemplate.from_template(
        """
    Write a comprehensive but easy-to-read review of this AI tool in a human-like tone.
    Include the following information in a natural way:
    
    Website: {url}
    Category: {category}
    Key Features: {features}
    Additional Details: {details}
    
    Make it sound like a helpful friend reviewing the tool, highlighting both strengths 
    and potential considerations. Keep it informative but conversational.
    """
    )

    llm = get_llm(temperature=0.7)
    messages = prompt.format_messages(
        url=url, category=category, features=features, details=details
    )
    response = llm.invoke(messages)
    return response.content

def review_url(url: str) -> Dict[str, Any]:
    """
    Master method to review a URL and return all components
    
    Args:
        url: Website URL to review
        
    Returns:
        Dict containing all review components
    """
    # Ensure URL has protocol
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        
    # Process website
    retriever = load_and_process_url(url)
    
    # Get all components
    category = _categorizer_agent(retriever)
    features = _analyzer_agent(retriever)
    details = _details_agent(retriever)
    final_review = _final_reviewer_agent(url, category, features, details)
    
    return {
        "url": url,
        "category": category,
        "features": features,
        "details": details,
        "final_review": final_review
    } 