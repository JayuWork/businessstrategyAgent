from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from html_generator import save_review
from llm_config import get_llm, get_rag_chain

# Load environment variables
load_dotenv()


def load_and_process_url(url: str):
    """Load and process website content"""
    print("\n1️⃣ Loading website content...")
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    print("✅ Website content processed and vectorized")
    return retriever


def categorizer_agent(retriever):
    """Categorize the AI tool"""
    print("\n2️⃣ Categorizer Agent analyzing tool type...")

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

    response = get_rag_chain(retriever, prompt).invoke(
        {"input": "categorize this tool"}
    )
    print("✅ Categorization complete")
    return response["answer"]


def analyzer_agent(retriever):
    """Analyze top features"""
    print("\n3️⃣ Analyzer Agent extracting key features...")

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

    response = get_rag_chain(retriever, prompt).invoke(
        {"input": "analyze features with urls"}
    )
    print("✅ Feature analysis complete")
    return response["answer"]


def mischievous_agent(retriever):
    """Extract additional details"""
    print("\n4️⃣ Mischievous Agent finding hidden gems...")

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

    response = get_rag_chain(retriever, prompt).invoke(
        {"input": "extract details and links"}
    )
    print("✅ Detail extraction complete")
    return response["answer"]


def reviewer_agent(url: str, category: str, features: str, details: str):
    """Create final review"""
    print("\n5️⃣ Reviewer Agent writing final review...")

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

    # Use higher temperature for more natural-sounding review
    llm = get_llm(temperature=0.7)
    messages = prompt.format_messages(
        url=url, category=category, features=features, details=details
    )

    response = llm.invoke(messages)
    print("✅ Review complete")
    return response.content


def save_review_json(url: str, category: str, features: str, details: str, final_review: str) -> str:
    """Save review data as JSON file"""
    import json
    from pathlib import Path
    from urllib.parse import urlparse
    
    # Create json directory if it doesn't exist
    json_dir = Path("reviews/json")
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from URL and timestamp
    domain = urlparse(url).netloc.replace("www.", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{domain}_{timestamp}.json"
    
    # Prepare data structure
    review_data = {
        "url": url,
        "category": category,
        "features": features,
        "details": details,
        "final_review": final_review,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to JSON file
    json_path = json_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
        
    return str(json_path)


def main():
    print("\n🤖 Welcome to GenAI Tool Reviewer!\n")
    url = input("Which GenAI tool you would like to review? Share URL ( ex: elevenlabs.io) : ")
    
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    retriever = load_and_process_url(url)
    category = categorizer_agent(retriever)
    features = analyzer_agent(retriever)
    details = mischievous_agent(retriever)
    final_review = reviewer_agent(url, category, features, details)

    html_file = save_review(url, category, features, details, final_review)
    json_file = save_review_json(url, category, features, details, final_review)

    print("\n📝 === Review Generation Complete ===")
    print(f"\n✨ Review saved to:")
    print(f"HTML: {html_file}")
    print(f"JSON: {json_file}")
    print("\nOpen the HTML file in your browser to view the formatted review.")


if __name__ == "__main__":
    main()
