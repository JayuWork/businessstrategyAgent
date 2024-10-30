from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# Load environment variables
load_dotenv()

def load_and_process_url(url: str):
    """Load and process website content"""
    print("\n1️⃣ Loading website content...")
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    print("✅ Website content processed and vectorized")
    return retriever

def make_chain(retriever, prompt):
    """Create a RAG chain with given prompt"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create RAG chain
    chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    
    return retrieval_chain

def categorizer_agent(retriever):
    """Categorize the AI tool"""
    print("\n2️⃣ Categorizer Agent analyzing tool type...")
    
    prompt = ChatPromptTemplate.from_template("""
    Based on the provided context, categorize this AI tool:
    1. Main category (e.g., Text-to-Speech, Image Generation, etc.)
    2. Subcategories if applicable
    3. Primary use cases
    
    Context: {context}
    
    Provide a concise categorization.
    """)
    
    response = make_chain(retriever, prompt).invoke({"input": "categorize this tool"})
    print("✅ Categorization complete")
    return response["answer"]

def analyzer_agent(retriever):
    """Analyze top features"""
    print("\n3️⃣ Analyzer Agent extracting key features...")
    
    prompt = ChatPromptTemplate.from_template("""
    Based on the provided context, identify and explain the top 3 features of this AI tool:
    1. What are the most powerful/unique features?
    2. What makes them stand out?
    3. What problems do they solve?
    
    Context: {context}
    
    List and explain the top 3 features.
    """)
    
    response = make_chain(retriever, prompt).invoke({"input": "analyze features"})
    print("✅ Feature analysis complete")
    return response["answer"]

def mischievous_agent(retriever):
    """Extract additional details"""
    print("\n4️⃣ Mischievous Agent finding hidden gems...")
    
    prompt = ChatPromptTemplate.from_template("""
    Based on the provided context, extract the following details:
    1. Pricing information and plans
    2. Demo/trial availability
    3. API documentation links
    4. Getting started guides
    5. Any unique offerings or limitations
    
    Context: {context}
    
    Provide detailed information about pricing, demos, and technical integration.
    """)
    
    response = make_chain(retriever, prompt).invoke({"input": "extract details"})
    print("✅ Detail extraction complete")
    return response["answer"]

def reviewer_agent(url: str, category: str, features: str, details: str):
    """Create final review"""
    print("\n5️⃣ Reviewer Agent writing final review...")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("""
    Write a balanced and professional review of this AI tool in clear, accessible language.
    Structure the review to cover the following key points:
    
    Website: {url}
    Category: {category}
    Key Features: {features}
    Additional Details: {details}
    
    Present the information objectively, highlighting both capabilities and limitations.
    Focus on providing practical insights while maintaining a professional yet readable tone.
    """)
    messages = prompt.format_messages(
        url=url,
        category=category,
        features=features,
        details=details
    )
    
    response = llm.invoke(messages)
    print("✅ Review complete")
    return response.content

def save_review(url: str, review: str):
    """Save the review to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_tool_review_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"AI Tool Review: {url}\n")
        f.write("=" * 50 + "\n\n")
        f.write(review)
    
    return filename

def main():
    print("\n🤖 Welcome to GenAI Tool Reviewer!\n")
    url = input("Enter the GenAI tool website URL (e.g., https://elevenlabs.io): ")
    
    # Process website
    retriever = load_and_process_url(url)
    
    # Run agents
    category = categorizer_agent(retriever)
    features = analyzer_agent(retriever)
    details = mischievous_agent(retriever)
    
    # Generate review
    final_review = reviewer_agent(url, category, features, details)
    
    # Save and display review
    filename = save_review(url, final_review)
    
    print("\n📝 === AI Tool Review ===\n")
    print(final_review)
    print(f"\n✨ Review saved to: {filename} ✨")

if __name__ == "__main__":
    main()
