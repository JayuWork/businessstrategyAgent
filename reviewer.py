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
import re

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

    response = make_chain(retriever, prompt).invoke({"input": "categorize this tool"})
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

    response = make_chain(retriever, prompt).invoke(
        {"input": "analyze features with urls"}
    )
    print("✅ Feature analysis complete")
    return response["answer"]


def format_features_html(features_text):
    """Convert features text to HTML with optional links"""
    import re

    html_parts = []

    # Split into individual features
    features = re.split(r"\d+\.", features_text)[1:]  # Skip empty first part

    for feature in features:
        if not feature.strip():
            continue

        # Try to extract feature name and URL if present
        url_match = re.search(r"\[(.*?)\]\((https?://[^\s)]+)\)", feature)

        if url_match:
            # Feature has a URL
            feature_name = url_match.group(1)
            feature_url = url_match.group(2)
            feature_content = feature.replace(
                url_match.group(0), ""
            )  # Remove the [name](url) part
            html_parts.append(
                f"""
                <div class="feature">
                    <h3><a href="{feature_url}" target="_blank">🔗 {feature_name}</a></h3>
                    {feature_content.replace('-', '•')}
                </div>
            """
            )
        else:
            # Feature without URL
            feature_lines = feature.strip().split("\n")
            feature_name = feature_lines[0].strip()
            feature_content = "\n".join(feature_lines[1:])
            html_parts.append(
                f"""
                <div class="feature">
                    <h3>{feature_name}</h3>
                    {feature_content.replace('-', '•')}
                </div>
            """
            )

    return "\n".join(html_parts)


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

    # Make multiple retrieval attempts with different queries to increase URL coverage
    queries = [
        "pricing page url link cost",
        "documentation api developer docs",
        "demo trial getting started tutorial",
        "community support integration guide",
    ]

    all_responses = []
    for query in queries:
        response = make_chain(retriever, prompt).invoke({"input": query})
        all_responses.append(response["answer"])

    # Combine all responses while removing duplicates
    combined_response = "\n".join(all_responses)

    # Clean up the response to remove duplicates and format consistently
    import re

    urls = re.findall(r'https?://[^\s<>"]+', combined_response)
    unique_urls = list(dict.fromkeys(urls))  # Remove duplicates while preserving order

    # Reconstruct the response with unique URLs
    sections = []

    # Add pricing section if pricing URLs exist
    pricing_urls = [url for url in unique_urls if "pricing" in url.lower()]
    if pricing_urls:
        sections.append(
            "• Pricing Information:\n" + "\n".join(f"- {url}" for url in pricing_urls)
        )

    # Add demo section if demo URLs exist
    demo_urls = [
        url
        for url in unique_urls
        if any(x in url.lower() for x in ["demo", "trial", "try"])
    ]
    if demo_urls:
        sections.append(
            "• Demo/Trial Access:\n" + "\n".join(f"- {url}" for url in demo_urls)
        )

    # Add documentation section if doc URLs exist
    doc_urls = [
        url
        for url in unique_urls
        if any(x in url.lower() for x in ["doc", "api", "developer"])
    ]
    if doc_urls:
        sections.append(
            "• Documentation:\n" + "\n".join(f"- {url}" for url in doc_urls)
        )

    # Add resources section if resource URLs exist
    resource_urls = [
        url
        for url in unique_urls
        if any(x in url.lower() for x in ["tutorial", "guide", "community", "support"])
    ]
    if resource_urls:
        sections.append(
            "• Additional Resources:\n" + "\n".join(f"- {url}" for url in resource_urls)
        )

    final_response = "\n\n".join(sections)

    print("✅ Detail extraction complete")
    return final_response


def reviewer_agent(url: str, category: str, features: str, details: str):
    """Create final review"""
    print("\n5️⃣ Reviewer Agent writing final review...")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    prompt = ChatPromptTemplate.from_template(
        """
    Write a balanced and professional review of this AI tool in clear, accessible language.
    Structure the review to cover the following key points:
    
    Website: {url}
    Category: {category}
    Key Features: {features}
    Additional Details: {details}
    
    Present the information objectively, highlighting both capabilities and limitations.
    Focus on providing practical insights while maintaining a professional yet readable tone.
    """
    )
    messages = prompt.format_messages(
        url=url, category=category, features=features, details=details
    )

    response = llm.invoke(messages)
    print("✅ Review complete")
    return response.content


def extract_links(text, section_pattern):
    """Extract links from a specific section of text"""
    import re

    section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
    if not section_match:
        return ""  # Return empty string instead of "No links available"

    section_text = section_match.group(1)
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', section_text)

    if not urls:
        return ""  # Return empty string if no URLs found

    links_html = ""
    for url in urls:
        description = section_text.split(url)[0].split("\n")[-1].strip()
        if not description:
            description = url
        links_html += f'<a href="{url}" target="_blank">🔗 {description}</a>\n'

    return links_html


def create_html_content(
    url: str, category: str, features: str, details: str, review: str
):
    """Create formatted HTML content"""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Tool Review: {url}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            .header {{
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
            h1 {{
                color: #2c3e50;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .feature {{
                background-color: #f8f9fa;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .feature h3 {{
                margin-top: 0;
                color: #2c3e50;
                font-size: 1.2em;
            }}
            .feature h3 a {{
                color: #2c3e50;
                text-decoration: none;
            }}
            .feature h3 a:hover {{
                color: #007bff;
            }}
            .links-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .link-section {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .link-section h3 {{
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 10px;
            }}
            .link-section a {{
                display: block;
                color: #007bff;
                text-decoration: none;
                margin: 8px 0;
                padding: 5px;
                border-radius: 4px;
            }}
            .link-section a:hover {{
                background-color: #e9ecef;
                color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1><a href="{url}">{display_url}</a> : Review</h1>
            <p><strong>Reviewed on:</strong> {date}</p>
        </div>

        <div class="section">
            <h2>Quick Summary</h2>
            {summary}
        </div>

        <div class="section">
            <h2>Categorization</h2>
            {category}
        </div>

        <div class="section">
            <h2>Top Features</h2>
            {features}
        </div>

        <div class="section">
            <h2>Detailed Review</h2>
            {detailed_review}
        </div>

        <div class="section">
            <h2>Technical Details & Resources</h2>
            {details}
        </div>

        {quick_links_section}
    </body>
    </html>
    """

    from datetime import datetime

    current_date = datetime.now().strftime("%B %d, %Y")

    # Generate a quick summary using OpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this review in exactly 100 words: {review}"
    )
    summary_message = summary_prompt.format_messages(review=review)
    summary = llm.invoke(summary_message).content

    # Format the detailed review to ensure minimum length
    detailed_review_prompt = ChatPromptTemplate.from_template(
        "Expand this review into a detailed analysis of at least 250 words while maintaining accuracy: {review}"
    )
    detailed_message = detailed_review_prompt.format_messages(review=review)
    detailed_review = llm.invoke(detailed_message).content

    # Format features with potential links
    features_html = format_features_html(features)

    # Format features and details for HTML
    features_html = features.replace("\n", "<br>")
    details_html = details.replace("\n", "<br>")
    category_html = category.replace("\n", "<br>")

    # Process links from details
    pricing_links = extract_links(details, r"Pricing Information:(.*?)(?=\n\d\.|\Z)")
    documentation_links = extract_links(details, r"Documentation:(.*?)(?=\n\d\.|\Z)")
    resource_links = extract_links(details, r"Additional Resources:(.*?)(?=\n\d\.|\Z)")

    # Only show Quick Links section if we have any links
    quick_links_section = ""
    if any([pricing_links, documentation_links, resource_links]):
        quick_links_section = f"""
        <div class="section">
            <h2>Quick Links</h2>
            <div class="links-container">
                {f'''
                <div class="link-section">
                    <h3>🏷️ Pricing</h3>
                    {pricing_links}
                </div>
                ''' if pricing_links else ''}
                
                {f'''
                <div class="link-section">
                    <h3>🔧 Documentation</h3>
                    {documentation_links}
                </div>
                ''' if documentation_links else ''}
                
                {f'''
                <div class="link-section">
                    <h3>🎯 Resources</h3>
                    {resource_links}
                </div>
                ''' if resource_links else ''}
            </div>
        </div>
        """

    # Modify the HTML template to use conditional Quick Links section
    html_template = html_template.replace(
        """<div class="section">
            <h2>Quick Links</h2>
            <div class="links-container">
                <div class="link-section">
                    <h3>🏷️ Pricing</h3>
                    {pricing_links}
                </div>
                <div class="link-section">
                    <h3>🔧 Documentation</h3>
                    {documentation_links}
                </div>
                <div class="link-section">
                    <h3>🎯 Resources</h3>
                    {resource_links}
                </div>
            </div>
        </div>""",
        "{quick_links_section}",
    )

    # Format the URL for display
    display_url = url.replace('https://', '').replace('http://', '').split('/')[0]
    
    return html_template.format(
        url=url,
        display_url=display_url,
        date=current_date,
        summary=summary,
        category=category_html,
        features=features_html,
        detailed_review=detailed_review,
        details=details_html,
        quick_links_section=quick_links_section
    )


def save_review(url: str, category: str, features: str, details: str, review: str):
    """Save the review as HTML"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_tool_review_{timestamp}.html"

    html_content = create_html_content(url, category, features, details, review)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filename


def main():
    print("\n🤖 Welcome to GenAI Tool Reviewer!\n")
    url = input("Enter the GenAI tool website URL (e.g., https://elevenlabs.io): ")

    # Process website
    # Add http:// prefix if not present
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    retriever = load_and_process_url(url)

    # Run agents
    category = categorizer_agent(retriever)
    features = analyzer_agent(retriever)
    details = mischievous_agent(retriever)

    # Generate review
    final_review = reviewer_agent(url, category, features, details)

    # Save and display review
    filename = save_review(url, category, features, details, final_review)

    print("\n📝 === Review Generation Complete ===")
    print(f"\n✨ Review saved to: {filename} ✨")
    print("\nOpen the HTML file in your browser to view the formatted review.")


if __name__ == "__main__":
    main()
