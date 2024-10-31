from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any
from reviewer import (
    load_and_process_url,
    categorizer_agent,
    analyzer_agent,
    mischievous_agent,
    reviewer_agent
)

app = FastAPI(
    title="AI Tool Reviewer API",
    description="API for generating comprehensive reviews of AI tools",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    url: HttpUrl

class ReviewResponse(BaseModel):
    url: str
    category: str
    features: str
    details: str
    final_review: str

@app.post("/review", response_model=ReviewResponse)
async def review_ai_tool(request: ReviewRequest) -> Dict[str, Any]:
    """
    Generate a comprehensive review for an AI tool website
    
    Args:
        request: ReviewRequest containing the website URL
        
    Returns:
        Dict containing the review components
    """
    try:
        url = str(request.url)
        
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        # Process website content
        retriever = load_and_process_url(url)
        
        # Generate review components
        category = categorizer_agent(retriever)
        features = analyzer_agent(retriever)
        details = mischievous_agent(retriever)
        final_review = reviewer_agent(url, category, features, details)
        
        return {
            "url": url,
            "category": category,
            "features": features,
            "details": details,
            "final_review": final_review
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating review: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"} 