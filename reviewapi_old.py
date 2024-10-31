from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any
from agents import review_url

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
    """Generate a comprehensive review for an AI tool website"""
    try:
        return review_url(str(request.url))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating review: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}