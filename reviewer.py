from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import json
from html_generator import save_review
from agents import review_url

# Load environment variables
load_dotenv()

def save_review_json(review_data: dict) -> str:
    """Save review data as JSON file"""
    # Create json directory if it doesn't exist
    json_dir = Path("output/json")
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from URL and timestamp
    domain = urlparse(review_data["url"]).netloc.replace("www.", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{domain}_{timestamp}.json"
    
    # Add timestamp to data
    review_data["timestamp"] = datetime.now().isoformat()
    
    # Save to JSON file
    json_path = json_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
        
    return str(json_path)

def main():
    print("\n🤖 Welcome to GenAI Tool Reviewer!\n")
    url = input("Which GenAI tool would you like to review? Share URL (ex: elevenlabs.io): ")
    
    print("\n🔄 Generating review...")
    review_data = review_url(url)
    
    # Save both formats
    html_file = save_review(**review_data)
    json_file = save_review_json(review_data)

    print("\n📝 === Review Generation Complete ===")
    print(f"\n✨ Review saved to:")
    print(f"HTML: {html_file}")
    print(f"JSON: {json_file}")
    print("\nOpen the HTML file in your browser to view the formatted review.")

if __name__ == "__main__":
    main()
