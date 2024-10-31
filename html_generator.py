from datetime import datetime

def format_features_html(features_text):
    """Convert features text to HTML with optional links"""
    import re
    
    html_parts = []
    features = re.split(r'\d+\.', features_text)[1:]
    
    for feature in features:
        if not feature.strip():
            continue
            
        url_match = re.search(r'\[(.*?)\]\((https?://[^\s)]+)\)', feature)
        
        if url_match:
            feature_name = url_match.group(1)
            feature_url = url_match.group(2)
            feature_content = feature.replace(url_match.group(0), '')
            html_parts.append(
                f'<div class="feature">'
                f'<h3><a href="{feature_url}" target="_blank">🔗 {feature_name}</a></h3>'
                f'{feature_content.replace("-", "•")}'
                f'</div>'
            )
        else:
            feature_lines = feature.strip().split('\n')
            feature_name = feature_lines[0].strip()
            feature_content = '\n'.join(feature_lines[1:])
            html_parts.append(
                f'<div class="feature">'
                f'<h3>{feature_name}</h3>'
                f'{feature_content.replace("-", "•")}'
                f'</div>'
            )
    
    return '\n'.join(html_parts)

def extract_links(text, section_pattern):
    """Extract links from a specific section of text"""
    import re
    
    section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
    if not section_match:
        return ""
    
    section_text = section_match.group(1)
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', section_text)
    
    if not urls:
        return ""
    
    links_html = ""
    for url in urls:
        description = section_text.split(url)[0].split('\n')[-1].strip()
        if not description:
            description = url
        links_html += f'<a href="{url}" target="_blank">🔗 {description}</a>\n'
    
    return links_html

def create_html_content(url: str, category: str, features: str, details: str, review: str):
    """Create formatted HTML content"""
    # Format the URL for display
    display_url = url.replace('https://', '').replace('http://', '').split('/')[0]
    
    # Get current date
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Process links
    pricing_links = extract_links(details, r"Pricing Information:(.*?)(?=\n\d\.|\Z)")
    documentation_links = extract_links(details, r"Documentation:(.*?)(?=\n\d\.|\Z)")
    resource_links = extract_links(details, r"Additional Resources:(.*?)(?=\n\d\.|\Z)")
    
    # Format features
    features_html = format_features_html(features)
    
    # Generate HTML template
    with open('./templates/review_template.html', 'r') as f:
        html_template = f.read()
    
    # Create quick links section if links exist
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
    
    return html_template.format(
        url=url,
        display_url=display_url,
        date=current_date,
        category=category,
        features=features_html,
        detailed_review=review,
        details=details,
        quick_links_section=quick_links_section
    )

def save_review(url: str, category: str, features: str, details: str, final_review: str, **kwargs) -> str:
    """Save review as HTML file"""
    from pathlib import Path
    from datetime import datetime
    from urllib.parse import urlparse
    
    # Create output directory if it doesn't exist
    output_dir = Path("output/html")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from URL and timestamp
    domain = urlparse(url).netloc.replace("www.", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{domain}_{timestamp}.html"
    
    # Generate HTML content
    html_content = format_review_html(
        url=url,
        category=category,
        features=features,
        details=details,
        final_review=final_review
    )
    
    # Save to file
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return str(output_path)

def format_review_html(**kwargs) -> str:
    """Format review data as HTML"""
    from datetime import datetime
    
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Tool Review</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            .section {{
                margin: 2rem 0;
                padding: 1rem;
                background: #f9f9f9;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                border-bottom: 2px solid #eee;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .feature {{
                margin: 1rem 0;
                padding: 1rem;
                background: white;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .timestamp {{
                color: #666;
                font-size: 0.9em;
                margin-top: 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI Tool Review</h1>
            <p><strong>Website:</strong> <a href="{url}" target="_blank">{url}</a></p>
            <p class="timestamp">Generated on: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Category</h2>
            {category}
        </div>
        
        <div class="section">
            <h2>Key Features</h2>
            {features}
        </div>
        
        <div class="section">
            <h2>Additional Details</h2>
            {details}
        </div>
        
        <div class="section">
            <h2>Final Review</h2>
            {final_review}
        </div>
    </body>
    </html>
    """
    
    return template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs
    )