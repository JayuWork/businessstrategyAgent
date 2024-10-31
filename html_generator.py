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

def save_review(url: str, category: str, features: str, details: str, review: str):
    """Save the review as HTML"""
    # Create filename from URL
    domain_name = url.replace('https://', '').replace('http://', '').split('/')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/{domain_name}_review_{timestamp}.html"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # Generate and save HTML content
    html_content = create_html_content(url, category, features, details, review)
    
    with open(filename, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    return filename