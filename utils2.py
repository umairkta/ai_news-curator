import requests
import feedparser
from typing import List, Dict

# ============ RSS FEED SOURCES ============
RSS_FEEDS = {
    "HuggingFace Blog": "https://huggingface.co/blog/feed.xml",
    "OpenAI Blog": "https://openai.com/blog/feed.xml",
    "DeepMind Blog": "https://www.deepmind.com/blog/feed.xml",
    "AI Weekly": "https://aiweekly.co/issues.rss",
    "Phys.org AI": "https://phys.org/rss-feed/technology-news/machine-learning-ai/",
    "Google Developers": "https://feeds.feedburner.com/GDBcode",
}


# ============ FETCH NEWS FUNCTION ============
def fetch_news_from_rss() -> List[Dict]:
    """
    Fetch news from all configured RSS feeds.
    
    Returns:
        List of article dictionaries with title, summary, link, source
    """
    all_articles = []
    
    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  # Get top 5 from each source
                article = {
                    "title": entry.get("title", "No title"),
                    "summary": entry.get("summary", "No summary")[:300],
                    "link": entry.get("link", "#"),
                    "source": source_name,
                    "published": entry.get("published", "Unknown")
                }
                all_articles.append(article)
        except Exception as e:
            print(f"⚠️ Could not fetch from {source_name}: {str(e)}")
    
    return all_articles

# ============ MISTRAL CLASSIFICATION FUNCTION ============
def classify_with_mistral(
    article_title: str, 
    article_summary: str, 
    persona: str
) -> bool:
    """
    Use Mistral 7B to classify article relevance to a persona.
    
    Args:
        article_title: Title of the article
        article_summary: Summary of the article
        persona: The persona to classify for
        
    Returns:
        Boolean indicating if article is relevant
    """
    
    prompt = f"""Classify if this article is relevant to a {persona}.

Article Title: {article_title}
Article Summary: {article_summary}

Is this article relevant to {persona}? Answer with ONLY "yes" or "no":"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1  # Low temperature for consistent classification
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").lower().strip()
            return "yes" in response_text or "relevant" in response_text
        else:
            return False
    except Exception as e:
        print(f"❌ Error classifying with Mistral: {str(e)}")
        return False

# ============ ADD CUSTOM PERSONAS ============
def get_personas() -> Dict:
    """
    Get all available personas.
    
    Returns:
        Dictionary of personas with keywords
    """
    return {
        "Developers and Programmers": {
            "keywords": ["python", "api", "framework", "code", "algorithm", "tool", "library", "sdk"],
            "description": "For developers building AI applications"
        },
        "Investors and VCs": {
            "keywords": ["startup", "funding", "investment", "market", "growth", "valuation", "series"],
            "description": "For investors tracking AI company news"
        },
        "Students and Researchers": {
            "keywords": ["paper", "research", "study", "experiment", "model", "dataset", "breakthrough"],
            "description": "For academics and researchers"
        }
    }

# ============ HELPER FUNCTIONS ============
def format_article(article: Dict) -> str:
    """Format an article for display."""
    return f"""
    Title: {article['title']}
    Summary: {article['summary']}
    Source: {article['source']}
    Link: {article['link']}
    """

def validate_ollama_connection() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False
