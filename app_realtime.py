import streamlit as st
import subprocess
import json
import re
import feedparser
import requests
from datetime import datetime
from datetime import timedelta
from bs4 import BeautifulSoup
import os

# ============================================
# CONFIGURATION
# ============================================

# Configure Ollama to use port 11435 if 11434 is stuck
os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'

st.set_page_config(
    page_title="AI News Curator (Plan A)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI News Curator - Plan A (Real-Time News)")
st.caption("Live AI News Feeds + Mistral 7B Curation")

# ============================================
# REAL AI NEWS SOURCES (RSS FEEDS)
# ============================================

AI_NEWS_SOURCES = {
    "arXiv AI": "https://arxiv.org/rss/cs.AI",
    "arXiv ML": "https://arxiv.org/rss/cs.LG",
    "Google AI Blog": "https://ai.googleblog.com/feeds/posts/default",
    "OpenAI Blog": "https://openai.com/blog/feed.rss",
    "DeepMind Blog": "https://www.deepmind.com/blog/feed/rss.xml",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
}

# ============================================
# PERSONAS (FIXED LIST)
# ============================================

PERSONAS = [
    "Developers and Programmers",
    "Investors and Venture Capitalists",
    "Students and Researchers",
    "Founders and Business Leaders",
    "Healthcare Professionals",
    "Designers and Creative Professionals",
    "Journalists and Media Professionals",
    "Marketing and Advertising Professionals"
]

# ============================================
# FETCH REAL NEWS FROM RSS FEEDS
# ============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_real_ai_news():
    """
    Fetches real AI news from RSS feeds.
    Returns list of articles with title, link, summary, source.
    """
    all_articles = []
    
    for source_name, feed_url in AI_NEWS_SOURCES.items():
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:5]:  # Get top 5 from each source
                article = {
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", "")[:300],  # Truncate to 300 chars
                    "published": entry.get("published", "Recently"),
                    "source": source_name,
                    "domain": "AI Research" if "arxiv" in source_name.lower() else "AI News"
                }
                
                # Clean up summary (remove HTML tags)
                article["summary"] = BeautifulSoup(article["summary"], "html.parser").get_text()
                
                all_articles.append(article)
        
        except Exception as e:
            st.warning(f"Could not fetch from {source_name}: {str(e)}")
            continue
    
    return all_articles

# ============================================
# USE MISTRAL TO CURATE FOR PERSONA
# ============================================

def run_llm_model(prompt_text, model="mistral"):
    """
    Runs the LLM model locally using Ollama.
    Tries both port 11434 and 11435.
    """
    ports = [11435, 11434]  # Try 11435 first (less congested)
    
    for port in ports:
        try:
            host = f"127.0.0.1:{port}"
            process = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, 'OLLAMA_HOST': host}
            )
            
            try:
                output, error = process.communicate(input=prompt_text + "\n", timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                st.error("Time-out: Model took >5 minutes.")
                return ""
            
            if error and "error" in error.lower():
                continue  # Try next port
            
            return output
        
        except Exception as e:
            if port == ports[-1]:  # Last port tried
                st.error(f"ERROR: Ollama not responding on any port. Make sure 'ollama serve' is running.")
            continue
    
    return ""

def curate_articles_for_persona(articles, persona, num_updates):
    """
    Uses Mistral to filter and curate articles for the selected persona.
    """
    if not articles:
        return []
    
    # Create a summary of articles to send to Mistral
    articles_text = "\n\n".join([
        f"Title: {a['title']}\nSource: {a['source']}\nSummary: {a['summary']}\nLink: {a['link']}"
        for a in articles[:15]  # Limit to top 15 articles
    ])
    
    prompt = f"""You are an AI news curator. Filter these {len(articles)} real AI news articles 
and select the TOP {num_updates} most relevant for: {persona}

Articles:
{articles_text}

For each selected article, output EXACTLY this format:

TITLE: [headline]
SOURCE: [source name]
RELEVANCE: [brief explanation of why this is relevant to {persona}]
LINK: [exact URL from the article]
---

Select {num_updates} articles that matter most to {persona}. Be selective and relevant."""
    
    response = run_llm_model(prompt, model="mistral")
    
    if not response:
        return articles[:num_updates]  # Fallback to raw articles
    
    # Parse Mistral's recommendations
    curated = []
    sections = response.split("---")
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split("\n")
        item = {
            "title": "",
            "source": "",
            "relevance": "",
            "link": ""
        }
        
        for line in lines:
            if line.startswith("TITLE:"):
                item["title"] = line.replace("TITLE:", "").strip()
            elif line.startswith("SOURCE:"):
                item["source"] = line.replace("SOURCE:", "").strip()
            elif line.startswith("RELEVANCE:"):
                item["relevance"] = line.replace("RELEVANCE:", "").strip()
            elif line.startswith("LINK:"):
                item["link"] = line.replace("LINK:", "").strip()
        
        if item["title"] and item["link"]:
            curated.append(item)
    
    return curated[:num_updates] if curated else articles[:num_updates]

# ============================================
# SIDEBAR CONFIGURATION
# ============================================

st.sidebar.header("Configuration")

selected_persona = st.sidebar.selectbox(
    "Select Your Persona:",
    PERSONAS,
    help="Choose the persona to filter AI news"
)

num_updates = st.sidebar.slider(
    "Number of Updates:",
    min_value=3,
    max_value=8,
    value=5,
    step=1
)

# Model selection
model_choice = st.sidebar.radio(
    "Select Model:",
    ["Mistral 7B (Fast)", "Llama 2 (Balanced)", "DeepSeek-R1 (Slow)"],
    help="Mistral is fastest for curation"
)

model_map = {
    "Mistral 7B (Fast)": "mistral",
    "Llama 2 (Balanced)": "llama2",
    "DeepSeek-R1 (Slow)": "deepseek-r1"
}
selected_model = model_map[model_choice]

st.sidebar.subheader("Model Status")
model_display = model_choice.split("(")[0].strip()
st.sidebar.info(f"Selected: {model_display}\nRuntime: Ollama (Port 11435)\nNews Source: Real RSS Feeds")

st.sidebar.subheader("System Info")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption("Fetching REAL AI news from RSS feeds. Filtering with Mistral.")

# ============================================
# MAIN CONTENT AREA
# ============================================

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["AI News", "Browse All", "Sources", "About"])

# ============================================
# TAB 1: CURATED AI NEWS
# ============================================

with tab1:
    st.subheader(f"AI News Curated for: {selected_persona}")
    st.write(f"Fetching real AI news from {len(AI_NEWS_SOURCES)} sources and curating for {selected_persona}...")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        curate_button = st.button("🔄 Fetch & Curate News", key="curate_news", use_container_width=True)
    
    with col2:
        refresh_button = st.button("↻ Refresh", key="refresh_news", use_container_width=True)
    
    if curate_button or refresh_button:
        with st.spinner(f"Fetching real AI news from {len(AI_NEWS_SOURCES)} sources..."):
            try:
                # Step 1: Fetch real news
                all_articles = fetch_real_ai_news()
                
                if not all_articles:
                    st.error("Could not fetch news from any source. Check your internet connection.")
                else:
                    st.success(f"✅ Fetched {len(all_articles)} real AI articles from {len(AI_NEWS_SOURCES)} sources")
                    
                    # Step 2: Curate with Mistral
                    with st.spinner(f"Curating top {num_updates} articles for {selected_persona}..."):
                        curated = curate_articles_for_persona(all_articles, selected_persona, num_updates)
                    
                    if curated:
                        st.success(f"✅ Curated {len(curated)} articles for {selected_persona}")
                        st.markdown("---")
                        
                        for idx, article in enumerate(curated, 1):
                            with st.container(border=True):
                                # Title with link
                                st.markdown(f"### {idx}. [{article.get('title', 'No title')}]({article.get('link', '#')})")
                                
                                # Summary
                                if "summary" in article and article["summary"]:
                                    st.write(article["summary"])
                                elif "relevance" in article and article["relevance"]:
                                    st.write(article["relevance"])
                                
                                # Metadata
                                meta_col1, meta_col2, meta_col3 = st.columns(3)
                                
                                with meta_col1:
                                    st.caption(f"📰 Source: {article.get('source', 'Unknown')}")
                                
                                with meta_col2:
                                    if "published" in article:
                                        st.caption(f"📅 {article['published']}")
                                    else:
                                        st.caption(f"🎯 Relevance: {article.get('relevance', 'High')[:50]}...")
                                
                                with meta_col3:
                                    st.caption(f"👤 For: {selected_persona.split(' ')[0]}")
                                
                                # Direct link button
                                st.markdown(f"[🔗 Read Full Article]({article.get('link', '#')})", unsafe_allow_html=False)
                                st.markdown("---")
                    else:
                        st.warning("Could not curate articles. Try again or use default articles.")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================
# TAB 2: BROWSE ALL FETCHED NEWS
# ============================================

with tab2:
    st.subheader("Browse All Fetched AI News")
    st.write("View all articles fetched from RSS feeds (not curated).")
    
    if st.button("📥 Fetch All Articles", key="fetch_all", use_container_width=True):
        with st.spinner(f"Fetching from {len(AI_NEWS_SOURCES)} sources..."):
            all_articles = fetch_real_ai_news()
            
            if all_articles:
                st.success(f"✅ Fetched {len(all_articles)} articles")
                
                # Filter by source
                sources = list(set([a["source"] for a in all_articles]))
                selected_sources = st.multiselect("Filter by source:", sources, default=sources[:3])
                
                filtered = [a for a in all_articles if a["source"] in selected_sources]
                
                st.markdown("---")
                
                for idx, article in enumerate(filtered, 1):
                    with st.container(border=True):
                        st.markdown(f"### {idx}. [{article['title']}]({article['link']})")
                        st.write(article['summary'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📰 {article['source']}")
                        with col2:
                            st.caption(f"📅 {article['published']}")
                        
                        st.markdown("---")
            else:
                st.error("Could not fetch articles.")

# ============================================
# TAB 3: NEWS SOURCES
# ============================================

with tab3:
    st.subheader("AI News Sources (Real-Time RSS Feeds)")
    st.write("Click any link to visit the source directly.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Research Papers")
        st.markdown("- [🔗 arXiv - AI Research](https://arxiv.org/list/cs.AI/recent)")
        st.markdown("- [🔗 arXiv - Machine Learning](https://arxiv.org/list/cs.LG/recent)")
    
    with col2:
        st.markdown("#### Major AI Companies")
        st.markdown("- [🔗 Google AI Blog](https://ai.googleblog.com/)")
        st.markdown("- [🔗 OpenAI Blog](https://openai.com/blog/)")
        st.markdown("- [🔗 DeepMind Blog](https://www.deepmind.com/blog)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### AI Communities")
        st.markdown("- [🔗 Hugging Face Blog](https://huggingface.co/blog)")
    
    with col2:
        st.markdown("#### Tech News")
        st.markdown("- [🔗 TechCrunch AI](https://techcrunch.com/category/artificial-intelligence/)")

# ============================================
# TAB 4: ABOUT
# ============================================

with tab4:
    st.subheader("About Plan A")
    
    st.markdown("""
    ## Plan A - Real-Time AI News Curator
    
    **Now with REAL NEWS!** Fetches live AI news from RSS feeds and curates with Mistral 7B.
    
    ### ✨ Features
    - ✅ Real-Time News: Fetches from 7+ live RSS feeds
    - ✅ AI Curation: Mistral filters for your persona
    - ✅ Zero Cost: No APIs, no subscriptions
    - ✅ Multiple Models: Mistral 7B, Llama 2, DeepSeek-R1
    - ✅ 8 Personas: Custom filtering for different audiences
    - ✅ Direct Links: Click to read full articles
    
    ### 📰 News Sources
    
    | Source | Type | Updates |
    |--------|------|---------|
    | arXiv AI & ML | Research Papers | Daily |
    | Google AI Blog | Official | Weekly |
    | OpenAI Blog | Official | Weekly |
    | DeepMind Blog | Official | Weekly |
    | Hugging Face | Community | Daily |
    | TechCrunch | News | Hourly |
    
    ### 🚀 How It Works
    
    1. **Fetch Real News**: App downloads latest articles from 7 RSS feeds
    2. **Filter & Curate**: Mistral 7B reads all articles
    3. **Personalize**: Mistral selects top articles for YOUR persona
    4. **Read & Share**: Click links to read full articles
    
    ### 🎯 Personas
    
    Choose from 8 personas:
    - Developers and Programmers
    - Investors and Venture Capitalists
    - Students and Researchers
    - Founders and Business Leaders
    - Healthcare Professionals
    - Designers and Creative Professionals
    - Journalists and Media Professionals
    - Marketing and Advertising Professionals
    
    ### 💻 Technology Stack
    
    - **News Source**: Real RSS feeds (arXiv, Google, OpenAI, DeepMind, etc.)
    - **Framework**: Streamlit (web app)
    - **Runtime**: Ollama (local, port 11435)
    - **AI Model**: Mistral 7B (fast curation)
    - **Language**: Python
    - **Cost**: FREE
    
    ### 📥 Setup Instructions
    
    **1. Install Ollama:**
    ```
    Visit https://ollama.com/download
    ```
    
    **2. Download Mistral:**
    ```bash
    ollama pull mistral
    ```
    
    **3. Start Ollama on Port 11435:**
    ```bash
    set OLLAMA_HOST=127.0.0.1:11435
    ollama serve
    ```
    
    **4. Install Python packages:**
    ```bash
    pip install streamlit feedparser requests beautifulsoup4
    ```
    
    **5. Run App:**
    ```bash
    streamlit run app_realtime.py
    ```
    
    ### ⚡ Performance
    
    - Fetch: 5-10 seconds (7 RSS feeds)
    - Curate: 20-60 seconds (Mistral filtering)
    - Total: ~30-70 seconds for real news
    
    ### 🔍 What's Real vs AI?
    
    | Component | Real? | Source |
    |-----------|-------|--------|
    | News Headlines | ✅ YES | Live RSS feeds |
    | News Links | ✅ YES | Original sources |
    | News Summary | ✅ YES | Original articles |
    | Curation | 🤖 AI | Mistral 7B |
    | Relevance | 🤖 AI | Mistral analysis |
    | Persona Filtering | 🤖 AI | Mistral judgment |
    
    ### ⚠️ Troubleshooting
    
    **Q: Port 11434 is stuck?**
    A: The app now uses port 11435 by default. To use it:
    ```bash
    set OLLAMA_HOST=127.0.0.1:11435
    ollama serve
    ```
    
    **Q: Still getting connection errors?**
    A: Make sure the ollama serve terminal is still open and showing "Listening on 127.0.0.1:11435"
    
    ---
    
    **Status**: Production Ready | 100% Real News + AI Curation | Free & Local
    """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.caption("Plan A - Real-Time AI News Curator | Live RSS Feeds + Mistral Curation | 100% Free & Open Source | Using Port 11435")