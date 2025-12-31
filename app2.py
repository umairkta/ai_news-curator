import streamlit as st
import requests
import feedparser
from datetime import datetime
import time
from utils2 import classify_with_mistral, fetch_news_from_rss

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="AI News Curator - Plan B",
    page_icon="📰",
    layout="wide"
)

# ============ STYLING ============
st.markdown("""
<style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
    }
    .persona-title {
        font-size: 2em;
        font-weight: bold;
        color: #2ca02c;
    }
    .article-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .article-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #1f77b4;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# ============ TITLE & HEADER ============
st.markdown('<div class="main-title">📰 AI News Curator - Plan B</div>', unsafe_allow_html=True)
st.markdown("*Real-time AI news | Mistral 7B | 100% Local & Private | $0 Cost*")

# =========== PERSONAS ============
PERSONAS = {
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

# ============ INITIALIZE SESSION STATE ============
if "articles" not in st.session_state:
    st.session_state.articles = []
if "persona" not in st.session_state:
    st.session_state.persona = None
if "relevant_articles" not in st.session_state:
    st.session_state.relevant_articles = []

# ============ SIDEBAR CONFIG ============
st.sidebar.title("⚙️ Configuration")

with st.sidebar:
    st.markdown("### 👤 Select Your Persona")
    selected_persona = st.selectbox(
        "Choose your professional role:",
        list(PERSONAS.keys()),
        index=0
    )
    
    st.markdown("### 📊 Number of Updates")
    num_articles = st.slider(
        "How many articles do you want?",
        min_value=1,
        max_value=8,
        value=3,
        step=1
    )
    
    st.markdown("### 🤖 AI Model")
    st.info("**Using:** Mistral 7B\n**Status:** ✅ Running locally\n**Port:** 11434")
    
    st.markdown("### ℹ️ System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Persona", selected_persona[:15])
    with col2:
        st.metric("Articles", num_articles)

# ============ MAIN APP LAYOUT ============
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f'<div class="persona-title">📖 Curated for: {selected_persona}</div>', unsafe_allow_html=True)

with col3:
    fetch_button = st.button("🔄 Fetch & Curate", key="fetch_btn", use_container_width=True)

# ============ FETCH AND CLASSIFY LOGIC ============
if fetch_button:
    # Step 1: Fetch articles
    with st.spinner("🔍 Fetching real AI news from 3 sources..."):
        articles = fetch_news_from_rss()
        st.session_state.articles = articles
    
    # Show success
    st.markdown(f'<div class="status-box status-success">✅ Fetched {len(articles)} articles from RSS feeds</div>', 
              unsafe_allow_html=True)
    
    # Step 2: Classify articles
    with st.spinner(f"🤖 Classifying with Mistral 7B for {selected_persona}..."):
        relevant_articles = []
        progress_bar = st.progress(0)
        
        for idx, article in enumerate(articles):
            is_relevant = classify_with_mistral(
                article["title"],
                article["summary"],
                selected_persona
            )
            
            if is_relevant:
                relevant_articles.append(article)
            
            progress = (idx + 1) / len(articles)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        st.session_state.relevant_articles = relevant_articles
        st.session_state.persona = selected_persona

# ============ DISPLAY RESULTS ============
if st.session_state.relevant_articles and st.session_state.persona == selected_persona:
    articles = st.session_state.relevant_articles
    
    # Success message
    st.markdown(f'<div class="status-box status-success">✅ Curated {len(articles)} articles for {selected_persona}</div>', 
              unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display articles
    if articles:
        for idx, article in enumerate(articles[:num_articles], 1):
            with st.container():
                # Article card
                st.markdown(f'<div class="article-container">', unsafe_allow_html=True)
                
                # Title
                st.markdown(f'<div class="article-title">{idx}. {article["title"]}</div>', 
                          unsafe_allow_html=True)
                
                # Summary
                st.write(article["summary"])
                
                # Metadata
                col1, col2, col3 = st.columns([1.5, 1.5, 1])
                
                with col1:
                    st.markdown(f"📰 **Source:** {article['source']}")
                
                with col2:
                    st.markdown(f"👤 **For:** {selected_persona}")
                
                with col3:
                    st.markdown(f"[🔗 Read Full]({article['link']})")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("ℹ️ No relevant articles found for this persona.")

# ============ FOOTER ============
st.markdown("---")
footer_text = """
**Plan B - AI News Curator**
- Built with: Streamlit + Mistral 7B + Ollama
- 3 RSS News Sources (HuggingFace, OpenAI, DeepMind)
- Free & Private (100% local)
- Using Port 11434 (Ollama)
"""
st.markdown(footer_text)

with st.expander("📚 About This App"):
    st.markdown("""
    ### Features
    - **Local Processing**: All classification on your computer
    - **Zero Cost**: Free Mistral 7B model
    - **Private**: No data sent anywhere
    - **Real News**: Actual RSS feeds
    - **Customizable**: Easy to extend
    
    ### Technology Stack
    - **Ollama**: Local LLM server
    - **Mistral 7B**: 7 billion parameter model
    - **Streamlit**: Python web framework
    - **Feedparser**: RSS feed parsing
    - **Utils**: Custom utility functions
    """)
