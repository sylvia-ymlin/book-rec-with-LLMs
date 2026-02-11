import gradio as gr
import logging
import os
import requests
import json
from typing import List, Tuple, Any
from src.utils import setup_logger

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://localhost:6006")  # Localhost via SSH Tunnel

# --- Initialize Logger ---
logger = setup_logger(__name__)

# --- Module Initialization ---
# (We no longer load model locally; we query the remote API)
categories = ["All", "Fiction", "History", "Science", "Technology"] # Fallback/Mock for now
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

def fetch_tones():
    try:
        resp = requests.get(f"{API_URL}/tones", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            tns = data.get("tones") if isinstance(data, dict) else None
            if isinstance(tns, list) and len(tns) > 0:
                return tns
    except Exception as e:
        logger.warning(f"fetch_tones failed: {e}")
    return tones

def fetch_categories():
    try:
        resp = requests.get(f"{API_URL}/categories", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            cats = data.get("categories") if isinstance(data, dict) else None
            if isinstance(cats, list) and len(cats) > 0:
                return cats
    except Exception as e:
        logger.warning(f"fetch_categories failed: {e}")
    return categories

# Try to fetch real categories on startup
categories = fetch_categories()
tones = fetch_tones()

# Initialize Shopping Agent (Mock or Real)
# Note: Real agent requires FAISS index. We'll handle checks later.
try:
    # from legacy.agent.agent_core import ShoppingAgent
    # shopping_agent = ShoppingAgent(...)
    pass 
except ImportError:
    logger.warning("Shopping Agent module not found or failed to import.")

# --- Business Logic: Tab 1 (Discovery) ---
def recommend_books(query: str, category: str, tone: str):
    """Fetch recommendations and return both gallery items and raw data."""
    try:
        if not query.strip():
            return [], []
        
        payload = {
            "query": query,
            "category": category if category else "All",
            "tone": tone if tone else "All"
        }
        
        logger.info(f"Sending request to {API_URL}/recommend")
        response = requests.post(f"{API_URL}/recommend", json=payload, timeout=25)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("recommendations", [])
            gallery_items = [(item["thumbnail"], f"{item['title']}\n{item['authors']}") for item in results]
            return gallery_items, results
        else:
            logger.error(f"API Error: {response.text}")
            return [], []
            
    except Exception as e:
        logger.error(f"Error in recommend_books: {e}")
        return [], []


def show_book_details(evt: Any, recs: List[dict]):
    """Populate detail panel when a gallery item is selected and prep a QA hint."""
    try:
        if recs is None:
            return "", "", "", "", "", -1
        idx = evt.index if evt and hasattr(evt, "index") else None
        if idx is None or idx >= len(recs):
            return "", "", "", "", "", -1
        book = recs[idx]
        title_block = f"### {book['title']}\n**Authors:** {book['authors']}\n**ISBN:** {book['isbn']}"
        desc_block = f"**Description**\n\n{book['description']}"
        rank_block = f"**Rank:** #{idx + 1}"  # simple positional rank
        comments_block = "**Reviews (sample):**\n- Exceptional pacing and character depth.\n- A must-read for this genre."
        qa_hint = f"Ask the assistant: Tell me more about '{book['title']}' by {book['authors']}."
        return title_block, rank_block, comments_block, desc_block, qa_hint, idx
    except Exception as e:
        logger.error(f"Error showing book details: {e}")
        return "", "", "", "", "", -1

def clear_discovery():
    return "", "All", "All", []


def add_to_favorites(selected_idx: int, recs: List[dict]):
    try:
        if selected_idx is None or selected_idx < 0 or not recs or selected_idx >= len(recs):
            return "Please select a book from the gallery first."
        book = recs[selected_idx]
        payload = {"user_id": "local", "isbn": book["isbn"]}
        resp = requests.post(f"{API_URL}/favorites/add", json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return f"✅ Added to favorites: {book['title']} ({data.get('favorites_count', '?')} books in collection)"
        return f"❌ Failed to add: {resp.text}"
    except Exception as e:
        logger.error(f"add_to_favorites error: {e}")
        return "❌ Error adding to favorites. Try again later."


def generate_highlights(selected_idx: int, recs: List[dict]):
    try:
        if selected_idx is None or selected_idx < 0 or not recs or selected_idx >= len(recs):
            return "(Hint) Please select a book from the gallery, then click Generate Highlights."
        book = recs[selected_idx]
        payload = {"isbn": book["isbn"], "user_id": "local"}
        resp = requests.post(f"{API_URL}/marketing/highlights", json=payload, timeout=12)
        if resp.status_code != 200:
            return "Failed to generate highlights. Try again later."
        data = resp.json()
        persona = data.get("persona", {})
        highlights = data.get("highlights", [])
        header = f"### Personalized Highlights ({book['title']})\n"
        persona_md = f"> Your Profile: {persona.get('summary','N/A')}\n\n" if persona else ""
        bullets = "\n".join([f"- {h}" for h in highlights]) if highlights else "- No highlights available"
        return header + persona_md + bullets
    except Exception as e:
        logger.error(f"generate_highlights error: {e}")
        return "Error generating highlights. Try again later."

# --- Business Logic: Tab 2 (Assistant) ---
def chat_response(message, history):
    """Answer book questions using the recommender API as a knowledge source."""
    try:
        if not message.strip():
            return "Please describe the book or question you have."

        # Use the same recommend endpoint as retrieval to ground answers
        payload = {"query": message, "category": "All", "tone": "All"}
        resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=20)
        if resp.status_code != 200:
            return "Unable to retrieve book information. Try again later."

        data = resp.json()
        recs = data.get("recommendations", [])
        if not recs:
            return "No matching books found. Try a different query."

        top = recs[0]
        answer = [
            f"**{top.get('title','')}**",
            f"Author: {top.get('authors','Unknown')}",
            f"Summary: {top.get('description','No summary available')}"
        ]
        # If more results, suggest to check discovery tab
        if len(recs) > 1:
            answer.append("More results available in the Find Books tab.")
        return "\n\n".join(answer)
    except Exception as e:
        logger.error(f"chat_response error: {e}")
        return "Error processing your question. Try again later."

# --- Business Logic: Tab 3 (Marketing) ---
def generate_marketing_copy(product_name, features, target_audience):
    # Placeholder for Marketing Content Engine
    # from src.marketing.guardrails import SafetyCheck...
    return f"""
    📣 **CALLING ALL {target_audience.upper()}!**
    
    Presenting **{product_name}** — the treasure you've been seeking.
    
    ✨ **Why you'll love it:**
    {features}
    
    Perfect for your collection. Add it to your shelf today.
    """

# --- UI Construction ---
with gr.Blocks(title="Paper Shelf - Book Discovery", theme=gr.themes.Soft()) as dashboard:
    
    gr.Markdown("# 📚 Paper Shelf")
    gr.Markdown("Intelligent book discovery powered by semantic search: **Find Books**, **Ask Questions**, **Generate Marketing Copy**.")
    
    with gr.Tabs():
        
        # --- Tab 1: Discovery ---
        with gr.TabItem("🔍 Find Books (Search & Recommendations)"):
            rec_state = gr.State([])  # store full recommendation data
            qa_hint = gr.State("")
            sel_idx = gr.State(-1)
            with gr.Row():
                with gr.Column(scale=3):
                    q_input = gr.Textbox(label="What are you looking for?", placeholder="e.g., a mystery novel with fast pacing")
                with gr.Column(scale=1):
                    cat_input = gr.Dropdown(label="Category", choices=categories, value="All")
                    tone_input = gr.Dropdown(label="Mood/Tone", choices=tones, value="All")
            
            btn_rec = gr.Button("Find Books", variant="primary")
            gallery = gr.Gallery(label="Results", columns=4, height="auto")
            with gr.Row():
                with gr.Column(scale=2):
                    title_info = gr.Markdown(label="Book Info")
                    desc_info = gr.Markdown(label="Description")
                with gr.Column(scale=1):
                    rank_info = gr.Markdown(label="Ranking")
                    comments_info = gr.Markdown(label="Reviews")
            qa_hint_md = gr.Markdown(label="Ask the Assistant", value="(Click a book to see suggested questions)")

            with gr.Row():
                btn_fav = gr.Button("⭐ Add to Favorites", variant="secondary")
                btn_high = gr.Button("✨ Generate Highlights", variant="primary")
            fav_status = gr.Markdown(label="Status")
            highlights_md = gr.Markdown(label="Personalized Highlights")

            btn_rec.click(recommend_books, [q_input, cat_input, tone_input], [gallery, rec_state])
            gallery.select(show_book_details, [rec_state], [title_info, rank_info, comments_info, desc_info, qa_hint_md, sel_idx])
            btn_fav.click(add_to_favorites, [sel_idx, rec_state], [fav_status])
            btn_high.click(generate_highlights, [sel_idx, rec_state], [highlights_md])

        # --- Tab 2: AI Assistant ---
        with gr.TabItem("💬 Ask Questions (RAG Assistant)"):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                examples=["Is there a mystery with time travel?", "Recommend sci-fi with female protagonists"],
                title="Intelligent Book Assistant",
                description="Search and learn about books through conversational AI."
            )

        # --- Tab 3: Marketing ---
        with gr.TabItem("✍️ Create Marketing Copy (GenAI)"):
            with gr.Row():
                m_name = gr.Textbox(label="Book Title/Hook", value="The Hobbit - First Edition, Near Mint")
                m_feat = gr.Textbox(label="Key Features/Condition", value="Near mint condition, no markings, ships worldwide")
                m_aud = gr.Textbox(label="Target Audience", value="Fantasy enthusiasts, collectors")
            
            btn_gen = gr.Button("Generate Listing", variant="primary")
            m_out = gr.Markdown(label="Generated Copy")
            
            btn_gen.click(generate_marketing_copy, [m_name, m_feat, m_aud], m_out)

if __name__ == "__main__":
    import os
    assets_path = os.path.join(os.path.dirname(__file__), "assets")
    dashboard.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        allowed_paths=[assets_path],
        share=True
    )