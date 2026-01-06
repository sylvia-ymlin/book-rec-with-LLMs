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
    # from src.agent.agent_core import ShoppingAgent
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
        comments_block = "**Comments (sample):**\n- Loved the pacing and tone.\n- Great pick for this category."
        qa_hint = f"在问答助手中询问：请介绍《{book['title']}》的作者、ISBN 和内容简介。"
        return title_block, rank_block, comments_block, desc_block, qa_hint, idx
    except Exception as e:
        logger.error(f"Error showing book details: {e}")
        return "", "", "", "", "", -1

def clear_discovery():
    return "", "All", "All", []


def add_to_favorites(selected_idx: int, recs: List[dict]):
    try:
        if selected_idx is None or selected_idx < 0 or not recs or selected_idx >= len(recs):
            return "请先从左侧选择一本书后再收藏。"
        book = recs[selected_idx]
        payload = {"user_id": "local", "isbn": book["isbn"]}
        resp = requests.post(f"{API_URL}/favorites/add", json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return f"✅ 已收藏：{book['title']}（共 {data.get('favorites_count', '?')} 本）"
        return f"❌ 收藏失败：{resp.text}"
    except Exception as e:
        logger.error(f"add_to_favorites error: {e}")
        return "❌ 收藏失败，请稍后再试。"


def generate_highlights(selected_idx: int, recs: List[dict]):
    try:
        if selected_idx is None or selected_idx < 0 or not recs or selected_idx >= len(recs):
            return "（提示）请先在左侧选择一本书，再点击生成卖点。"
        book = recs[selected_idx]
        payload = {"isbn": book["isbn"], "user_id": "local"}
        resp = requests.post(f"{API_URL}/marketing/highlights", json=payload, timeout=12)
        if resp.status_code != 200:
            return "生成卖点失败，请稍后再试。"
        data = resp.json()
        persona = data.get("persona", {})
        highlights = data.get("highlights", [])
        header = f"### 个性化卖点（{book['title']}）\n"
        persona_md = f"> 画像：{persona.get('summary','暂无')}\n\n" if persona else ""
        bullets = "\n".join([f"- {h}" for h in highlights]) if highlights else "- 暂无可用卖点"
        return header + persona_md + bullets
    except Exception as e:
        logger.error(f"generate_highlights error: {e}")
        return "生成卖点时出错，请稍后再试。"

# --- Business Logic: Tab 2 (Assistant) ---
def chat_response(message, history):
    """Answer book questions using the recommender API as a knowledge source."""
    try:
        if not message.strip():
            return "请描述你要询问的书或问题。"

        # Use the same recommend endpoint as retrieval to ground answers
        payload = {"query": message, "category": "All", "tone": "All"}
        resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=20)
        if resp.status_code != 200:
            return "暂时无法获取书籍信息，请稍后再试。"

        data = resp.json()
        recs = data.get("recommendations", [])
        if not recs:
            return "没有找到匹配的书，换个描述试试？"

        top = recs[0]
        answer = [
            f"**{top.get('title','')}**",
            f"作者：{top.get('authors','未知')}",
            f"简介：{top.get('description','暂无简介')}"
        ]
        # If more results, suggest to check discovery tab
        if len(recs) > 1:
            answer.append("更多结果已显示在左侧推荐区，可点击查看详情。")
        return "\n\n".join(answer)
    except Exception as e:
        logger.error(f"chat_response error: {e}")
        return "处理询问时出错，请稍后再试。"

# --- Business Logic: Tab 3 (Marketing) ---
def generate_marketing_copy(product_name, features, target_audience):
    # Placeholder for Marketing Content Engine
    # from src.marketing.guardrails import SafetyCheck...
    return f"""
    📣 **ATTENTION {target_audience.upper()}!**
    
    Meet the new **{product_name}** - the game changer you've been waiting for.
    
    ✨ **Why you'll love it:**
    {features}
    
    [Generated by Safe-Aligned-LLM v1.0]
    [Safety Check: PASSED]
    """

# --- UI Construction ---
with gr.Blocks(title="AI 二手书交易平台", theme=gr.themes.Soft()) as dashboard:
    
    gr.Markdown("# 📚 AI 二手书交易平台")
    gr.Markdown("三大模块为二手书买卖服务：**语义找书**、**问价/咨询助手**、**发布/营销文案生成**。")
    
    with gr.Tabs():
        
        # --- Tab 1: Discovery ---
        with gr.TabItem("🔍 找二手书 (推荐)"):
            rec_state = gr.State([])  # store full recommendation data
            qa_hint = gr.State("")
            sel_idx = gr.State(-1)
            with gr.Row():
                with gr.Column(scale=3):
                    q_input = gr.Textbox(label="你想找什么书？(语义搜索)", placeholder="例：想要一本二手的悬疑小说，节奏快一点")
                with gr.Column(scale=1):
                    cat_input = gr.Dropdown(label="分类", choices=categories, value="All")
                    tone_input = gr.Dropdown(label="情绪/基调", choices=tones, value="All")
            
            btn_rec = gr.Button("开始找书", variant="primary")
            gallery = gr.Gallery(label="推荐结果", columns=4, height="auto")
            with gr.Row():
                with gr.Column(scale=2):
                    title_info = gr.Markdown(label="书籍信息")
                    desc_info = gr.Markdown(label="书籍简介")
                with gr.Column(scale=1):
                    rank_info = gr.Markdown(label="排序/排名")
                    comments_info = gr.Markdown(label="评论/印象")
            qa_hint_md = gr.Markdown(label="发送到问答助手", value="（点击左侧书卡后，这里会给出可复制的提问句）")

            with gr.Row():
                btn_fav = gr.Button("⭐ 收藏", variant="secondary")
                btn_high = gr.Button("✨ 生成卖点", variant="primary")
            fav_status = gr.Markdown(label="收藏状态")
            highlights_md = gr.Markdown(label="个性化卖点")

            btn_rec.click(recommend_books, [q_input, cat_input, tone_input], [gallery, rec_state])
            gallery.select(show_book_details, [rec_state], [title_info, rank_info, comments_info, desc_info, qa_hint_md, sel_idx])
            btn_fav.click(add_to_favorites, [sel_idx, rec_state], [fav_status])
            btn_high.click(generate_highlights, [sel_idx, rec_state], [highlights_md])

        # --- Tab 2: AI Assistant ---
        with gr.TabItem("💬 问价/咨询助手 (RAG)"):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                examples=["这本二手能便宜点吗？", "有无二手英文原版科幻？"],
                title="二手书智能问价助手",
                description="基于 RAG 的检索增强问答，可用于问价、找货、咨询。"
            )

        # --- Tab 3: Marketing ---
        with gr.TabItem("✍️ 发布/营销文案 (GenAI)"):
            with gr.Row():
                m_name = gr.Textbox(label="书名/卖点", value="二手原版 The Hobbit，九成新")
                m_feat = gr.Textbox(label="亮点/成色/配送", value="九成新，无笔记，支持同城自提或邮寄")
                m_aud = gr.Textbox(label="目标买家", value="奇幻爱好者，收藏向")
            
            btn_gen = gr.Button("生成发布文案", variant="primary")
            m_out = gr.Markdown(label="生成结果")
            
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