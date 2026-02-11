import gradio as gr
import logging
from typing import List, Tuple
from src.recommender import BookRecommender
from src.utils import setup_logger

# --- 初始化与配置 ---
logger = setup_logger(__name__)

try:
    recommender = BookRecommender()
    categories = recommender.get_categories()
    tones = recommender.get_tones()
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    # 提供备选方案以防初始化失败
    recommender = None
    categories = ["All", "Fiction", "Non-Fiction", "Sci-Fi", "Mystery"]
    tones = ["All", "Happy", "Dark", "Inspiring", "Thoughtful"]

# --- 业务逻辑函数 ---
def recommend_books(query: str, category: str, tone: str) -> List[Tuple[str, str]]:
    """包装推荐引擎，返回 Gradio Gallery 格式数据"""
    try:
        if not query or not query.strip():
            return []
        if recommender is None:
            return []
        results = recommender.get_recommendations(query, category, tone)
        # 将结果转换为 (图片路径, 描述文本) 的元组列表
        return [(item["thumbnail"], f"{item['title']}\n{item['authors']}") for item in results]
    except Exception as e:
        logger.error(f"Error in recommend_books: {e}")
        return []

def clear_all():
    """重置所有输入和状态"""
    return "", "All", "All", []

# --- 构建界面 (Gradio 6.0 兼容) ---
with gr.Blocks(title="AI 图书智能推荐系统") as dashboard:
    
    # 头部区域
    gr.Markdown(
        """
        # 📚 Intelligent Book Discovery
        
        探索属于你的文字灵魂。基于向量检索与深度情感分析技术。
        """
    )

    # 输入区域
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="📖 描述您想看的书",
                placeholder="例如：一本关于星际旅行的硬科幻，带有孤独感和哲学思考...",
                lines=4
            )
        with gr.Column(scale=1):
            category_input = gr.Dropdown(
                label="图书分类",
                choices=categories,
                value="All"
            )
            tone_input = gr.Dropdown(
                label="情感偏好",
                choices=tones,
                value="All"
            )
    
    with gr.Row():
        recommend_button = gr.Button("🔍 获取智能推荐", variant="primary")
        clear_button = gr.Button("🗑️ 清空条件", variant="secondary")

    # 结果展示区域
    gr.Markdown("## 📖 为您精心挑选")
    
    # 结果画廊
    output_gallery = gr.Gallery(
        label="推荐结果",
        show_label=False,
        elem_id="gallery",
        columns=4,
        rows=2,
        height="auto",
        object_fit="contain"
    )

    # --- 交互逻辑绑定 ---
    recommend_button.click(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=output_gallery,
    )
    
    clear_button.click(
        fn=clear_all,
        outputs=[query_input, category_input, tone_input, output_gallery]
    )

# --- 启动服务 ---
if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )