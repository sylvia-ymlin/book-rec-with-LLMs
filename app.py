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
    categories = ["All", "Fiction", "Non-Fiction", "Sci-Fi", "Mystery"]
    tones = ["All", "Happy", "Dark", "Inspiring", "Thoughtful"]

# --- UI 主题与样式设计 ---
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_label_text_size="*text_sm",
    button_primary_background_fill="linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338CA 0%, #6D28D9 100%)",
    button_primary_text_color="white",
)

custom_css = """
.container { max-width: 1100px !important; margin: 0 auto !important; padding-top: 2rem !important; }
.header-area { 
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
    padding: 2.5rem; 
    border-radius: 1.25rem; 
    color: white; 
    margin-bottom: 2rem; 
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    text-align: center;
}
.header-area h1 { color: white !important; font-weight: 800 !important; font-size: 2.5rem !important; margin-bottom: 0.5rem !important; }
.header-area p { color: #94a3b8 !important; font-size: 1.1rem; }
.input-card { 
    border-radius: 1rem !important; 
    border: 1px solid #e2e8f0 !important; 
    background: white !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; 
}
#gallery { background: transparent !important; border: none !important; margin-top: 1rem; }
#gallery img { border-radius: 12px; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#gallery img:hover { transform: translateY(-5px) scale(1.03); box-shadow: 0 12px 20px rgba(0,0,0,0.15); }
.action-btn { transition: all 0.2s ease !important; font-weight: 600 !important; }
.status-msg { font-style: italic; color: #64748b; margin-top: 10px; }
"""

# --- 业务逻辑函数 ---
def recommend_books(query: str, category: str, tone: str) -> List[Tuple[str, str]]:
    """包装推荐引擎，返回 Gradio Gallery 格式数据"""
    try:
        if not query.strip():
            return []
        results = recommender.get_recommendations(query, category, tone)
        # 将结果转换为 (图片路径, 描述文本) 的元组列表
        return [(item["thumbnail"], f"{item['title']}\n{item['authors']}") for item in results]
    except Exception as e:
        logger.error(f"Error in recommend_books: {e}")
        return []

def clear_all():
    """重置所有输入和状态"""
    return "", "All", "All", "✨ 准备就绪，请输入您的需求", []

# --- 构建界面 ---
with gr.Blocks(theme=custom_theme, css=custom_css, title="AI 图书智能推荐系统") as dashboard:
    with gr.Div(elem_classes="container"):
        
        # 头部区域
        with gr.Div(elem_classes="header-area"):
            gr.Markdown(
                """
                # 📚 Intelligent Book Discovery
                探索属于你的文字灵魂。基于向量检索与深度情感分析技术。
                """
            )

        # 输入卡片区域
        with gr.Group(elem_classes="input-card"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="📖 描述您想看的书",
                        placeholder="例如：一本关于星际旅行的硬科幻，带有孤独感和哲学思考...",
                        lines=4,
                        container=True
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
                recommend_button = gr.Button(
                    "🔍 获取智能推荐", 
                    variant="primary",
                    elem_classes="action-btn"
                )
                clear_button = gr.Button("🗑️ 清空条件", variant="secondary")

        # 结果展示区域
        gr.HTML("<div style='height: 40px;'></div>") # 间距
        with gr.Column():
            gr.Markdown("## 📖 为您精心挑选")
            
            # 状态提示
            status_text = gr.Markdown("✨ 准备就绪，请输入您的需求", elem_classes="status-msg")
            
            # 结果画廊
            output_gallery = gr.Gallery(
                label="推荐结果",
                show_label=False,
                elem_id="gallery",
                columns=[2, 3, 4], # 响应式：手机2列，平板3列，电脑4列
                rows=[2],
                height="auto",
                object_fit="contain",
                preview=True
            )

    # --- 交互逻辑绑定 ---
    
    # 点击推荐按钮后的流程：显示加载 -> 调用算法 -> 更新状态
    recommend_button.click(
        fn=lambda: "🔄 正在深度检索馆藏资源，请稍候...",
        outputs=status_text,
        queue=False
    ).then(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=output_gallery,
    ).then(
        fn=lambda x: "✅ 已为您找到最匹配的 5,000+ 馆藏图书：" if len(x) > 0 else "❌ 未找到完全匹配的图书，尝试换个描述试试？",
        inputs=output_gallery,
        outputs=status_text
    )
    
    # 点击清空按钮
    clear_button.click(
        fn=clear_all,
        outputs=[query_input, category_input, tone_input, status_text, output_gallery]
    )

# --- 启动服务 ---
if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )