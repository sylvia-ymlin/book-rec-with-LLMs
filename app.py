import gradio as gr
import logging
from src.recommender import BookRecommender
from src.utils import setup_logger

logger = setup_logger(__name__)

# Initialize Recommender
try:
    recommender = BookRecommender()
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    raise RuntimeError("Failed to initialize application data")

def recommend_books(query: str, category: str, tone: str):
    """Wrapper for the recommender engine"""
    try:
        results = recommender.get_recommendations(query, category, tone)
        # Transform results for Gradio Gallery (image_path, caption)
        return [(item["thumbnail"], item["caption"]) for item in results]
    except Exception as e:
        logger.error(f"Error in recommend_books: {e}")
        return []

# Get options from recommender
categories = recommender.get_categories()
tones = recommender.get_tones()

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Glass(),
    title="📚 Book Recommendation System",
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .gallery {
        height: 400px !important;
    }
    """
) as dashboard:
    
    # Dashboard Title and Description
    gr.Markdown(
        """
        # 📚 Intelligent Book Recommendation System
        
        Welcome to the Book Recommendation Dashboard! This AI-powered system uses semantic search, 
        emotion analysis, and machine learning to provide personalized book recommendations.
        
        **How it works:**
        1. Enter a description of the type of book you're looking for
        2. Select your preferred category and emotional tone
        3. Get personalized recommendations powered by 5,000+ books
        """
    )

    # Input Section
    with gr.Row():
        with gr.Column(scale=2):
            # Textbox for user query
            query_input = gr.Textbox(
                label="📝 Describe the type of book you're looking for",
                placeholder="e.g., A thrilling mystery novel set in Victorian London with complex characters",
                lines=3,
                info="Be as specific as possible for better recommendations"
            )
            
        with gr.Column(scale=1):
            # Dropdown for category selection
            category_input = gr.Dropdown(
                label="📚 Select Category",
                choices=categories,
                value="All",
                info="Filter by book category"
            )
            
            # Dropdown for tone selection
            tone_input = gr.Dropdown(
                label="😊 Select Emotional Tone",
                choices=tones,
                value="All",
                info="Filter by emotional content"
            )
    
    # Action buttons
    with gr.Row():
        recommend_button = gr.Button(
            "🔍 Get Recommendations", 
            variant="primary",
            size="lg"
        )
        clear_button = gr.Button("🗑️ Clear", variant="secondary")

    # Results section
    gr.Markdown("## 📖 Your Personalized Recommendations")
    
    # Status indicator
    status_text = gr.Textbox(
        label="Status",
        value="Ready to generate recommendations!",
        interactive=False,
        visible=False
    )
    
    # Output gallery
    output_gallery = gr.Gallery(
        label="Recommended Books",
        show_label=True,
        elem_id="gallery",
        columns=5, 
        rows=2,
        height=400,
        object_fit="cover"
    )

    # Define button behaviors
    def clear_inputs():
        return "", "All", "All", "Inputs cleared!"
    
    def show_loading():
        return "🔄 Generating recommendations... Please wait."
    
    recommend_button.click(
        fn=show_loading,
        outputs=status_text,
        queue=False
    ).then(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=output_gallery,
    ).then(
        fn=lambda: "✅ Recommendations generated successfully!",
        outputs=status_text
    )
    
    clear_button.click(
        fn=clear_inputs,
        outputs=[query_input, category_input, tone_input, status_text]
    )

# Launch configuration
if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
