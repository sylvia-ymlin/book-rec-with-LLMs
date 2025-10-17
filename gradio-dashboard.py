import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize global variables
books = None
db_books = None
embeddings = None

def initialize_data():
    """Initialize the data and models for the application."""
    global books, db_books, embeddings
    
    try:
        logger.info("Loading books data...")
        books = pd.read_csv('books_with_emotions.csv')
        
        # Process thumbnails
        books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna(),
            "cover-not-found.jpg",  # Default image when thumbnail is not available
            books["large_thumbnail"],
        )
        
        logger.info("Loading documents for vector search...")
        raw_documents = TextLoader("books_descriptions.txt").load()
        text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(raw_documents)
        
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        logger.info("Creating vector database...")
        db_books = Chroma.from_documents(
            documents,
            embedding=embeddings
        )
        
        logger.info("Data initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        return False

# Initialize data on startup
if not initialize_data():
    raise RuntimeError("Failed to initialize the application data. Please check your data files and API tokens.")


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 10,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    """Generate book recommendations based on user input."""
    try:
        if not query or not query.strip():
            return []
            
        logger.info(f"Processing recommendation request: query='{query}', category='{category}', tone='{tone}'")
        
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row["authors"]

            caption = f"{row['title']} by {authors_str}: {truncated_description}"
            results.append((row["large_thumbnail"], caption))
            
        logger.info(f"Generated {len(results)} recommendations")
        return results
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return []

# Prepare category options
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


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
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True,
        quiet=False
    )