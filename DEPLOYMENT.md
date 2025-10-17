# 🚀 Deployment Guide

This guide provides step-by-step instructions for deploying your Book Recommendation System to various platforms.

## 📋 Prerequisites

Before deploying, ensure you have:

- ✅ All data files (`books_with_emotions.csv`, `books_descriptions.txt`, `cover-not-found.jpg`)
- ✅ Hugging Face API token
- ✅ Git repository with your code
- ✅ Python 3.8+ environment

## 🌟 Recommended: Hugging Face Spaces

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** (if not already done):
```bash
git init
git add .
git commit -m "Initial commit: Book Recommendation System"
git remote add origin https://github.com/yourusername/book-recommender.git
git push -u origin main
```

2. **Ensure all required files are present**:
```bash
ls -la
# Should include:
# - gradio-dashboard.py
# - books_with_emotions.csv
# - books_descriptions.txt
# - cover-not-found.jpg
# - requirements.txt
# - README.md
```

### Step 2: Deploy to Hugging Face Spaces

1. **Go to [Hugging Face Spaces](https://huggingface.co/spaces)**
2. **Click "Create new Space"**
3. **Fill in the details**:
   - **Space name**: `book-recommender`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free)
   - **Visibility**: Public
4. **Connect your GitHub repository**
5. **Add your Hugging Face API token**:
   - Go to Space Settings → Repository secrets
   - Add secret: `HUGGINGFACEHUB_API_TOKEN` = `your_token_here`
6. **Deploy!** (This may take 5-10 minutes)

### Step 3: Verify Deployment

- Your app will be available at: `https://huggingface.co/spaces/yourusername/book-recommender`
- Test the functionality with sample queries
- Check logs for any errors

## 🔧 Alternative Deployment Options

### Option 1: Streamlit Cloud

1. **Convert to Streamlit** (requires code modification):
```python
# Create app.py
import streamlit as st
# ... convert Gradio components to Streamlit
```

2. **Deploy via GitHub**:
   - Push to GitHub
   - Connect to [Streamlit Cloud](https://streamlit.io/cloud)
   - Deploy automatically

### Option 2: Heroku

1. **Create Procfile**:
```bash
echo "web: python gradio-dashboard.py" > Procfile
```

2. **Create runtime.txt**:
```bash
echo "python-3.9.16" > runtime.txt
```

3. **Deploy**:
```bash
heroku create your-app-name
git push heroku main
```

### Option 3: Google Cloud Run

1. **Create Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "gradio-dashboard.py"]
```

2. **Deploy**:
```bash
gcloud run deploy --source .
```

### Option 4: AWS/GCP/Azure

#### AWS EC2
```bash
# Launch EC2 instance
# Install dependencies
sudo apt update
sudo apt install python3-pip
pip install -r requirements.txt

# Run application
python gradio-dashboard.py
```

#### Azure App Service
```bash
# Create App Service
# Configure deployment from GitHub
# Set environment variables
```

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required
HUGGINGFACEHUB_API_TOKEN=your_token_here

# Optional
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false
```

### Performance Optimization

For production deployment:

1. **Caching**: Implement Redis caching for embeddings
2. **Database**: Use persistent vector database (Pinecone, Weaviate)
3. **Scaling**: Use load balancers for multiple instances
4. **Monitoring**: Add logging and error tracking

### Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Rate Limiting**: Implement request rate limiting
3. **Input Validation**: Validate user inputs
4. **HTTPS**: Always use HTTPS in production

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   - Check `requirements.txt` includes all dependencies
   - Verify Python version compatibility

2. **"API token not found" errors**:
   - Verify `HUGGINGFACEHUB_API_TOKEN` is set correctly
   - Check token permissions

3. **"File not found" errors**:
   - Ensure all data files are in the repository
   - Check file paths are correct

4. **Memory issues**:
   - Consider using smaller models
   - Implement lazy loading
   - Use CPU-optimized versions

### Debugging

1. **Check logs**:
```bash
# For Hugging Face Spaces
# Check the "Logs" tab in your Space

# For local debugging
python gradio-dashboard.py --debug
```

2. **Test locally first**:
```bash
python gradio-dashboard.py
# Visit http://localhost:7860
```

## 📊 Monitoring and Analytics

### Add Analytics (Optional)

```python
# Add to gradio-dashboard.py
import gradio as gr

# Add analytics tracking
def track_usage(query, category, tone, results_count):
    # Log usage statistics
    logger.info(f"Query: {query}, Results: {results_count}")
```

### Performance Monitoring

- Monitor response times
- Track user interactions
- Monitor error rates
- Set up alerts for failures

## 🔄 Updates and Maintenance

### Updating Your Deployment

1. **Make changes to your code**
2. **Commit and push to GitHub**:
```bash
git add .
git commit -m "Update: Add new features"
git push origin main
```

3. **Redeploy** (automatic for Hugging Face Spaces)

### Backup Strategy

1. **Data backup**: Regular backups of your data files
2. **Code backup**: Version control with Git
3. **Model backup**: Save model checkpoints
4. **Configuration backup**: Document all settings

## 📞 Support

If you encounter issues:

1. **Check the logs** in your deployment platform
2. **Test locally** to isolate issues
3. **Review the documentation** for your chosen platform
4. **Check GitHub issues** for similar problems
5. **Contact support** for your deployment platform

---

**Happy Deploying! 🚀**

Your Book Recommendation System is ready to help readers discover their next favorite book!
