#!/bin/bash

echo "🚀 准备部署到 Hugging Face Spaces..."

# 检查必要文件
echo "📋 检查必要文件..."
required_files=("gradio-dashboard.py" "books_with_emotions.csv" "books_descriptions.txt" "cover-not-found.jpg" "requirements.txt")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
        exit 1
    fi
done

# 重命名主文件为 app.py (Hugging Face 标准)
if [ -f "gradio-dashboard.py" ]; then
    cp gradio-dashboard.py app.py
    echo "✅ 已创建 app.py"
fi

# 检查 Git 状态
echo "📝 检查 Git 状态..."
if [ -d ".git" ]; then
    echo "✅ Git 仓库已初始化"
    git status
else
    echo "⚠️  未检测到 Git 仓库，请先运行："
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m '准备部署'"
    echo "   git remote add origin https://github.com/你的用户名/book-recommender.git"
    echo "   git push -u origin main"
fi

echo ""
echo "🎯 下一步操作："
echo "1. 访问 https://huggingface.co/spaces"
echo "2. 点击 'Create new Space'"
echo "3. 选择 'Gradio' SDK"
echo "4. 连接你的 GitHub 仓库"
echo "5. 在 Settings 中添加 HUGGINGFACEHUB_API_TOKEN"
echo "6. 等待自动部署完成"
echo ""
echo "📖 详细说明请查看 HUGGINGFACE_DEPLOYMENT.md"
