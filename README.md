# 智能助教

基于 LangChain 和 Streamlit 构建的本地智能问答系统，作为 MVP，以《构建之法》书籍内容为例。

## 功能特性

- 📚 **本地知识库**：支持多文档加载和向量化存储
- 🔍 **语义搜索**：基于中文嵌入模型的智能问答
- 💬 **搜索历史**：自动记录和快速重新搜索
- 🚀 **完全本地**：无需外部API，保护隐私
- 📱 **友好界面**：Streamlit Web界面，易于使用

## 技术栈

- **前端界面**: Streamlit
- **向量数据库**: Chroma
- **文本嵌入**: HuggingFace (text2vec-large-chinese)
- **文本处理**: LangChain
- **开发环境**: WSL2 + Python

## 快速开始

1. 克隆项目：
```bash
git clone <你的仓库地址>
cd building_art_agent

安装依赖：

bash
pip install -r requirements.txt
准备知识库：

bash
# 将《构建之法》文本文件放入 knowledge_base/ 目录
运行应用：

bash
streamlit run app.py
项目结构
text
building_art_agent/
├── app.py                 # 主应用文件
├── knowledge_base/        # 知识库文档
├── chroma_db/            # 向量数据库（自动生成）
├── building_art_venv/    # Python虚拟环境
└── README.md
许可证
MIT License
