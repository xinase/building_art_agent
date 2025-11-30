import streamlit as st
import os
from datetime import datetime


# === 必须在最开头初始化 session state ===
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel



# 设置页面基本信息
st.set_page_config(page_title="《构建之法》智能助教", layout="wide")
st.title("《构建之法》智能助教 (本地测试版)")

@st.cache_resource
def init_system():
    print("\n[DEBUG] --- 系统初始化开始 ---")
    persist_dir = "./chroma_db"
    knowledge_dir = "./knowledge_base"  # 移到函数开头

    # 1. 初始化 Embedding 模型
    print("[DEBUG] 正在加载 Embedding 模型...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
    except Exception as e:
        st.error(f"模型加载失败，请检查网络或缓存: {e}")
        return None

    # 2. 检查是否需要重建（新增的逻辑）
    need_rebuild = needs_rebuild(persist_dir, knowledge_dir)
    
    # 如果不需要重建且数据库存在，直接加载
    if not need_rebuild:
        db_file_path = os.path.join(persist_dir, "chroma.sqlite3")
        if os.path.exists(db_file_path):
            print(f"[DEBUG] ✅ 发现本地数据库 ({db_file_path})，正在直接加载...")
            try:
                vectordb = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings
                )
                print("[DEBUG] 本地数据库加载成功！")
                return vectordb
            except Exception as e:
                print(f"[DEBUG] ⚠️ 本地数据库加载出错，将尝试重新构建: {e}")
                need_rebuild = True

    # 3. 需要重建或数据库不存在
    print("[DEBUG] ⚠️ 开始构建/更新向量数据库 (这可能需要一些时间)...")

    if not os.path.exists(knowledge_dir):
        st.error(f"知识库文件夹不存在：{knowledge_dir}")
        return None

    # 定义支持 GBK 的加载器
    class CustomTextLoader(TextLoader):
        def __init__(self, file_path: str):
            super().__init__(file_path, encoding="gbk")
        def lazy_load(self):
            try:
                yield from super().lazy_load()
            except Exception as e:
                # 如果 GBK 失败，尝试 utf-8 容错
                try:
                    self.encoding = "utf-8"
                    yield from super().lazy_load()
                except Exception as e2:
                    st.warning(f"无法读取文件 {self.file_path}: {e}")
                    return

    # 加载文件
    print("[DEBUG] 开始扫描并加载文档...")
    loader = DirectoryLoader(
        knowledge_dir,
        glob="**/*.txt",
        loader_cls=CustomTextLoader,
        show_progress=True
    )

    documents = loader.load()
    if not documents:
        st.error("❌ 没有加载到任何文档，请检查 knowledge_base 文件夹")
        return None
    print(f"[DEBUG] 成功加载 {len(documents)} 个文档片段")

    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"[DEBUG] 文本已切分为 {len(texts)} 个块")

    # 构建并保存数据库
    print("[DEBUG] 正在计算向量并写入数据库 (Chroma)...")
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("[DEBUG] ✅ 数据库构建完成并已保存！")

    return vectordb

def needs_rebuild(persist_dir, knowledge_dir):
    """
    检查是否需要重新构建向量数据库
    返回 True 如果需要重建，False 如果可以使用现有数据库
    """
    # 检查数据库目录是否存在
    if not os.path.exists(persist_dir):
        print("[DEBUG] 向量数据库目录不存在，需要构建")
        return True
    
    # 检查数据库文件是否存在
    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.exists(db_file):
        print("[DEBUG] 向量数据库文件不存在，需要构建")
        return True
    
    # 检查知识库目录是否存在
    if not os.path.exists(knowledge_dir):
        print("[DEBUG] 知识库目录不存在")
        return False
    
    # 获取数据库的最后修改时间
    try:
        db_mtime = os.path.getmtime(db_file)
    except OSError:
        print("[DEBUG] 无法获取数据库文件修改时间，需要重建")
        return True
    
    # 遍历知识库中的所有txt文件，检查是否有文件比数据库更新
    for root, dirs, files in os.walk(knowledge_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime > db_mtime:
                        print(f"[DEBUG] 检测到更新的文件: {file}，需要重建数据库")
                        return True
                except OSError:
                    # 如果无法获取某个文件的修改时间，继续检查其他文件
                    continue
    
    print("[DEBUG] 知识库无更新，使用现有数据库")
    return False

# 初始化系统
vectordb = init_system()

def create_retrieval_chain(retriever):
    # 修复后的 LCEL 链：
    # 1. 并行执行：检索文档(source_documents) 和 透传问题(current_query)
    step1 = RunnableParallel(
        source_documents=retriever,
        current_query=RunnablePassthrough()
    )

    # 2. 如果未来接入 LLM，可以在这里用 .assign() 添加 context 和 prompt
    # 目前 MVP 阶段，我们只需要 step1 的结果来展示检索到的内容
    return step1

if vectordb:
    # 侧边栏配置
    with st.sidebar:
        st.header("设置")
        k_val = st.slider("检索文档数量 (K)", min_value=3, max_value=10, value=5)
        # 搜索历史区域
        st.header("📚 搜索历史")
    
        if st.session_state.search_history:
            # 显示最近的搜索记录（最新的在前面）
            for i, history_item in enumerate(reversed(st.session_state.search_history[-10:])):  # 只显示最近10条
                query, timestamp = history_item
                time_str = timestamp.strftime("%H:%M")
                
                # 点击历史记录可以重新搜索
                if st.button(f"{i+1}. {query}", key=f"history_{i}"):
                    st.session_state.current_query = query
                    st.rerun()
        else:
            st.caption("暂无搜索历史")

    retriever = vectordb.as_retriever(search_kwargs={"k": k_val})
    retrieval_chain = create_retrieval_chain(retriever)

    # 主界面输入
    current_query = st.text_input("请输入关于《构建之法》的问题：", placeholder="例如：什么是结对编程？")

    if current_query:
        # 添加到搜索历史（避免重复添加相同的查询）
        if not st.session_state.search_history or st.session_state.search_history[-1][0] != current_query:
            st.session_state.search_history.append((current_query, datetime.now()))
        
        with st.spinner('正在书中为您寻找答案...'):
            result = retrieval_chain.invoke(current_query)

    if current_query:
        with st.spinner('正在书中为您寻找答案...'):
            # 执行检索
            result = retrieval_chain.invoke(current_query)

        # 结果展示区
        st.subheader("📖 书中相关原文片段：")

        # 结果校验
        if not result.get('source_documents'):
            st.warning("没有找到相关文档。")
        else:
            for i, doc in enumerate(result['source_documents']):
                source_name = os.path.basename(doc.metadata.get('source', '未知文件'))
                with st.expander(f"参考片段 {i+1} (来源: {source_name})", expanded=True):
                    st.markdown(f"**原文内容：**\n\n{doc.page_content}")
                    st.caption(f"元数据: {doc.metadata}")
else:
    st.info("系统正在初始化，请查看终端输出...")

