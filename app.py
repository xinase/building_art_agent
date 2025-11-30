import streamlit as st
import os
import hashlib
import json
from datetime import datetime
import shutil

# === 1. 初始化 Session State (必须在最前) ===
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

st.set_page_config(page_title="《构建之法》智能助教", layout="wide")
st.title("《构建之法》智能助教 (增量更新版)")

# === 工具函数 ===

def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            file_stat = os.stat(file_path)
            combined_data = file_content + str(file_stat.st_size).encode()
            return hashlib.md5(combined_data).hexdigest()
    except Exception as e:
        print(f"[DEBUG] 无法计算文件哈希 {file_path}: {e}")
        return None

def load_file_metadata(persist_dir):
    """加载文件元数据"""
    metadata_file = os.path.join(persist_dir, "file_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[DEBUG] 从 {metadata_file} 加载了 {len(data)} 个文件的元数据")
                return data
        except Exception as e:
            print(f"[DEBUG] 加载元数据失败: {e}")
            return {}
    else:
        print(f"[DEBUG] 元数据文件不存在: {metadata_file}")
    return {}

def save_file_metadata(persist_dir, metadata):
    """保存文件元数据"""
    # 确保目录存在
    os.makedirs(persist_dir, exist_ok=True)
    
    metadata_file = os.path.join(persist_dir, "file_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] ✅ 已保存 {len(metadata)} 个文件的元数据到 {metadata_file}")
    except Exception as e:
        print(f"[DEBUG] ❌ 无法保存文件元数据: {e}")

def get_changed_files(knowledge_dir, existing_metadata):
    """获取需要更新的文件列表"""
    changed_files = []
    new_files = []
    
    if not os.path.exists(knowledge_dir):
        print(f"[DEBUG] 知识库目录不存在: {knowledge_dir}")
        return [], [], []
    
    # 扫描当前磁盘上的文件
    current_files_set = set()
    for root, dirs, files in os.walk(knowledge_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                current_files_set.add(file_path)
                
                current_hash = get_file_hash(file_path)
                if not current_hash: 
                    continue
                
                if file_path in existing_metadata:
                    if existing_metadata[file_path]['hash'] != current_hash:
                        changed_files.append(file_path)  # 内容变了
                        print(f"[DEBUG] 检测到文件修改: {os.path.basename(file_path)}")
                else:
                    new_files.append(file_path)  # 这是一个全新的文件
                    print(f"[DEBUG] 检测到新文件: {os.path.basename(file_path)}")
    
    # 检查有哪些文件在元数据里有，但磁盘上删了
    deleted_files = [f for f in existing_metadata if f not in current_files_set]
    for deleted_file in deleted_files:
        print(f"[DEBUG] 检测到文件删除: {os.path.basename(deleted_file)}")
    
    return changed_files, new_files, deleted_files

def update_vector_database(vectordb, knowledge_dir, existing_metadata):
    """执行增量更新 - 简单进度显示"""
    print("[DEBUG] 🔄 检查增量更新...")
    
    changed_files, new_files, deleted_files = get_changed_files(knowledge_dir, existing_metadata)
    
    if not changed_files and not new_files and not deleted_files:
        print("[DEBUG] ✅ 所有文件已是最新的，无需更新")
        return existing_metadata
    
    print(f"[DEBUG] 变更统计: 新增 {len(new_files)}, 修改 {len(changed_files)}, 删除 {len(deleted_files)}")

    # 1. 处理删除的文件
    for file_path in deleted_files:
        try:
            vectordb._collection.delete(where={"source": file_path})
            if file_path in existing_metadata:
                del existing_metadata[file_path]
            print(f"[DEBUG] ✅ 已删除无效索引: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[DEBUG] ❌ 删除失败: {e}")

    # 2. 处理修改和新增的文件
    files_to_process = changed_files + new_files
    
    # 清理修改文件的旧向量
    for file_path in changed_files:
        try:
            vectordb._collection.delete(where={"source": file_path})
            print(f"[DEBUG] ✅ 已清理旧向量: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[DEBUG] ❌ 清理旧向量失败: {e}")

    # 定义加载器
    class CustomTextLoader(TextLoader):
        def __init__(self, file_path: str):
            super().__init__(file_path, encoding="gbk")
        def lazy_load(self):
            try:
                yield from super().lazy_load()
            except Exception as e:
                try:
                    self.encoding = "utf-8"
                    yield from super().lazy_load()
                except:
                    print(f"[DEBUG] ❌ 无法读取文件 {self.file_path}")
                    return

    if files_to_process:
        print(f"[DEBUG] 正在处理 {len(files_to_process)} 个文件...")
        documents = []
        for file_path in files_to_process:
            try:
                loader = CustomTextLoader(file_path)
                file_docs = loader.load()
                documents.extend(file_docs)
                print(f"[DEBUG] ✅ 成功加载: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[DEBUG] ❌ 跳过文件 {file_path}: {e}")

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            print(f"[DEBUG] 文本已切分为 {len(texts)} 个块")
            
            if texts:
                # 分批写入，每10个块输出一次进度
                batch_size = 10
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                print(f"[DEBUG] 开始分批写入，共 {total_batches} 批...")
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    vectordb.add_documents(batch)
                    
                    current_batch = i // batch_size + 1
                    processed = min(i + batch_size, len(texts))
                    print(f"[DEBUG] ✅ 已完成第 {current_batch}/{total_batches} 批，已处理 {processed}/{len(texts)} 个块")
                
                print(f"[DEBUG] ✅ 所有 {len(texts)} 个文本块已成功添加到数据库")
                
                # 更新元数据
                for file_path in files_to_process:
                    file_hash = get_file_hash(file_path)
                    if file_hash:
                        existing_metadata[file_path] = {
                            'hash': file_hash,
                            'last_updated': datetime.now().isoformat(),
                            'chunk_count': len([t for t in texts if t.metadata.get('source') == file_path])
                        }
    else:
        print("[DEBUG] 没有需要处理的文件")

    return existing_metadata


@st.cache_resource
def init_system():
    print("\n[DEBUG] --- 系统初始化 ---")
    persist_dir = "./chroma_db"
    knowledge_dir = "./knowledge_base"
    
    # 确保知识库目录存在
    if not os.path.exists(knowledge_dir):
        st.error(f"❌ 知识库目录不存在: {knowledge_dir}")
        return None
    
    # 1. 加载 Embedding
    try:
        print("[DEBUG] 正在加载 Embedding 模型...")
        embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

    # 2. 初始化/加载 Chroma
    print("[DEBUG] 正在初始化向量数据库...")
    try:
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"❌ 向量数据库初始化失败: {e}")
        return None
    
    # 3. 加载元数据并执行增量更新
    file_metadata = load_file_metadata(persist_dir)
    updated_metadata = update_vector_database(vectordb, knowledge_dir, file_metadata)
    save_file_metadata(persist_dir, updated_metadata)
    
    print("[DEBUG] ✅ 系统初始化完成")
    return vectordb

# === 系统初始化 ===
vectordb = init_system()

def create_retrieval_chain(retriever):
    return RunnableParallel(
        source_documents=retriever,
        question=RunnablePassthrough()
    )

# === 界面逻辑 ===
if vectordb:
    # 侧边栏
    with st.sidebar:
        st.header("设置")
        k_val = st.slider("检索文档数量 (K)", 3, 10, 3)
        
        st.divider()
        st.header("📚 搜索历史")
        
        # 历史记录点击处理
        if st.session_state.search_history:
            for i, (hist_query, timestamp) in enumerate(reversed(st.session_state.search_history[-10:])):
                if st.button(f"{hist_query}", key=f"hist_{i}"):
                    st.session_state.current_query = hist_query
                    st.session_state.input_key += 1
                    st.rerun()
        else:
            st.caption("暂无历史")

        st.divider()
        if st.button("🗑️ 清空所有数据"):
            st.cache_resource.clear()
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            st.success("已重置，请刷新页面")
            st.rerun()

    # 主界面
    retriever = vectordb.as_retriever(search_kwargs={"k": k_val})
    chain = create_retrieval_chain(retriever)

    # 搜索框
    query = st.text_input(
        "请输入问题：", 
        value=st.session_state.current_query,
        key=f"search_input_{st.session_state.input_key}" 
    )

    # 执行搜索逻辑
    if query:
        # 如果是新输入的内容，更新 Session 并保存历史
        if query != st.session_state.current_query:
            st.session_state.current_query = query
        
        # 添加历史记录 (去重)
        if not st.session_state.search_history or st.session_state.search_history[-1][0] != query:
            st.session_state.search_history.append((query, datetime.now()))

        with st.spinner('🔍 正在检索...'):
            result = chain.invoke(query)
            
            # 结果展示
            if result.get('source_documents'):
                st.subheader(f"找到 {len(result['source_documents'])} 个相关片段：")
                for i, doc in enumerate(result['source_documents']):
                    src = os.path.basename(doc.metadata.get('source', '未知'))
                    with st.expander(f"参考 {i+1}: {src}", expanded=(i == 0)):  # 只展开第一个
                        st.markdown(doc.page_content)
                        st.caption(f"来源: {src}")
            else:
                st.warning("未找到相关内容。")

else:
    st.info("系统初始化中...")