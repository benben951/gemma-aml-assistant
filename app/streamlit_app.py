"""
AML Compliance Assistant - Streamlit 前端

Safety & Trust Hackathon 项目

功能：
- 文档上传与管理
- 合规问答（RAG + Gemma 4）
- Thinking 模式展示（Safety 核心）
- 来源引用显示（Trust 核心）
- 可信度评分（Trust 核心）
"""

import streamlit as st
import os
import time
from typing import Dict, Any, List, Optional

# 页面配置
st.set_page_config(
    page_title="AML Compliance Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 标题
st.title("🔍 AML Compliance Assistant")
st.markdown("""
> **Safety & Trust Hackathon** - 利用 Gemma 4 的离线部署能力构建金融合规助手

**核心价值**：
- 🔒 离线部署，数据隐私保护
- 💡 Thinking 模式，推理过程透明
- 📚 来源引用，每个回答可追溯
- ✅ 可信度评分，回答质量可视化
""")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")
    
    # Backend 选择
    backend = st.selectbox(
        "推理框架",
        ["ollama", "kaggle", "huggingface"],
        index=0,
        help="Ollama: 本地部署\nKaggle: Notebook演示\nHuggingFace: 云端API"
    )
    
    # Thinking 模式
    enable_thinking = st.checkbox(
        "启用 Thinking 模式",
        value=True,
        help="Safety 核心功能：展示推理过程"
    )
    
    # 检索数量
    top_k = st.slider(
        "检索文档数量",
        min_value=1,
        max_value=10,
        value=5,
        help="RAG 检索的相关文档数量"
    )
    
    # 存储模式
    storage_type = st.selectbox(
        "向量存储",
        ["memory", "qdrant_local", "qdrant_cloud"],
        index=0,
        help="Memory: 测试用\nLocal: 本地持久化\nCloud: 云端存储"
    )
    
    st.markdown("---")
    st.markdown("### 📊 统计")
    
    # 显示统计信息（如果有）
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    
    st.metric("查询次数", st.session_state.query_count)
    st.metric("文档数量", st.session_state.doc_count)


# 初始化客户端（延迟加载）
@st.cache_resource
def init_gemma_client(backend: str):
    """初始化 Gemma 客户端"""
    try:
        from src.core.gemma_client import GemmaClient
        
        client = GemmaClient(backend=backend)
        return client
    except Exception as e:
        st.error(f"初始化 Gemma 客户端失败: {e}")
        return None


@st.cache_resource
def init_retriever(storage_type: str):
    """初始化检索器"""
    try:
        from src.rag.retriever import AMLRetriever
        
        retriever = AMLRetriever(storage_type=storage_type)
        return retriever
    except Exception as e:
        st.error(f"初始化检索器失败: {e}")
        return None


@st.cache_resource
def init_explainability(gemma_client):
    """初始化可解释性引擎"""
    try:
        from src.safety.explainability import ExplainabilityEngine
        
        engine = ExplainabilityEngine(gemma_client)
        return engine
    except Exception as e:
        st.error(f"初始化可解释性引擎失败: {e}")
        return None


# 主界面 Tabs
tab1, tab2, tab3 = st.tabs(["💬 合规问答", "📄 文档管理", "⚙️ 系统信息"])

# =====================
# Tab 1: 合规问答
# =====================
with tab1:
    st.header("合规问答")
    
    # 问题输入
    question = st.text_area(
        "输入您的合规问题",
        placeholder="例如：什么是AML尽职调查？高风险客户的判断标准是什么？",
        height=100,
    )
    
    # 提交按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_btn = st.button("🔍 查询", type="primary")
    with col2:
        st.markdown("*提示：启用 Thinking 模式可查看推理过程*")
    
    # 处理查询
    if submit_btn and question:
        # 初始化客户端
        client = init_gemma_client(backend)
        retriever = init_retriever(storage_type)
        
        if client and retriever:
            # 显示加载状态
            with st.spinner("正在检索和分析..."):
                # RAG 流程
                from src.rag.retriever import RAGPipeline
                
                pipeline = RAGPipeline(
                    gemma_client=client,
                    retriever=retriever,
                    top_k=top_k,
                    enable_thinking=enable_thinking,
                )
                
                try:
                    result = pipeline.query(question)
                    
                    # 更新统计
                    st.session_state.query_count += 1
                    
                    # 显示结果
                    st.markdown("---")
                    
                    # Thinking 过程（如果启用）
                    if enable_thinking and result.get("thinking"):
                        with st.expander("🧠 Thinking 推理过程", expanded=True):
                            st.markdown(result["thinking"])
                    
                    # 最终回答
                    st.subheader("💡 回答")
                    st.markdown(result["answer"])
                    
                    # 来源引用（Safety & Trust 核心）
                    if result.get("sources"):
                        st.subheader("📚 来源引用")
                        
                        for i, source in enumerate(result["sources"][:3]):
                            with st.container():
                                st.markdown(f"**[{i+1}] {source.get('source', '未知来源')}**")
                                excerpt = source.get('excerpt', source.get('content', ''))
                                if excerpt:
                                    st.markdown(f"> {excerpt[:200]}...")
                                if source.get('score'):
                                    st.caption(f"相关度: {source['score']:.2f}")
                    
                    # 可信度评分（Trust 核心）
                    if result.get("confidence"):
                        confidence = result["confidence"]
                        
                        st.subheader("✅ 可信度评分")
                        
                        # 进度条显示
                        progress_color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.5 else "red"
                        st.progress(confidence)
                        
                        # 评分说明
                        if confidence >= 0.7:
                            st.success(f"可信度: {confidence:.1%} - 该回答有明确依据，可信度较高")
                        elif confidence >= 0.5:
                            st.warning(f"可信度: {confidence:.1%} - 该回答有一定依据，建议核实具体条款")
                        else:
                            st.error(f"可信度: {confidence:.1%} - 该回答依据较少，建议查阅原文或咨询专业人士")
                    
                except Exception as e:
                    st.error(f"查询失败: {e}")
                    st.info("请检查：1) Gemma 客户端是否正确初始化 2) 文档是否已上传")

# =====================
# Tab 2: 文档管理
# =====================
with tab2:
    st.header("文档管理")
    
    st.markdown("""
    上传 AML 相关法规文档，系统将自动解析并构建向量索引。
    
    **支持格式**：PDF、Word (.docx)、文本 (.txt)
    """)
    
    # 文件上传（限制大小和数量）
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_FILES = 10
    
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        # 检查数量
        if len(uploaded_files) > MAX_FILES:
            st.error(f"最多上传 {MAX_FILES} 个文件")
            uploaded_files = uploaded_files[:MAX_FILES]
        
        # 检查大小
        valid_files = []
        for file in uploaded_files:
            if file.size > MAX_FILE_SIZE:
                st.warning(f"{file.name} 超过 10MB 限制，已跳过")
            else:
                valid_files.append(file)
        
        uploaded_files = valid_files
        
        if uploaded_files:
            st.write(f"已上传 {len(uploaded_files)} 个文件")
            
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        # 处理按钮
        process_btn = st.button("📄 处理文档", type="primary")
        
        if process_btn:
            retriever = init_retriever(storage_type)
            
            if retriever:
                with st.spinner("正在处理文档..."):
                    try:
                        from src.rag.retriever import Document
                        
                        documents = []
                        
                        for file in uploaded_files:
                            # 读取内容
                            content = file.read()
                            
                            # 根据文件类型处理
                            if file.name.endswith(".txt"):
                                text = content.decode("utf-8")
                            elif file.name.endswith(".pdf"):
                                # PDF 解析（需要 pymupdf）
                                try:
                                    import fitz  # pymupdf
                                    doc = fitz.open(stream=content, filetype="pdf")
                                    text = ""
                                    for page in doc:
                                        text += page.get_text()
                                except ImportError:
                                    st.warning("PDF 解析需要安装 pymupdf")
                                    text = f"[PDF文件: {file.name}]"
                            elif file.name.endswith(".docx"):
                                # Word 解析
                                try:
                                    from docx import Document as DocxDocument
                                    doc = DocxDocument(file)
                                    text = "\n".join([para.text for para in doc.paragraphs])
                                except ImportError:
                                    st.warning("Word 解析需要安装 python-docx")
                                    text = f"[Word文件: {file.name}]"
                            else:
                                text = f"[未知文件类型: {file.name}]"
                            
                            # 分块（简单分块）
                            chunk_size = 500
                            for i in range(0, len(text), chunk_size):
                                chunk = text[i:i + chunk_size]
                                if chunk.strip():
                                    documents.append(
                                        Document(
                                            content=chunk,
                                            source=file.name,
                                            metadata={"chunk_index": i // chunk_size},
                                        )
                                    )
                        
                        # 添加到检索器
                        count = retriever.add_documents(documents)
                        
                        # 更新统计
                        st.session_state.doc_count += len(uploaded_files)
                        
                        st.success(f"成功处理 {len(uploaded_files)} 个文件，生成 {count} 个文档块")
                        
                    except Exception as e:
                        st.error(f"文档处理失败: {e}")
    
    # 显示已有文档
    st.markdown("---")
    st.subheader("当前文档库")
    
    retriever = init_retriever(storage_type)
    if retriever:
        try:
            info = retriever.get_collection_info()
            st.json(info)
        except Exception as e:
            st.info("文档库暂无内容")
    
    # 清空按钮
    if st.button("🗑️ 清空文档库"):
        retriever = init_retriever(storage_type)
        if retriever:
            retriever.delete_collection()
            st.session_state.doc_count = 0
            st.success("文档库已清空")
            st.rerun()

# =====================
# Tab 3: 系统信息
# =====================
with tab3:
    st.header("系统信息")
    
    # Gemma 客户端信息
    st.subheader("Gemma 4 客户端")
    
    client = init_gemma_client(backend)
    if client:
        st.json({
            "backend": client.backend.value,
            "model": client.model,
            "load_in_4bit": client.load_in_4bit,
        })
    else:
        st.warning("客户端未初始化")
    
    # 检索器信息
    st.subheader("向量检索器")
    
    retriever = init_retriever(storage_type)
    if retriever:
        try:
            info = retriever.get_collection_info()
            st.json(info)
        except Exception:
            st.json({
                "storage_type": retriever.storage_type.value,
                "collection_name": retriever.collection_name,
                "embedding_dimension": retriever.embedding.dimension,
            })
    else:
        st.warning("检索器未初始化")
    
    # 使用说明
    st.markdown("---")
    st.subheader("📖 使用说明")
    
    st.markdown("""
    ### 快速开始
    
    1. **配置 Backend**: 在侧边栏选择推理框架（推荐 Ollama 本地部署）
    2. **上传文档**: 在"文档管理"Tab上传 AML 法规文档
    3. **提出问题**: 在"合规问答"Tab输入问题，查看回答和来源
    
    ### Safety & Trust 核心功能
    
    | 功能 | 说明 |
    |------|------|
    | **Thinking 模式** | 展示推理过程，让用户理解分析逻辑 |
    | **来源引用** | 每个回答标注出处，可追溯原文 |
    | **可信度评分** | 可视化回答质量，帮助用户判断可靠性 |
    
    ### 技术架构
    
    ```
    用户问题 → [向量检索] → 相关文档
                         ↓
                    [Gemma 4 生成] → 回答 + Thinking
                         ↓
                    [来源引用] → 可解释结果
    ```
    
    ### 依赖安装
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ### Ollama 本地部署
    
    ```bash
    # 安装 Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # 下载 Gemma 4
    ollama pull gemma4:26b-a4b
    ```
    """)


# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>
        AML Compliance Assistant | Safety & Trust Hackathon | 
        Powered by Gemma 4 | 
        GitHub: <a href="#">gemma-aml-assistant</a>
    </small>
</div>
""", unsafe_allow_html=True)