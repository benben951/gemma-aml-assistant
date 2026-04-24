# Gemma AML Compliance Assistant

> **Safety & Trust Hackathon Project**
> 
> 利用 Gemma 4 的离线部署能力，构建金融合规知识助手

## 🎯 项目定位

### Safety & Trust 赛道核心价值

| Safety需求 | 我们的方案 |
|------------|------------|
| **可解释** | Thinking模式展示推理过程 |
| **透明** | 每个回答标注来源引用 |
| **Trust** | 可信度评分可视化 |

### 为什么选择离线部署？

- ✅ **数据隐私保护** - 敏感AML数据不离开本地服务器
- ✅ **符合金融合规要求** - 金融行业数据安全法规
- ✅ **自主可控** - 不依赖云端API，稳定可靠

---

## 🚀 快速开始

### Docker 一键部署（推荐）

```bash
# 启动 Qdrant + Streamlit
docker compose up -d

# 访问 http://localhost:8501
```

### Ollama 本地部署

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载Gemma 4
ollama pull gemma4:26b-a4b
```

```python
from src.logic.gemma_client import GemmaClient

client = GemmaClient(backend="ollama")
response = client.generate("什么是AML尽职调查？", enable_thinking=True)

print(response.thinking)  # 推理过程
print(response.content)   # 最终回答
```

### Kaggle Notebook 运行

```python
from src.logic.gemma_client import GemmaClient

client = GemmaClient(backend="kaggle")
response = client.generate("什么是AML尽职调查？", enable_thinking=True)
```

---

## 📁 项目结构

```
gemma-aml-assistant/
├── src/
│   ├── data/
│   │   ├── config.py            # 配置管理
│   │   ├── models.py            # 统一数据模型 (Document, SearchResult, GemmaResponse)
│   │   └── vector_store.py      # 向量数据库封装 (Qdrant)
│   ├── logic/
│   │   ├── gemma_client.py      # Gemma 4 推理客户端 (Kaggle/Ollama/HuggingFace)
│   │   ├── rag_pipeline.py      # RAG 流水线 (Embedding + Qdrant + Retrieve)
│   │   ├── qa_service.py        # QA 编排服务 (Retrieve → Generate → Explain)
│   │   └── explainability.py    # 可解释性引擎 (置信度 + 引用 + 推理链)
│   └── ui/
│       └── response_formatter.py # 响应格式化
│
├── app/
│   └── streamlit_app.py         # Streamlit 前端
│
├── data/eval/
│   └── aml_eval.jsonl           # AML/DD 评测数据集 (25题)
│
├── tests/
│   ├── test_gemma.py            # GemmaClient 单元测试
│   ├── test_basic.py            # RAG 流水线基本测试
│   └── test_quick.py            # 数据结构快速测试
│
├── .github/workflows/ci.yml    # GitHub Actions CI
├── docker-compose.yml           # Docker Compose 一键部署
├── Dockerfile.streamlit         # Streamlit 容器
├── pyproject.toml               # 项目配置 + mypy/ruff/pytest
├── pyrightconfig.json           # 类型检查配置
├── requirements.txt             # Python 依赖
├── LICENSE                      # MIT License
└── README.md
```

---

## 🔧 技术栈

| 组件 | 技术 | 备注 |
|------|------|------|
| **LLM** | Gemma 4 26B A4B | 活跃参数仅3.8B，256K上下文 |
| **推理框架** | transformers + bitsandbytes | 4-bit量化 |
| **备选框架** | Ollama | 本地部署 |
| **向量数据库** | Qdrant | 高效检索 |
| **Embedding** | sentence-transformers | all-MiniLM-L6-v2 |
| **前端** | Streamlit | 快速原型 |

---

## 🧪 测试与评测

```bash
# 运行单元测试
pip install -e ".[dev]"
pytest tests/ -v

# Lint
ruff check src app

# 类型检查
pyright src app

# 评测 RAG 效果（需要 Qdrant + Ollama）
python -m data.eval.aml_eval
```

---

## 🏆 Hackathon 提交

### 评审标准映射

| 评审维度 | 我们的设计 |
|----------|------------|
| **Impact & Vision (40分)** | 金融合规场景实际价值，离线部署保护敏感数据 |
| **Video Pitch (30分)** | 3分钟视频展示：问题→方案→Demo→价值 |
| **Technical Execution (30分)** | RAG架构、Gemma 4集成、Thinking模式 |

### 提交内容

- ✅ Kaggle Notebook演示
- ✅ GitHub代码仓库
- ✅ 1500字Writeup
- ✅ 3分钟YouTube视频

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- Google DeepMind for Gemma 4
- Kaggle for free GPU resources
- LangChain for RAG framework