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

### Kaggle Notebook 运行

```python
import kagglehub
from src.core.gemma_client import GemmaClient

# 初始化客户端（自动下载模型）
client = GemmaClient(backend="kaggle")

# 启用Thinking模式进行合规分析
response = client.generate(
    "什么是AML尽职调查？",
    enable_thinking=True
)

print(response.thinking)  # 推理过程
print(response.content)   # 最终回答
```

### Ollama 本地部署

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载Gemma 4
ollama pull gemma4:26b-a4b

# 运行
ollama run gemma4:26b-a4b
```

```python
from src.core.gemma_client import GemmaClient

client = GemmaClient(backend="ollama")
response = client.generate("什么是AML尽职调查？")
```

---

## 📁 项目结构

```
gemma-aml-assistant/
├── src/
│   ├── core/
│   │   └── gemma_client.py      # Gemma 4 推理客户端
│   ├── rag/
│   │   ├── retriever.py         # 检索模块
│   │   └── pipeline.py          # RAG完整流程
│   ├── safety/
│   │   └── explainability.py    # 可解释性引擎
│   └── api/
│       └── routes.py            # API路由
│
├── app/
│   └── streamlit_app.py         # 前端界面
│
├── docs/
│   └── aml_regulations/         # AML法规文档
│
├── notebooks/
│   └── demo.ipynb               # Kaggle演示
│
└── tests/
    └── test_gemma.py            # 集成测试
```

---

## 🔧 技术栈

| 组件 | 技术 | 备注 |
|------|------|------|
| **LLM** | Gemma 4 26B A4B | 活跃参数仅3.8B，256K上下文 |
| **推理框架** | transformers + bitsandbytes | 4-bit量化 |
| **备选框架** | Ollama | 本地部署 |
| **向量数据库** | Qdrant | 高效检索 |
| **前端** | Streamlit | 快速原型 |

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

CC-BY 4.0 (Hackathon要求)

---

## 🙏 Acknowledgments

- Google DeepMind for Gemma 4
- Kaggle for free GPU resources
- LangChain for RAG framework