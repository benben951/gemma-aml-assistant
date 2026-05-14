# Gemma AML Compliance Assistant

> **Safety & Trust Hackathon Project**
> 
> 鍒╃敤 Gemma 4 鐨勭绾块儴缃茶兘鍔涳紝鏋勫缓閲戣瀺鍚堣鐭ヨ瘑鍔╂墜

## Portfolio Snapshot

This repository is positioned as an AML / due diligence RAG evaluation project for regulated financial workflows. It demonstrates local LLM deployment, retrieval grounding, citation-aware responses, and an evaluation layer for hallucination and risk-coverage review.

- Portfolio angle: LLM application engineering for AML, KYC, and due diligence workflows.
- Evaluation focus: grounding, citation accuracy, risk-point coverage, uncertainty handling, and analyst actionability.
- More details: [Evaluation Notes](docs/EVALUATION.md)

## 馃幆 椤圭洰瀹氫綅

### Safety & Trust 璧涢亾鏍稿績浠峰€?
| Safety闇€姹?| 鎴戜滑鐨勬柟妗?|
|------------|------------|
| **鍙В閲?* | Thinking妯″紡灞曠ず鎺ㄧ悊杩囩▼ |
| **閫忔槑** | 姣忎釜鍥炵瓟鏍囨敞鏉ユ簮寮曠敤 |
| **Trust** | 鍙俊搴﹁瘎鍒嗗彲瑙嗗寲 |

### 涓轰粈涔堥€夋嫨绂荤嚎閮ㄧ讲锛?
- 鉁?**鏁版嵁闅愮淇濇姢** - 鏁忔劅AML鏁版嵁涓嶇寮€鏈湴鏈嶅姟鍣?- 鉁?**绗﹀悎閲戣瀺鍚堣瑕佹眰** - 閲戣瀺琛屼笟鏁版嵁瀹夊叏娉曡
- 鉁?**鑷富鍙帶** - 涓嶄緷璧栦簯绔疉PI锛岀ǔ瀹氬彲闈?
---

## 馃殌 蹇€熷紑濮?
### Docker 涓€閿儴缃诧紙鎺ㄨ崘锛?
```bash
# 鍚姩 Qdrant + Streamlit
docker compose up -d

# 璁块棶 http://localhost:8501
```

### Ollama 鏈湴閮ㄧ讲

```bash
# 瀹夎Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 涓嬭浇Gemma 4
ollama pull gemma4:26b-a4b
```

```python
from src.logic.gemma_client import GemmaClient

client = GemmaClient(backend="ollama")
response = client.generate("浠€涔堟槸AML灏借亴璋冩煡锛?, enable_thinking=True)

print(response.thinking)  # 鎺ㄧ悊杩囩▼
print(response.content)   # 鏈€缁堝洖绛?```

### Kaggle Notebook 杩愯

```python
from src.logic.gemma_client import GemmaClient

client = GemmaClient(backend="kaggle")
response = client.generate("浠€涔堟槸AML灏借亴璋冩煡锛?, enable_thinking=True)
```

---

## 馃搧 椤圭洰缁撴瀯

```
gemma-aml-assistant/
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ data/
鈹?  鈹?  鈹溾攢鈹€ config.py            # 閰嶇疆绠＄悊
鈹?  鈹?  鈹溾攢鈹€ models.py            # 缁熶竴鏁版嵁妯″瀷 (Document, SearchResult, GemmaResponse)
鈹?  鈹?  鈹斺攢鈹€ vector_store.py      # 鍚戦噺鏁版嵁搴撳皝瑁?(Qdrant)
鈹?  鈹溾攢鈹€ logic/
鈹?  鈹?  鈹溾攢鈹€ gemma_client.py      # Gemma 4 鎺ㄧ悊瀹㈡埛绔?(Kaggle/Ollama/HuggingFace)
鈹?  鈹?  鈹溾攢鈹€ rag_pipeline.py      # RAG 娴佹按绾?(Embedding + Qdrant + Retrieve)
鈹?  鈹?  鈹溾攢鈹€ qa_service.py        # QA 缂栨帓鏈嶅姟 (Retrieve 鈫?Generate 鈫?Explain)
鈹?  鈹?  鈹斺攢鈹€ explainability.py    # 鍙В閲婃€у紩鎿?(缃俊搴?+ 寮曠敤 + 鎺ㄧ悊閾?
鈹?  鈹斺攢鈹€ ui/
鈹?      鈹斺攢鈹€ response_formatter.py # 鍝嶅簲鏍煎紡鍖?鈹?鈹溾攢鈹€ app/
鈹?  鈹斺攢鈹€ streamlit_app.py         # Streamlit 鍓嶇
鈹?鈹溾攢鈹€ data/eval/
鈹?  鈹斺攢鈹€ aml_eval.jsonl           # AML/DD 璇勬祴鏁版嵁闆?(25棰?
鈹?鈹溾攢鈹€ tests/
鈹?  鈹溾攢鈹€ test_gemma.py            # GemmaClient 鍗曞厓娴嬭瘯
鈹?  鈹溾攢鈹€ test_basic.py            # RAG 娴佹按绾垮熀鏈祴璇?鈹?  鈹斺攢鈹€ test_quick.py            # 鏁版嵁缁撴瀯蹇€熸祴璇?鈹?鈹溾攢鈹€ .github/workflows/ci.yml    # GitHub Actions CI
鈹溾攢鈹€ docker-compose.yml           # Docker Compose 涓€閿儴缃?鈹溾攢鈹€ Dockerfile.streamlit         # Streamlit 瀹瑰櫒
鈹溾攢鈹€ pyproject.toml               # 椤圭洰閰嶇疆 + mypy/ruff/pytest
鈹溾攢鈹€ pyrightconfig.json           # 绫诲瀷妫€鏌ラ厤缃?鈹溾攢鈹€ requirements.txt             # Python 渚濊禆
鈹溾攢鈹€ LICENSE                      # MIT License
鈹斺攢鈹€ README.md
```

---

## 馃敡 鎶€鏈爤

| 缁勪欢 | 鎶€鏈?| 澶囨敞 |
|------|------|------|
| **LLM** | Gemma 4 26B A4B | 娲昏穬鍙傛暟浠?.8B锛?56K涓婁笅鏂?|
| **鎺ㄧ悊妗嗘灦** | transformers + bitsandbytes | 4-bit閲忓寲 |
| **澶囬€夋鏋?* | Ollama | 鏈湴閮ㄧ讲 |
| **鍚戦噺鏁版嵁搴?* | Qdrant | 楂樻晥妫€绱?|
| **Embedding** | sentence-transformers | all-MiniLM-L6-v2 |
| **鍓嶇** | Streamlit | 蹇€熷師鍨?|

---

## 馃И 娴嬭瘯涓庤瘎娴?
```bash
# 杩愯鍗曞厓娴嬭瘯
pip install -e ".[dev]"
pytest tests/ -v

# Lint
ruff check src app

# 绫诲瀷妫€鏌?pyright src app

# 璇勬祴 RAG 鏁堟灉锛堥渶瑕?Qdrant + Ollama锛?python -m data.eval.aml_eval
```

---

## 馃弳 Hackathon 鎻愪氦

### 璇勫鏍囧噯鏄犲皠

| 璇勫缁村害 | 鎴戜滑鐨勮璁?|
|----------|------------|
| **Impact & Vision (40鍒?** | 閲戣瀺鍚堣鍦烘櫙瀹為檯浠峰€硷紝绂荤嚎閮ㄧ讲淇濇姢鏁忔劅鏁版嵁 |
| **Video Pitch (30鍒?** | 3鍒嗛挓瑙嗛灞曠ず锛氶棶棰樷啋鏂规鈫扗emo鈫掍环鍊?|
| **Technical Execution (30鍒?** | RAG鏋舵瀯銆丟emma 4闆嗘垚銆乀hinking妯″紡 |

### 鎻愪氦鍐呭

- 鉁?Kaggle Notebook婕旂ず
- 鉁?GitHub浠ｇ爜浠撳簱
- 鉁?1500瀛梂riteup
- 鉁?3鍒嗛挓YouTube瑙嗛

---

## 馃摐 License

MIT License

---

## 馃檹 Acknowledgments

- Google DeepMind for Gemma 4
- Kaggle for free GPU resources
- LangChain for RAG framework
- Luohe High School Yuhan Zhang
