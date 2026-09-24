# 算力芯片概念股财报问答知识库（作业 3）

面向 A 股**算力芯片概念股（12 家）**的财报知识库与问答系统：

1. 从交易所官方指定披露平台**自动爬取**样本公司近两年年报与半年报全文（48 份 PDF）；
2. 使用 **MinerU** 完成版面分析与表格结构化抽取（表格按行列还原为 HTML/Markdown）；
3. 构建 **Qwen3-Embedding-0.6B 向量索引 + BM25 倒排索引**的混合检索知识库，
   每个文本块携带**公司、章节标题、页码**三类元数据；
4. 提供**问答页面**，答案自动标注引用出处（公司 / 章节 / 页码）；
5. 设计 **10 道测试题**（含 2 道以上跨公司全景题）并逐题记录召回块、正误与错误原因。

## 目录结构

```
├── configs/
│   └── companies.yml            # 样本池：12 家算力芯片概念股 + 报告期口径
├── src/
│   ├── common.py                # 公共工具（路径/配置/日志/HTTP）
│   ├── step1_download_reports.py# ① 自动检索并下载年报/半年报全文
│   ├── step2_extract_mineru.py  # ② MinerU 结构化抽取（含表格/页码）
│   ├── step3_chunk.py           # ③ 切块并附加 公司/章节/页码 元数据
│   ├── step4_build_index.py     # ④ Qwen3-Embedding + BM25 双索引
│   ├── step5_qa.py              # ⑤ 混合检索 + 答案生成（引用出处）
│   └── app.py                   # 问答页面（Flask）
├── eval/
│   ├── test_questions.yml       # 10 道测试题（≥2 道跨公司）
│   ├── run_eval.py              # 批量评测：召回/正误/错误原因
│   └── results/                 # 评测记录输出
├── data/                        # 数据产物（raw_pdfs/extracted/chunks/index）
├── docs/
│   ├── screenshots/             # 问答页面运行截图
│   └── report/                  # 一页总结报告
└── README.md
```

## 快速开始

```powershell
# 1. 数据环境（Python 3.12）
uv venv .venv-kb --python 3.12
uv pip install --python .venv-kb\Scripts\python.exe -r requirements.txt

# 2. 一键复现流水线
.venv-kb\Scripts\python.exe src\step1_download_reports.py   # 下载 48 份报告
.venv-mineru\Scripts\python.exe -m mineru_kb_extract        # MinerU 抽取（另见 step2）
.venv-kb\Scripts\python.exe src\step3_chunk.py              # 切块 + 元数据
.venv-kb\Scripts\python.exe src\step4_build_index.py        # 混合索引
.venv-kb\Scripts\python.exe src\app.py                      # 启动问答页面

# 3. 评测
.venv-kb\Scripts\python.exe eval\run_eval.py
```

## 数据源说明

公告检索与 PDF 下载均通过**巨潮资讯网**（www.cninfo.com.cn，深圳证券交易所下属
深圳证券信息有限公司运营，沪深两市上市公司公告的官方指定披露平台）的公开接口完成；
下载文件直接来自其官方静态资源站（static.cninfo.com.cn），全程无需人工干预。

## 报告期口径

"近两年年报与半年报" 取：**2024 年年度报告、2025 年年度报告、2025 年半年度报告、
2026 年半年度报告**（截至 2026 年 9 月已披露的最新两期年报 + 两期半年报），
共 12 家 × 4 期 = 48 份文档。

## 技术方案

| 环节 | 方案 |
|---|---|
| 爬取 | cninfo 公告检索接口，标题精确匹配 + 分页遍历 + PDF 魔数校验 + SHA256 |
| 抽取 | MinerU 4.x（本地，PyTorch/CUDA 后端），版面分析 + 表格识别（行列结构）+ 页码定位 |
| 切块 | 按页遍历内容块；标题启发式识别章节层级；表格独立成块（保留 HTML 行列结构） |
| 检索 | Qwen3-Embedding-0.6B 向量（余弦）+ jieba/BM25 倒排，加权融合排序 |
| 问答 | 本地小参数 LLM 基于召回证据生成答案，输出 [n] 引用标记 → 公司/章节/页码 |

## 说明与边界

- 抽取质量受版面复杂度影响：合并表头、跨页表格等场景可能存在结构损失，
  评测环节对此进行专门分析（见 `eval/` 与总结报告）。
- 本仓库不含大体量数据产物（`data/` 下 PDF 与抽取结果可通过脚本重建）。
