# 算力芯片概念股财报问答知识库（作业 3）

面向 A 股**算力芯片概念股（12 家）**的财报知识库与问答系统：

1. 从交易所官方指定披露平台**自动爬取**样本公司近两年年报与半年报全文（48 份 PDF、9,755 页）；
2. 使用 **MinerU 4.0（本地部署）** 完成版面分析与表格结构化抽取（表格按行列还原为 HTML）；
3. 构建 **Qwen3-Embedding-0.6B 向量索引 + BM25 倒排索引**的混合检索知识库，
   每个文本块携带**公司、章节标题、页码**三类元数据；
4. 提供**问答页面**（Flask），调用 **DeepSeek API** 生成回答，自动标注引用出处（公司 / 章节 / 页码）；
5. 设计 **10 道测试题**（含 2 道跨公司全景题）并逐题记录召回块、正误与错误原因。

## 交付物对照

| 交付物 | 位置 |
|---|---|
| 完整代码仓库 | 本目录（`git log` 含开发过程；`src/` 为五步流水线） |
| 问答页面运行截图 | `docs/screenshots/`（首页 + 2 组问答示例） |
| 一页总结报告 | `docs/report/总结报告.md` |
| 逐题评测记录 | `eval/results/评估记录.md`、`eval_results.csv`、`eval_raw.md/json` |

## 目录结构

```
├── configs/companies.yml        # 样本池：12 家算力芯片概念股 + 报告期口径
├── src/
│   ├── common.py                # 公共工具（路径/配置/日志/HTTP）
│   ├── step1_download_reports.py# ① 自动检索并下载年报/半年报全文（幂等）
│   ├── step2_extract_mineru.py  # ② MinerU 结构化抽取（表格/页码，可断点续跑）
│   ├── step3_chunk.py           # ③ 切块并附加 公司/章节/页码 元数据
│   ├── step4_build_index.py     # ④ Qwen3-Embedding + BM25 双索引
│   ├── step5_qa.py              # ⑤ 混合检索 + 答案生成（引用出处）
│   ├── app.py + web/index.html  # 问答页面（支持 ?q= 自动提问、&k= 指定召回数，便于截图/集成）
├── eval/
│   ├── test_questions.yml       # 10 道测试题（标准答案已核实）
│   ├── run_eval.py              # 批量评测：召回/回答/引用
│   └── results/                 # 评测结果（评估记录.md / csv / 原始记录）
├── tools/                       # 辅助脚本（多连接下载/审计工具）
├── data/                        # 数据产物（raw_pdfs/extracted/chunks/index/models）
└── docs/                        # 截图与总结报告
```

## 环境准备（Windows）

两个独立环境（Python 3.12）：

```powershell
# 主环境：爬取、切块、索引、页面、评测
uv venv .venv-kb --python 3.12
uv pip install --python .venv-kb\Scripts\python.exe -r requirements.txt
# torch（CUDA 版，按显卡驱动选轮子）：
uv pip install --python .venv-kb\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128

# 抽取环境：MinerU 4.x（本地解析）
uv venv .venv-mineru --python 3.12
uv pip install --python .venv-mineru\Scripts\python.exe -r requirements-mineru.txt
uv pip install --python .venv-mineru\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
$env:MINERU_MODEL_SOURCE="modelscope"; .venv-mineru\Scripts\mineru-kit.exe models download --tier basic
```

模型（检索用）经 ModelScope 下载至 `data/models/`：`Qwen/Qwen3-Embedding-0.6B`。
生成由 DeepSeek API 完成，配置 Key（二选一）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."                 # 环境变量（推荐）
# 或写入 data/config/deepseek.json：{"api_key": "sk-..."}
```

## 一键复现流水线

```powershell
.venv-kb\Scripts\python.exe src\step1_download_reports.py        # 下载 48 份报告（幂等）
.venv-kb\Scripts\python.exe src\step2_extract_mineru.py           # MinerU 抽取（默认 flash 档）
.venv-kb\Scripts\python.exe src\step3_chunk.py                    # 切块 + 元数据
.venv-kb\Scripts\python.exe src\step4_build_index.py --device cuda --batch-size 16 --max-length 640
.venv-kb\Scripts\python.exe src\app.py --port 8000                # 问答页面
.venv-kb\Scripts\python.exe eval\run_eval.py                      # 10 题评测
```

## 数据源说明

公告检索与 PDF 下载均通过**巨潮资讯网**（www.cninfo.com.cn，深圳证券交易所下属深圳证券信息有限公司运营，沪深两市上市公司公告的官方指定披露平台）的公开接口完成；下载文件直接来自其官方静态资源站（static.cninfo.com.cn），全程无需人工干预。

## 报告期口径

"近两年年报与半年报" 取：**2024 年年度报告、2025 年年度报告、2025 年半年度报告、2026 年半年度报告**（截至 2026 年 9 月已披露的最新两期年报 + 两期半年报），共 12 家 × 4 期 = 48 份文档，9,755 页。

## 技术方案要点

| 环节 | 方案与关键参数 |
|---|---|
| 爬取 | cninfo 公告检索接口；标题精确匹配（含"（修订版）/全文"变体）、分页遍历、PDF 魔数校验、SHA256 |
| 抽取 | MinerU 4.0，**flash 档**（与 basic 档实测对比：表格输出逐字节一致，速度/内存更优，见总结报告）；跳过页眉/页码噪声块；表格保留 HTML 行列结构 |
| 切块 | 页内合并（目标 800 字）、跨页不合并（页码引用唯一）、表格独立成块；标题启发式 + MinerU 标注双通道（层级重映射 + 噪声过滤） |
| 索引 | Qwen3-Embedding-0.6B（文档含轻量上下文头；查询加任务指令前缀）；jieba/BM25（公司/章节字段加权）；向量 fp16 |
| 检索 | RRF 融合 + 三类加权：公司名提及（+0.30/61）、期间消歧（+0.50/61）、叙述型对比数据（+0.20/61）、归母口径定向（+0.35/61） |
| 生成 | DeepSeek API（deepseek-chat，temperature=0）；证据预算：≤16 块、合计 ≤12000 字（成本与信噪比控制）；输出 [n] 引用标记 |
| 页面 | Flask + 原生前端；答案内 [n] 引用可点击定位；展示召回证据与检索得分 |

## 评测摘要

10 题（单公司 8 + 跨公司 2）：**9 题完全正确 + 1 题部分正确**（q09 多公司对比题结论正确、覆盖不全）。
主要迭代记录：本地 1.7B 4/10 → 换 DeepSeek API 7/10 → 修复嵌入池化缺陷重建索引 9/10 → 检索终调。
逐题记录与归因见 `eval/results/评估记录.md`（含四版对比表）；遗留短板与改进清单见总结报告第四节。

## 已知边界

- `data/` 下的大体量产物（PDF、抽取结果、索引、模型）不入 git，可由上述脚本重建；
- 生成依赖外部 API（需网络与 Key），检索与嵌入（Qwen3-Embedding + BM25）完全本地运行；
- 个别 PDF 源文件存在字符错序（如数字"632,,.225584"），两个解析档位均无法修复。

## 运行环境备注

本项目在 Windows / RTX 3060 Laptop（6GB 显存）/ 23GB 内存下开发与全量运行：
- 全量抽取（48 份、9,755 页）约 48 分钟（flash 档，2 并发）；
- 全量索引构建约 29 分钟（嵌入约 18 条/秒）；
- 单题问答响应：检索 1~10 秒 + 生成 0.8~2.5 秒（DeepSeek API，另计网络往返）。
