# Distributed Creative Search on Digital Platforms

## Which script does what?

| File | Responsibility | Local outputs |
| --- | --- | --- |
| `run.py` | Single command-line entry point; runs each stage in order | Stage-specific output folders |
| `common.py` | Fixed creator list, input checks, half-year indexing, random-seed and serialization helpers | None directly |
| `study1.py` | Individual change configurations, changes in the dominant dimension, search gateways, continuation follow-up and creator fixed-effect associations | Association estimates, configuration and gateway tables, descriptive summaries |
| `study2.py` | Common-period positions, original and finite-video-corrected dispersion decomposition, nearest-peer retention, three-period comparisons, coverage and leave-one-creator-out checks | Transition-level tables for three representation specifications |
| `diagnostics.py` | Movement-assignment comparisons, stationary pooled-video simulation and disjoint-baseline diagnostics | Diagnostic tables and simulation settings |
| `validation.py` | Two independent coders and consensus ratings; weighted kappa, exact agreement, Spearman correspondence, cluster bootstrap and leave-one-creator-out sensitivity | Human-validation metrics |
| `tests/test_analysis.py` | Small synthetic checks of mathematical identities and edge cases | Test status only |

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Run the analyses

Prepare the private files described in [INPUTS.md](INPUTS.md), outside this repository. Input and output directories must be separate.

```bash
python run.py study1 --inputs /path/to/private_inputs --outputs /path/to/local_results
python run.py study2 --inputs /path/to/private_inputs --outputs /path/to/local_results
python run.py diagnostics --inputs /path/to/private_inputs --outputs /path/to/local_results
python run.py validation --inputs /path/to/private_inputs --outputs /path/to/local_results
# Or run all four stages:
python run.py all --inputs /path/to/private_inputs --outputs /path/to/local_results
```

For Windows, use quoted paths such as `--inputs "D:/research/private_inputs"`. The diagnostics and 10,000-replicate validation bootstrap take longer than the descriptive analyses. The programs never upload inputs or results.

## Analysis conventions

- The focal panel is fixed. The scripts reject other creator identities in focal event and inventory inputs.
- Study 1 retains the historical peer-reference distances supplied with the frozen events. That reference population is broader than the ten focal creators; reducing it to ten would change the estimand.
- Existing individual-period evidence labels retain their original multiple-testing context. The code does not recompute those labels using a smaller testing family.
- Study 2 distinguishes the original three-block representation, a two-block representation with exactly matched videos and membership, and a two-block representation using its fuller available coverage. It reports both original and corrected quantities, including sign changes.
- Associations are exploratory and conditional on the focal panel. Cluster-based intervals and nominal p-values do not provide independent confirmation after outcome-guided case selection.
- Human-model correspondence is a rank correlation, not a classification accuracy. Exact agreement and weighted kappa refer to the two independent human coders. Consensus scores are used for human-model correspondence.

The code covers the analysis stages listed above. It does not recreate the raw-data collection, model inference, individual-period simulation classifications or publication-layout figures. Those upstream measurements are explicitly documented inputs, not silently reconstructed alternatives.

## Sharing and rights

This repository contains source code and documentation only. Research inputs and generated outputs are excluded by `.gitignore` and are not distributed here. No license for third-party videos, model representations or ratings is implied. No open-source license has yet been selected by the authors.

## 中文快速说明

先安装依赖，再运行 `run.py`；无需逐个修改脚本中的路径。`study1.py` 负责个体变化与延续，`study2.py` 负责群体差异的变化，`diagnostics.py` 负责 Study 2 的诊断实验，`validation.py` 负责人工编码验证。每个程序的输入字段在 `INPUTS.md` 中说明。

本仓库仅发布代码，不含任何真实研究数据或结果文件，也不含前期寻找、排序或筛选最终样本的程序。运行需要研究者自行提供冻结后的分析输入。

