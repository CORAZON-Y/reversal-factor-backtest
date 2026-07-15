# A股因子检验与二次规划组合优化

项目包含两部分：

1. 单因子构造、IC检验和分层回测。
2. 基于因子预测收益，并在风险和行业约束下进行组合优化回测。

## 目录结构

```text
.
├── basic_data/              原始Parquet数据
├── data/                    数据查看和旧入口兼容脚本
├── docs/                    数据、方法和输出说明
├── factor_analysis/         单因子回测代码
├── 二次规划组合优化/        收益预测和组合优化代码
├── tests/                   回归测试
├── .cache/                  内部中间数据，不提交Git
└── output/                  最终PNG图片
```

## 环境配置

当前推荐使用Python 3.13和`uv`：

```bash
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

验证环境：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 运行顺序

当前默认策略参数：

```text
第二题因子       factor_rank_zscore
MAD去极值倍数    3
单股权重上限     3%
组合日方差上限   0.0004
```

### 1. 单因子回测

```bash
.venv/bin/python -m factor_analysis --industry-neutral
```

程序会把内部处理表写到：

```text
.cache/单因子回测/processed_data.parquet
```

最终图片写到：

```text
output/单因子回测/
├── IC检验/
├── 直接分层回测/
└── 行业中性分层回测/
```

### 2. 生成预期收益

```bash
.venv/bin/python 二次规划组合优化/expected_return.py
```

内部结果写到：

```text
.cache/二次规划组合优化/expected_returns.parquet
```

### 3. 组合优化短试跑

```bash
.venv/bin/python 二次规划组合优化/portfolio_optimization.py \
  --start-date 2010-04-01 \
  --max-dates 5 \
  --output-dir output/二次规划组合优化/试跑
```

### 4. 全量组合优化

```bash
.venv/bin/python 二次规划组合优化/portfolio_optimization.py
```

最终只生成：

```text
output/二次规划组合优化/portfolio_cumulative_return.png
```

组合汇总指标和每日优化状态计数会直接打印在终端中。

## 输出原则

- `output/`：只保存PNG图片，不生成CSV、Parquet或Markdown。
- `.cache/`：保存跨步骤需要的Parquet中间数据。
- 运行结果可以由源码和原始数据重新生成。

更详细的方法和字段说明见[`docs/`](docs/README.md)。
