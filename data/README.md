# 辅助入口

`data/` 只保留两个轻量辅助脚本，核心实现均在 `factor_analysis/`。

- `data.py`：查看原始 Parquet 文件的形状、字段和前几行。
- `data_process.py`：兼容旧命令，实际调用 `python -m factor_analysis`。

推荐使用项目主入口：

```bash
.venv/bin/python -m factor_analysis
```
