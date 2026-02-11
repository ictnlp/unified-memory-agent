# KnowMeBench 通用语义分块工具使用说明

## 📖 概述

`semantic_chunker_universal.py` 是一个通用的语义分块工具，支持 KnowMeBench 的所有三个数据集（dataset1, dataset2, dataset3）。

## 🆚 三个Dataset的差异

| 特性 | Dataset1 | Dataset2 | Dataset3 |
|------|----------|----------|----------|
| **内容字段** | `inner_thought` | `mind` | `Mind` (大写) |
| **环境字段** | `environment` | `environment` | `Environment` (大写) |
| **背景字段** | `background` | `background` | `Background` (大写) |
| **记录数量** | 6,644 | 11,995 | 8,423 |
| **额外字段** | - | `category` | `category` |

## 🚀 使用方法

### 基本用法

```bash
# 处理单个数据集
python3 semantic_chunker_universal.py --dataset dataset1

# 处理所有数据集
python3 semantic_chunker_universal.py --dataset all
```

### 完整参数

```bash
python3 semantic_chunker_universal.py \
  --dataset dataset1 \              # 数据集选择: dataset1, dataset2, dataset3, all
  --input-dir ./KnowmeBench \       # 输入目录路径
  --output-dir ./chunked_output \   # 输出目录路径
  --min-tokens 3000 \               # 最小chunk大小（tokens）
  --max-tokens 6000                 # 最大chunk大小（tokens）
```

## 📊 处理结果

### Dataset1
- **记录数**: 6,644
- **生成chunks**: 240
- **平均大小**: 3,624 tokens
- **地点覆盖**: 1,496 个

### Dataset2
- **记录数**: 11,995
- **生成chunks**: 256
- **平均大小**: 3,620 tokens
- **地点覆盖**: 627 个

### Dataset3
- **记录数**: 8,423
- **生成chunks**: 219
- **平均大小**: 3,678 tokens
- **地点覆盖**: 402 个

## 📁 输出文件

每个数据集会生成两个文件：

```
chunked_output/
├── dataset1_chunks.json          # JSON格式（包含元数据）
├── dataset1_chunks_text.txt      # 纯文本格式（便于阅读）
├── dataset2_chunks.json
├── dataset2_chunks_text.txt
├── dataset3_chunks.json
└── dataset3_chunks_text.txt
```

### JSON格式示例

```json
[
  {
    "chunk_id": 0,
    "text": "[时间戳] 地点 | 内容...",
    "start_id": 0,
    "end_id": 25,
    "record_count": 26,
    "token_count": 3456,
    "start_time": "1969-08-15 14:00:00",
    "end_time": "1969-08-16 10:30:00",
    "locations": ["地点1", "地点2", ...]
  },
  ...
]
```

## 🔍 核心特性

### 1. 自适应字段映射
- 自动识别不同数据集的字段名差异
- Dataset3 的大写字段（`Environment`, `Background`, `Mind`）自动适配

### 2. 智能语义分块
- 基于时间跳跃、地点变化、内容密度等多维度检测语义边界
- 动态调整分块大小（3k-6k tokens）

### 3. 内容优先级
所有数据集按相同优先级组合内容：
1. `action` - 行动
2. `dialogue` - 对话
3. `environment/Environment` - 环境
4. `background/Background` - 背景
5. `inner_thought/mind/Mind` - 内心想法

## 🛠️ 技术细节

### Token估算算法
```python
tokens = (
    英文字符 / 4.0 +
    中文字符 / 2.0 +
    数字 / 3.0 +
    符号 / 2.0 +
    空格 / 1.5
) × 1.1  # +10%缓冲
```

### 语义边界检测
```
边界强度 = 时间跳跃强度 + 地点变化强度 + 内容密度变化 + 内容长度异常

阈值 = 0.5 (可调整)
```

## 💡 使用建议

1. **首次使用**：先处理单个数据集验证效果
   ```bash
   python3 semantic_chunker_universal.py --dataset dataset1
   ```

2. **批量处理**：验证无误后批量处理
   ```bash
   python3 semantic_chunker_universal.py --dataset all
   ```

3. **调整参数**：根据实际需求调整token范围
   ```bash
   python3 semantic_chunker_universal.py --dataset all --min-tokens 2000 --max-tokens 5000
   ```

## 📝 与原脚本的区别

| 特性 | 原脚本 (semantic_chunker.py) | 新脚本 (semantic_chunker_universal.py) |
|------|------------------------------|----------------------------------------|
| **支持数据集** | 仅 dataset1 | dataset1, dataset2, dataset3 |
| **字段适配** | 硬编码 | 自动适配 |
| **命令行参数** | 无 | 完整CLI支持 |
| **批量处理** | 不支持 | 支持 `--dataset all` |
| **输出命名** | 固定 | 按数据集名称区分 |

## ⚙️ 高级配置

### 字段映射配置
如果需要添加新的数据集，只需在 `FIELD_MAPPINGS` 中添加配置：

```python
FIELD_MAPPINGS = {
    'dataset4': {
        'content_fields': ['action', 'dialogue', 'thought'],
        'timestamp': 'time',
        'location': 'place'
    }
}
```

### 边界检测参数
```python
chunker = UniversalSemanticChunker(
    min_tokens=3000,          # 最小chunk大小
    max_tokens=6000,          # 最大chunk大小
    overlap_tokens=200,       # 重叠保护
    boundary_threshold=0.5,   # 语义边界阈值
    dataset_type='dataset1'
)
```

## 🐛 常见问题

### Q: 为什么有些chunk小于3000 tokens？
A: 最后一个chunk可能不满足最小大小要求，或者是由于语义边界强制分割。

### Q: 如何调整chunk大小分布？
A: 修改 `--min-tokens` 和 `--max-tokens` 参数，或调整 `boundary_threshold`。

### Q: Dataset3的大写字段是否正确处理？
A: 是的，脚本已自动适配 `Environment`, `Background`, `Mind` 这些大写字段。

## 📞 技术支持

如有问题，请检查：
1. 输入文件路径是否正确
2. 数据格式是否符合预期
3. Python版本 >= 3.7

---

**作者**: Claude
**更新时间**: 2026-02-09
**版本**: 2.0 - Universal Edition
