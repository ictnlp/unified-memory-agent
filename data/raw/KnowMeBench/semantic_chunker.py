#!/usr/bin/env python3
"""
KnowMeBench通用语义分块系统
支持dataset1, dataset2, dataset3的自适应处理
"""

import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import statistics
import argparse
from pathlib import Path


class TokenEstimator:
    """Token估算器"""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的token数量
        考虑多种字符类型：
        - 英文字符：4字符/token
        - 中文字符：2字符/token
        - 数字符号：单独计算
        - 保守估计：+10%缓冲
        """
        if not text:
            return 0

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text.strip())

        # 计算不同类型字符
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        numbers = len(re.findall(r'\d', text))
        symbols = len(re.findall(r'[^a-zA-Z0-9\s]', text))

        # 计算基本token数
        tokens = (english_chars / 4.0 +
                 chinese_chars / 2.0 +
                 numbers / 3.0 +
                 symbols / 2.0 +
                 text.count(' ') / 1.5)

        # 添加10%缓冲
        tokens = int(tokens * 1.1)

        return max(1, tokens)


class UniversalSemanticChunker:
    """通用语义分块器 - 支持所有三个dataset"""

    # 定义三个dataset的字段映射
    FIELD_MAPPINGS = {
        'dataset1': {
            'content_fields': ['action', 'dialogue', 'environment', 'background', 'inner_thought'],
            'timestamp': 'timestamp',
            'location': 'location'
        },
        'dataset2': {
            'content_fields': ['action', 'dialogue', 'environment', 'background', 'mind'],
            'timestamp': 'timestamp',
            'location': 'location'
        },
        'dataset3': {
            'content_fields': ['action', 'dialogue', 'Environment', 'Background', 'Mind'],
            'timestamp': 'timestamp',
            'location': 'location'
        }
    }

    def __init__(self,
                 min_tokens: int = 3000,
                 max_tokens: int = 6000,
                 overlap_tokens: int = 200,
                 boundary_threshold: float = 0.5,
                 dataset_type: str = 'dataset1'):
        """
        初始化通用语义分块器

        Args:
            min_tokens: 最小分块大小（tokens）
            max_tokens: 最大分块大小（tokens）
            overlap_tokens: 重叠保护大小（tokens）
            boundary_threshold: 语义边界强度阈值
            dataset_type: 数据集类型 ('dataset1', 'dataset2', 'dataset3')
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.boundary_threshold = boundary_threshold
        self.token_estimator = TokenEstimator()

        # 设置数据集类型
        if dataset_type not in self.FIELD_MAPPINGS:
            raise ValueError(f"不支持的dataset类型: {dataset_type}. 支持的类型: {list(self.FIELD_MAPPINGS.keys())}")

        self.dataset_type = dataset_type
        self.field_config = self.FIELD_MAPPINGS[dataset_type]

        print(f"初始化分块器 - 数据集类型: {dataset_type}")
        print(f"内容字段: {self.field_config['content_fields']}")

    def combine_record_to_text(self, record: Dict[str, Any]) -> str:
        """
        将单条记录组合成文本（自适应不同dataset的字段）

        Args:
            record: 数据记录

        Returns:
            组合后的文本
        """
        parts = []

        # 添加时间和地点
        timestamp_field = self.field_config['timestamp']
        location_field = self.field_config['location']

        timestamp = record.get(timestamp_field, '')
        location = record.get(location_field, '')
        if timestamp and location:
            parts.append(f"[{timestamp}] {location}")

        # 按配置的内容字段优先级添加内容
        content_parts = []
        for field in self.field_config['content_fields']:
            content = record.get(field)
            if content and isinstance(content, str) and content.strip():
                content_parts.append(content)

        if content_parts:
            parts.append(' '.join(content_parts))

        return ' | '.join(parts) if parts else ''

    def calculate_time_gap_strength(self, time1: str, time2: str) -> float:
        """
        计算时间跳跃强度

        Args:
            time1: 第一个时间戳
            time2: 第二个时间戳

        Returns:
            强度值 (0-1)
        """
        try:
            dt1 = datetime.strptime(time1, '%Y-%m-%d %H:%M:%S')
            dt2 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')
            gap = abs((dt2 - dt1).total_seconds())
        except (ValueError, TypeError):
            return 0.0

        # 根据时间跳跃大小计算强度
        if gap > 7 * 24 * 3600:  # 超过一周
            return 0.6
        elif gap > 24 * 3600:  # 超过一天
            return 0.4
        elif gap > 6 * 3600:  # 超过6小时
            return 0.2
        else:
            return 0.1

    def calculate_location_change_strength(self, loc1: str, loc2: str) -> float:
        """
        计算地点变化强度

        Args:
            loc1: 第一个地点
            loc2: 第二个地点

        Returns:
            强度值 (0-1)
        """
        if not loc1 or not loc2:
            return 0.0

        if loc1 != loc2:
            return 0.3

        return 0.0

    def calculate_content_density_change(self,
                                       prev_content_length: int,
                                       curr_content_length: int) -> float:
        """
        计算内容密度变化强度

        Args:
            prev_content_length: 前一条记录内容长度
            curr_content_length: 当前记录内容长度

        Returns:
            强度值 (0-1)
        """
        if prev_content_length == 0:
            return 0.0

        density_ratio = abs(curr_content_length - prev_content_length) / prev_content_length

        if density_ratio > 0.5:
            return 0.2
        elif density_ratio > 0.3:
            return 0.15
        else:
            return 0.05

    def calculate_content_anomaly_strength(self, content_length: int,
                                         avg_content_length: float) -> float:
        """
        计算内容长度异常强度

        Args:
            content_length: 当前内容长度
            avg_content_length: 平均内容长度

        Returns:
            强度值 (0-1)
        """
        if avg_content_length == 0:
            return 0.0

        ratio = content_length / avg_content_length

        if ratio > 3.0 or ratio < 0.3:
            return 0.15
        else:
            return 0.0

    def detect_semantic_boundary(self,
                               prev_record: Dict[str, Any],
                               curr_record: Dict[str, Any],
                               avg_content_length: float) -> Tuple[bool, float]:
        """
        检测语义边界

        Args:
            prev_record: 前一条记录
            curr_record: 当前记录
            avg_content_length: 平均内容长度

        Returns:
            (是否语义边界, 边界强度)
        """
        # 计算各种边界指标
        time_strength = 0.0
        location_strength = 0.0
        density_strength = 0.0
        anomaly_strength = 0.0

        timestamp_field = self.field_config['timestamp']
        location_field = self.field_config['location']

        # 时间跳跃检测
        if prev_record.get(timestamp_field) and curr_record.get(timestamp_field):
            time_strength = self.calculate_time_gap_strength(
                prev_record[timestamp_field],
                curr_record[timestamp_field]
            )

        # 地点变化检测
        location_strength = self.calculate_location_change_strength(
            prev_record.get(location_field, ''),
            curr_record.get(location_field, '')
        )

        # 内容密度变化检测
        prev_content = self._get_record_content_length(prev_record)
        curr_content = self._get_record_content_length(curr_record)
        density_strength = self.calculate_content_density_change(
            prev_content, curr_content
        )

        # 内容长度异常检测
        anomaly_strength = self.calculate_content_anomaly_strength(
            curr_content, avg_content_length
        )

        # 综合边界强度
        boundary_strength = (time_strength + location_strength +
                           density_strength + anomaly_strength)

        # 判断是否为语义边界
        is_semantic_boundary = boundary_strength >= self.boundary_threshold

        return is_semantic_boundary, boundary_strength

    def _get_record_content_length(self, record: Dict[str, Any]) -> int:
        """获取记录内容长度（自适应字段）"""
        total_length = 0

        for field in self.field_config['content_fields']:
            content = record.get(field)
            if content and isinstance(content, str):
                total_length += len(content)

        return total_length

    def create_chunks(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        创建语义分块

        Args:
            data: 数据记录列表

        Returns:
            分块列表
        """
        print("开始语义分块...")

        # 计算平均内容长度
        content_lengths = [self._get_record_content_length(record) for record in data]
        avg_content_length = statistics.mean(content_lengths) if content_lengths else 0

        chunks = []
        current_chunk = {
            'chunk_id': 0,
            'text': '',
            'start_id': 0,
            'end_id': 0,
            'record_count': 0,
            'token_count': 0,
            'start_time': None,
            'end_time': None,
            'locations': []
        }

        timestamp_field = self.field_config['timestamp']
        location_field = self.field_config['location']

        for i, record in enumerate(data):
            # 组合当前记录为文本
            record_text = self.combine_record_to_text(record)

            if not record_text:
                continue

            # 检查是否需要分割
            if current_chunk['text']:
                # 计算新chunk大小
                new_token_count = self.token_estimator.estimate_tokens(
                    current_chunk['text'] + '\n' + record_text
                )

                # 如果超过最大限制，需要分割
                if new_token_count > self.max_tokens:
                    # 保存当前chunk
                    current_chunk['end_id'] = i - 1
                    current_chunk['token_count'] = self.token_estimator.estimate_tokens(
                        current_chunk['text']
                    )
                    chunks.append(current_chunk.copy())

                    # 创建新chunk
                    current_chunk = {
                        'chunk_id': len(chunks),
                        'text': '',
                        'start_id': i,
                        'end_id': 0,
                        'record_count': 0,
                        'token_count': 0,
                        'start_time': record.get(timestamp_field),
                        'end_time': None,
                        'locations': []
                    }

                # 检查语义边界（仅当当前chunk超过最小大小时）
                elif new_token_count >= self.min_tokens:
                    is_boundary, strength = self.detect_semantic_boundary(
                        data[i-1], record, avg_content_length
                    )

                    if is_boundary:
                        # 保存当前chunk
                        current_chunk['end_id'] = i - 1
                        current_chunk['token_count'] = self.token_estimator.estimate_tokens(
                            current_chunk['text']
                        )
                        chunks.append(current_chunk.copy())

                        # 创建新chunk
                        current_chunk = {
                            'chunk_id': len(chunks),
                            'text': '',
                            'start_id': i,
                            'end_id': 0,
                            'record_count': 0,
                            'token_count': 0,
                            'start_time': record.get(timestamp_field),
                            'end_time': None,
                            'locations': []
                        }

            # 添加记录到当前chunk
            if current_chunk['text']:
                current_chunk['text'] += '\n' + record_text
            else:
                current_chunk['text'] = record_text
                current_chunk['start_time'] = record.get(timestamp_field)

            current_chunk['end_time'] = record.get(timestamp_field)
            current_chunk['record_count'] += 1

            # 记录地点
            location = record.get(location_field)
            if location and location not in current_chunk['locations']:
                current_chunk['locations'].append(location)

        # 添加最后一个chunk
        if current_chunk['text']:
            current_chunk['end_id'] = len(data) - 1
            current_chunk['token_count'] = self.token_estimator.estimate_tokens(
                current_chunk['text']
            )
            chunks.append(current_chunk)

        print(f"完成语义分块，共生成 {len(chunks)} 个chunk")

        return chunks

    def save_chunks(self,
                   chunks: List[Dict[str, Any]],
                   output_dir: str,
                   dataset_name: str) -> None:
        """
        保存分块结果

        Args:
            chunks: 分块列表
            output_dir: 输出目录
            dataset_name: 数据集名称
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 文件路径
        output_json = output_path / f"{dataset_name}_chunks.json"
        output_text = output_path / f"{dataset_name}_chunks_text.txt"

        # 保存JSON格式（包含元数据）
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"JSON格式分块已保存到: {output_json}")

        # 保存纯文本格式（便于阅读）
        with open(output_text, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"\n{'='*80}\n")
                f.write(f"Chunk {i+1}/{len(chunks)}\n")
                f.write(f"{'='*80}\n")
                f.write(f"ID: {chunk['chunk_id']}\n")
                f.write(f"记录范围: {chunk['start_id']} - {chunk['end_id']}\n")
                f.write(f"记录数: {chunk['record_count']}\n")
                f.write(f"Token数: {chunk['token_count']}\n")
                f.write(f"时间范围: {chunk['start_time']} - {chunk['end_time']}\n")
                f.write(f"地点数: {len(chunk['locations'])}\n")
                f.write(f"\n{chunk['text']}\n")

        print(f"纯文本格式分块已保存到: {output_text}")

    def print_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> None:
        """
        打印分块统计信息

        Args:
            chunks: 分块列表
        """
        if not chunks:
            print("没有分块数据")
            return

        token_counts = [chunk['token_count'] for chunk in chunks]

        # 统计不同大小的chunk数量
        size_distribution = {
            '<3k': 0,
            '3k-4k': 0,
            '4k-5k': 0,
            '5k-6k': 0,
            '>6k': 0
        }

        for token_count in token_counts:
            if token_count < 3000:
                size_distribution['<3k'] += 1
            elif token_count < 4000:
                size_distribution['3k-4k'] += 1
            elif token_count < 5000:
                size_distribution['4k-5k'] += 1
            elif token_count < 6000:
                size_distribution['5k-6k'] += 1
            else:
                size_distribution['>6k'] += 1

        print("\n" + "="*60)
        print(f"语义分块统计结果 - {self.dataset_type}")
        print("="*60)

        print(f"\n📊 分块概览:")
        print(f"  总分块数: {len(chunks)}")
        print(f"  平均大小: {round(statistics.mean(token_counts))} tokens")
        print(f"  中位数大小: {round(statistics.median(token_counts))} tokens")
        print(f"  最小大小: {min(token_counts)} tokens")
        print(f"  最大大小: {max(token_counts)} tokens")

        print(f"\n📈 大小分布:")
        for size_range, count in size_distribution.items():
            percentage = (count / len(chunks)) * 100
            print(f"  {size_range}: {count} 个 ({percentage:.1f}%)")

        print(f"\n📍 地点覆盖:")
        all_locations = set()
        for chunk in chunks:
            all_locations.update(chunk['locations'])
        print(f"  总涉及地点: {len(all_locations)} 个")

        print("\n" + "="*60)


def process_dataset(input_file: str, output_dir: str, dataset_type: str,
                   min_tokens: int = 3000, max_tokens: int = 6000):
    """
    处理单个数据集

    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        dataset_type: 数据集类型
        min_tokens: 最小token数
        max_tokens: 最大token数
    """
    # 创建分块器
    chunker = UniversalSemanticChunker(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=200,
        boundary_threshold=0.5,
        dataset_type=dataset_type
    )

    try:
        # 加载数据
        print(f"\n加载数据集: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"成功加载 {len(data)} 条记录")

        # 执行分块
        chunks = chunker.create_chunks(data)

        # 打印统计信息
        chunker.print_chunk_statistics(chunks)

        # 保存结果
        chunker.save_chunks(chunks, output_dir, dataset_type)

        print(f"\n✅ {dataset_type} 语义分块完成!")

        return chunks

    except Exception as e:
        print(f"❌ 处理 {dataset_type} 时出现错误: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowMeBench通用语义分块工具')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['dataset1', 'dataset2', 'dataset3', 'all'],
                       help='要处理的数据集 (默认: all)')
    parser.add_argument('--input-dir', type=str,
                       default='./KnowmeBench',
                       help='输入目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='./chunked_output',
                       help='输出目录路径')
    parser.add_argument('--min-tokens', type=int, default=3000,
                       help='最小chunk大小（tokens）')
    parser.add_argument('--max-tokens', type=int, default=6000,
                       help='最大chunk大小（tokens）')

    args = parser.parse_args()

    base_input_path = Path(args.input_dir)

    # 定义数据集配置
    dataset_configs = {
        'dataset1': base_input_path / 'dataset1/input/dataset1.json',
        'dataset2': base_input_path / 'dataset2/input/dataset2.json',
        'dataset3': base_input_path / 'dataset3/input/dataset3.json'
    }

    # 确定要处理的数据集
    if args.dataset == 'all':
        datasets_to_process = list(dataset_configs.keys())
    else:
        datasets_to_process = [args.dataset]

    print("="*60)
    print("KnowMeBench 通用语义分块工具")
    print("="*60)
    print(f"处理数据集: {', '.join(datasets_to_process)}")
    print(f"Token范围: {args.min_tokens} - {args.max_tokens}")
    print("="*60)

    # 处理每个数据集
    results = {}
    for dataset_name in datasets_to_process:
        input_file = dataset_configs[dataset_name]

        if not input_file.exists():
            print(f"\n⚠️  跳过 {dataset_name}: 文件不存在 - {input_file}")
            continue

        try:
            chunks = process_dataset(
                str(input_file),
                args.output_dir,
                dataset_name,
                args.min_tokens,
                args.max_tokens
            )
            results[dataset_name] = chunks
        except Exception as e:
            print(f"\n❌ {dataset_name} 处理失败: {e}")
            continue

    # 打印总结
    print("\n" + "="*60)
    print("处理总结")
    print("="*60)
    for dataset_name, chunks in results.items():
        print(f"{dataset_name}: {len(chunks)} 个chunks")
    print("="*60)
    print("\n✅ 全部处理完成!")


if __name__ == "__main__":
    main()
