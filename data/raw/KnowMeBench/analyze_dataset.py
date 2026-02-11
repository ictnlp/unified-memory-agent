#!/usr/bin/env python3
"""
KnowMeBench通用数据集分析脚本
支持dataset1, dataset2, dataset3的自适应分析
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import statistics
from collections import Counter
from pathlib import Path


class UniversalDatasetAnalyzer:
    """KnowMeBench通用数据集分析器"""

    # 定义三个dataset的字段映射
    FIELD_MAPPINGS = {
        'dataset1': {
            'content_fields': ['action', 'dialogue', 'environment', 'background', 'inner_thought'],
            'timestamp': 'timestamp',
            'location': 'location',
            'id': 'id'
        },
        'dataset2': {
            'content_fields': ['action', 'dialogue', 'environment', 'background', 'mind'],
            'timestamp': 'timestamp',
            'location': 'location',
            'id': 'id'
        },
        'dataset3': {
            'content_fields': ['action', 'dialogue', 'Environment', 'Background', 'Mind'],
            'timestamp': 'timestamp',
            'location': 'location',
            'id': 'id'
        }
    }

    def __init__(self, dataset_path: str, dataset_type: str = None):
        """
        初始化分析器

        Args:
            dataset_path: 数据集文件路径
            dataset_type: 数据集类型 (auto, dataset1, dataset2, dataset3)
        """
        self.dataset_path = dataset_path
        self.data = []
        self.analysis_results = {}
        self.dataset_type = dataset_type
        self.field_config = None

        # 自动检测数据集类型
        if dataset_type == 'auto' or dataset_type is None:
            self.dataset_type = self._detect_dataset_type()
        else:
            if dataset_type not in self.FIELD_MAPPINGS:
                raise ValueError(f"不支持的dataset类型: {dataset_type}. 支持的类型: {list(self.FIELD_MAPPINGS.keys())}")

        self.field_config = self.FIELD_MAPPINGS[self.dataset_type]
        print(f"检测到数据集类型: {self.dataset_type}")
        print(f"内容字段: {self.field_config['content_fields']}")

    def _detect_dataset_type(self) -> str:
        """自动检测数据集类型"""
        # 先加载数据
        print(f"加载数据集: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not self.data:
            raise ValueError("数据集为空")

        # 检查第一条记录的字段
        first_record = self.data[0]
        fields = set(first_record.keys())

        # 根据字段特征判断
        if 'inner_thought' in fields:
            return 'dataset1'
        elif 'mind' in fields and 'Mind' not in fields:
            return 'dataset2'
        elif 'Mind' in fields:
            return 'dataset3'
        else:
            # 默认返回dataset1
            print("无法自动检测数据集类型，默认使用dataset1")
            return 'dataset1'

    def load_dataset(self) -> None:
        """加载JSON数据集"""
        if not self.data:
            print(f"加载数据集: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        print(f"成功加载 {len(self.data)} 条记录")

    def analyze_basic_info(self) -> Dict[str, Any]:
        """分析基础信息"""
        if not self.data:
            raise ValueError("数据未加载")

        total_records = len(self.data)

        # 获取所有字段（包括额外字段如category）
        all_fields = set()
        for record in self.data:
            all_fields.update(record.keys())

        # 标准字段 + 内容字段
        fields = [
            self.field_config.get('id', 'id'),
            self.field_config.get('timestamp', 'timestamp'),
            self.field_config.get('location', 'location')
        ]
        fields.extend(self.field_config['content_fields'])

        # 添加额外字段（如category）
        extra_fields = list(all_fields - set(fields))
        fields.extend(extra_fields)

        # 时间范围
        timestamps = []
        for record in self.data:
            ts = record.get(self.field_config.get('timestamp', 'timestamp'))
            if ts:
                try:
                    timestamps.append(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'))
                except (ValueError, TypeError):
                    pass

        timestamps.sort()
        time_span_days = (timestamps[-1] - timestamps[0]).days if timestamps else 0

        # 地点统计
        location_field = self.field_config.get('location', 'location')
        locations = [record.get(location_field) for record in self.data if record.get(location_field)]
        unique_locations = len(set(locations))

        # 字段完整性统计
        field_completeness = {}
        for field in fields:
            non_null_count = sum(1 for record in self.data if record.get(field) is not None)
            completeness = (non_null_count / total_records) * 100
            field_completeness[field] = {
                'count': non_null_count,
                'percentage': round(completeness, 1)
            }

        basic_info = {
            'dataset_type': self.dataset_type,
            'total_records': total_records,
            'all_fields': sorted(list(all_fields)),
            'time_span_years': round(time_span_days / 365.25, 1),
            'unique_locations': unique_locations,
            'time_range': {
                'start': timestamps[0].strftime('%Y-%m-%d %H:%M:%S') if timestamps else None,
                'end': timestamps[-1].strftime('%Y-%m-%d %H:%M:%S') if timestamps else None
            },
            'field_completeness': field_completeness,
            'content_fields': self.field_config['content_fields']
        }

        return basic_info

    def analyze_time_distribution(self) -> Dict[str, Any]:
        """分析时间分布"""
        time_intervals = []
        prev_timestamp = None
        timestamp_field = self.field_config.get('timestamp', 'timestamp')

        for record in self.data:
            ts = record.get(timestamp_field)
            if ts:
                try:
                    current_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    if prev_timestamp:
                        interval_seconds = (current_time - prev_timestamp).total_seconds()
                        time_intervals.append(interval_seconds)
                    prev_timestamp = current_time
                except (ValueError, TypeError):
                    pass

        if not time_intervals:
            return {'error': '没有有效的时间戳数据'}

        time_intervals_minutes = [interval / 60 for interval in time_intervals]

        # 统计不同时间间隔的数量
        interval_counts = Counter()
        for interval in time_intervals_minutes:
            if interval <= 1:
                interval_counts['<=1分钟'] += 1
            elif interval <= 5:
                interval_counts['1-5分钟'] += 1
            elif interval <= 30:
                interval_counts['5-30分钟'] += 1
            elif interval <= 60:
                interval_counts['30-60分钟'] += 1
            elif interval <= 1440:  # 24小时
                interval_counts['1-24小时'] += 1
            elif interval <= 10080:  # 7天
                interval_counts['1-7天'] += 1
            else:
                interval_counts['>7天'] += 1

        time_stats = {
            'total_intervals': len(time_intervals),
            'avg_interval_minutes': round(statistics.mean(time_intervals_minutes), 2),
            'median_interval_minutes': round(statistics.median(time_intervals_minutes), 2),
            'min_interval_minutes': round(min(time_intervals_minutes), 2),
            'max_interval_minutes': round(max(time_intervals_minutes), 2),
            'interval_distribution': dict(interval_counts)
        }

        return time_stats

    def analyze_content_length(self) -> Dict[str, Any]:
        """分析各字段内容长度"""
        field_content_lengths = {}

        for field in self.field_config['content_fields']:
            lengths = []
            for record in self.data:
                content = record.get(field)
                if content and isinstance(content, str):
                    lengths.append(len(content))

            if lengths:
                field_content_lengths[field] = {
                    'avg_length': round(statistics.mean(lengths), 2),
                    'median_length': round(statistics.median(lengths), 2),
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'non_empty_records': len(lengths)
                }
            else:
                field_content_lengths[field] = {
                    'error': '无有效数据'
                }

        return field_content_lengths

    def analyze_location_distribution(self) -> Dict[str, Any]:
        """分析地点分布"""
        location_field = self.field_config.get('location', 'location')

        location_counts = Counter()
        location_with_time = {}  # 记录地点首次出现的时间

        for record in self.data:
            location = record.get(location_field)
            if location:
                location_counts[location] += 1
                if location not in location_with_time and record.get(self.field_config.get('timestamp', 'timestamp')):
                    location_with_time[location] = record.get(self.field_config.get('timestamp', 'timestamp'))

        top_locations = location_counts.most_common(20)

        location_stats = {
            'unique_locations': len(location_counts),
            'location_frequency': dict(location_counts),
            'top_20_locations': top_locations,
            'location_first_appearance': location_with_time
        }

        return location_stats

    def analyze_category_distribution(self) -> Dict[str, Any]:
        """分析category字段分布（如果有）"""
        category_field = 'category'

        # 检查是否存在category字段
        has_category = any('category' in record for record in self.data)

        if not has_category:
            return {'note': '该数据集没有category字段'}

        # 收集所有category
        categories = []
        category_sets = []
        for record in self.data:
            cat = record.get(category_field)
            if cat:
                categories.append(cat)
                # 尝试解析为列表（如 "background, mind"）
                if isinstance(cat, str):
                    cat_list = [c.strip() for c in cat.split(',')]
                    category_sets.append(set(cat_list))

        if not categories:
            return {'note': 'category字段没有有效数据'}

        # 统计category频率
        category_counts = Counter(categories)

        # 统计category组合
        unique_combinations = []
        if category_sets:
            seen_combinations = set()
            for cat_set in category_sets:
                combo = ', '.join(sorted(cat_set))
                if combo not in seen_combinations:
                    seen_combinations.add(combo)
                    unique_combinations.append(combo)

        category_stats = {
            'total_with_category': len(categories),
            'unique_categories': len(category_counts),
            'category_frequency': dict(category_counts),
            'unique_combinations': unique_combinations,
            'sample_categories': categories[:10]
        }

        return category_stats

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """分析时间模式"""
        timestamp_field = self.field_config.get('timestamp', 'timestamp')

        # 按小时、天、月、年统计
        hours = []
        days = []
        months = []
        years = []

        for record in self.data:
            ts = record.get(timestamp_field)
            if ts:
                try:
                    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    hours.append(dt.hour)
                    days.append(dt.weekday())  # 0=周一, 6=周日
                    months.append(dt.month)
                    years.append(dt.year)
                except (ValueError, TypeError):
                    pass

        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

        temporal_stats = {
            'hour_distribution': dict(Counter(hours)),
            'weekday_distribution': {weekday_names[d]: count for d, count in Counter(days).items()},
            'month_distribution': dict(Counter(months)),
            'year_distribution': dict(Counter(years)),
            'total_temporal_records': len(hours)
        }

        return temporal_stats

    def analyze_content_cooccurrence(self) -> Dict[str, Any]:
        """分析内容字段共现情况"""
        cooccurrence = {}
        content_fields = self.field_config['content_fields']

        for record in self.data:
            active_fields = []
            for field in content_fields:
                content = record.get(field)
                if content and isinstance(content, str) and content.strip():
                    active_fields.append(field)

            if len(active_fields) > 1:
                combo = tuple(sorted(active_fields))
                cooccurrence[combo] = cooccurrence.get(combo, 0) + 1

        # 排序
        sorted_cooccurrence = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_cooccurrences': sum(cooccurrence.values()),
            'top_combinations': sorted_cooccurrence[:20],
            'all_combinations': [f"{' + '.join(combo)}: {count}" for combo, count in sorted_cooccurrence]
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        print("开始分析数据集...")

        # 加载数据
        self.load_dataset()

        # 基础信息分析
        print("分析基础信息...")
        basic_info = self.analyze_basic_info()

        # 时间分布分析
        print("分析时间分布...")
        time_distribution = self.analyze_time_distribution()

        # 内容长度分析
        print("分析内容长度...")
        content_length = self.analyze_content_length()

        # 地点分布分析
        print("分析地点分布...")
        location_distribution = self.analyze_location_distribution()

        # Category分析（如果有）
        print("分析category分布...")
        category_distribution = self.analyze_category_distribution()

        # 时间模式分析
        print("分析时间模式...")
        temporal_patterns = self.analyze_temporal_patterns()

        # 内容字段共现分析
        print("分析内容字段共现...")
        content_cooccurrence = self.analyze_content_cooccurrence()

        # 汇总结果
        analysis_results = {
            'dataset_type': self.dataset_type,
            'basic_info': basic_info,
            'time_distribution': time_distribution,
            'content_length': content_length,
            'location_distribution': location_distribution,
            'category_distribution': category_distribution,
            'temporal_patterns': temporal_patterns,
            'content_cooccurrence': content_cooccurrence
        }

        self.analysis_results = analysis_results
        return analysis_results

    def print_summary(self) -> None:
        """打印分析摘要"""
        if not self.analysis_results:
            print("请先运行分析")
            return

        results = self.analysis_results

        print("\n" + "="*70)
        print(f"KnowMeBench数据集分析摘要 - {results['dataset_type'].upper()}")
        print("="*70)

        # 基础信息
        basic = results['basic_info']
        print(f"\n📊 基础信息:")
        print(f"  数据集类型: {basic['dataset_type']}")
        print(f"  总记录数: {basic['total_records']:,}")
        print(f"  时间跨度: {basic['time_span_years']} 年")
        print(f"  独特地点数: {basic['unique_locations']:,}")
        print(f"  所有字段: {', '.join(basic['all_fields'])}")

        print(f"\n📈 字段完整性:")
        for field, stats in sorted(basic['field_completeness'].items()):
            count = stats['count']
            percentage = stats['percentage']
            bar = '█' * int(percentage / 5)  # 每5%一个条
            print(f"  {field:20s}: {count:6,} ({percentage:5.1f}%) {bar}")

        # 时间分布
        time_dist = results['time_distribution']
        if 'error' not in time_dist:
            print(f"\n⏱️  时间分布:")
            print(f"  平均间隔: {time_dist['avg_interval_minutes']} 分钟")
            print(f"  中位数间隔: {time_dist['median_interval_minutes']} 分钟")
            print(f"  最小间隔: {time_dist['min_interval_minutes']} 分钟")
            print(f"  最大间隔: {time_dist['max_interval_minutes']} 分钟")

            print(f"\n  间隔分布:")
            for interval, count in time_dist['interval_distribution'].items():
                percentage = (count / time_dist['total_intervals']) * 100
                bar = '█' * int(percentage / 5)
                print(f"    {interval:15s}: {count:5,} ({percentage:5.1f}%) {bar}")

        # 时间模式
        temporal = results['temporal_patterns']
        if temporal['total_temporal_records'] > 0:
            print(f"\n📅 时间模式:")

            print(f"  星期分布:")
            for day, count in temporal['weekday_distribution'].items():
                percentage = (count / temporal['total_temporal_records']) * 100
                print(f"    {day}: {count} ({percentage:.1f}%)")

            print(f"  年份分布:")
            for year, count in sorted(temporal['year_distribution'].items()):
                print(f"    {year}: {count} 条记录")

        # Category分布
        category = results['category_distribution']
        if 'note' not in category:
            print(f"\n🏷️  Category分析:")
            print(f"  有category的记录: {category['total_with_category']:,}")
            print(f"  独特category数: {category['unique_categories']}")
            if category['unique_combinations']:
                print(f"  常见category组合:")
                for combo in category['unique_combinations'][:5]:
                    print(f"    {combo}")

        # 内容长度
        content_len = results['content_length']
        print(f"\n📝 内容长度统计:")
        for field, stats in content_len.items():
            if 'error' not in stats:
                print(f"  {field}:")
                print(f"    非空记录: {stats['non_empty_records']:,}")
                print(f"    平均长度: {stats['avg_length']:.0f} 字符")
                print(f"    中位数长度: {stats['median_length']:.0f} 字符")
                print(f"    长度范围: {stats['min_length']} - {stats['max_length']}")

        # 内容字段共现
        cooccurrence = results['content_cooccurrence']
        if cooccurrence['total_cooccurrences'] > 0:
            print(f"\n🔗 内容字段共现:")
            print(f"  总共现次数: {cooccurrence['total_cooccurrences']:,}")
            print(f"  常见组合:")
            for combo, count in cooccurrence['top_combinations'][:10]:
                print(f"    {' + '.join(combo)}: {count}")

        # 地点分布
        location_dist = results['location_distribution']
        print(f"\n📍 地点分布:")
        print(f"  独特地点: {location_dist['unique_locations']:,}")
        print(f"  最常见地点 (Top 10):")
        for location, count in location_dist['top_20_locations'][:10]:
            print(f"    {location}: {count}")

        print("\n" + "="*70)

    def save_analysis(self, output_path: str = None) -> None:
        """保存分析结果到文件"""
        if not self.analysis_results:
            raise ValueError("没有分析结果可保存")

        if output_path is None:
            output_path = f"dataset_analysis_{self.dataset_type}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"分析结果已保存到: {output_path}")


def process_dataset(input_file: str, dataset_type: str = 'auto',
                   output_dir: str = None) -> Dict[str, Any]:
    """
    处理单个数据集

    Args:
        input_file: 输入文件路径
        dataset_type: 数据集类型 (auto, dataset1, dataset2, dataset3)
        output_dir: 输出目录
    """
    # 创建分析器
    analyzer = UniversalDatasetAnalyzer(input_file, dataset_type)

    try:
        # 运行分析
        results = analyzer.run_full_analysis()

        # 打印摘要
        analyzer.print_summary()

        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"dataset_analysis_{analyzer.dataset_type}.json"
            analyzer.save_analysis(str(output_file))
        else:
            analyzer.save_analysis()

        return results

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        raise


def compare_datasets(input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    比较所有三个数据集

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    base_path = Path(input_dir)

    dataset_configs = {
        'dataset1': base_path / 'dataset1/input/dataset1.json',
        'dataset2': base_path / 'dataset2/input/dataset2.json',
        'dataset3': base_path / 'dataset3/input/dataset3.json'
    }

    comparison_results = {}

    for dataset_name, input_file in dataset_configs.items():
        if not input_file.exists():
            print(f"\n⚠️  跳过 {dataset_name}: 文件不存在")
            continue

        print(f"\n{'='*70}")
        print(f"处理 {dataset_name}")
        print(f"{'='*70}")

        try:
            results = process_dataset(str(input_file), dataset_name, output_dir)
            comparison_results[dataset_name] = results
        except Exception as e:
            print(f"❌ {dataset_name} 处理失败: {e}")
            continue

    # 生成对比报告
    print("\n" + "="*70)
    print("数据集对比报告")
    print("="*70)

    if comparison_results:
        metric_table = []
        for dataset_name in dataset_configs.keys():
            if dataset_name in comparison_results:
                results = comparison_results[dataset_name]
                basic = results['basic_info']
                content_len = results['content_length']

                # 计算总平均内容长度
                total_avg_length = sum(
                    stats.get('avg_length', 0)
                    for stats in content_len.values()
                    if 'error' not in stats
                )

                metric_table.append({
                    'Dataset': dataset_name,
                    'Records': basic['total_records'],
                    'Time Span (years)': basic['time_span_years'],
                    'Locations': basic['unique_locations'],
                    'Avg Content Length (chars)': round(total_avg_length, 0)
                })

        # 打印对比表格
        if metric_table:
            headers = list(metric_table[0].keys())
            col_widths = [max(len(str(row[h])) for row in metric_table) for h in headers]

            # 打印表头
            header_line = '  '.join(h.ljust(w) for h, w in zip(headers, col_widths))
            print(header_line)
            print('  '.join('-' * w for w in col_widths))

            # 打印各行
            for row in metric_table:
                row_line = '  '.join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
                print(row_line)

    print("="*70)

    return comparison_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowMeBench通用数据分析工具')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['dataset1', 'dataset2', 'dataset3', 'all'],
                       help='要分析的数据集 (默认: all)')
    parser.add_argument('--input-dir', type=str,
                       default='./KnowmeBench',
                       help='输入目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='./analysis_output',
                       help='输出目录路径')

    args = parser.parse_args()

    base_input_path = Path(args.input_dir)

    print("="*70)
    print("KnowMeBench 通用数据分析工具")
    print("="*70)

    if args.dataset == 'all':
        # 比较所有数据集
        print("处理所有数据集并生成对比报告...")
        comparison_results = compare_datasets(args.input_dir, args.output_dir)

        # 保存对比结果
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            comparison_file = output_path / "dataset_comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n对比结果已保存到: {comparison_file}")
    else:
        # 处理单个数据集
        input_file = base_input_path / args.dataset / 'input' / f'{args.dataset}.json'

        if not input_file.exists():
            print(f"❌ 文件不存在: {input_file}")
            return

        process_dataset(str(input_file), 'auto', args.output_dir)

    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()
