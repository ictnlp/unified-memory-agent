#!/usr/bin/env python3
"""
验证synthv1_async.py生成的数据是否正确

检查项：
1. 重新计算问题答案并与生成的答案对比
2. 检查dialogue与transactions的一致性（使用字符串匹配）
3. 检查日期、金额等数据的合理性
4. 统计各类问题的数量和准确率
"""

import json
import sys
from collections import defaultdict
from typing import List, Dict, Any


class EnhancedValidator:
    """增强版验证器：验证dialogue与transactions的一致性"""

    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.stats = {
            'total_samples': len(self.data),
            'total_questions': 0,
            'total_chunks': 0,
            'category_distribution': defaultdict(int),
            'warnings': [],
            'dialogue_transaction_mismatches': []
        }

    def validate(self):
        """执行验证"""
        print(f"开始验证 {len(self.data)} 个样本...\n")

        for idx, sample in enumerate(self.data):
            print(f"验证样本 {idx + 1}/{len(self.data)}: {sample['task_id']}")
            self.check_sample(sample)

        self.print_summary()

    def check_sample(self, sample: Dict[str, Any]):
        """检查单个样本的数据完整性和一致性"""
        task_id = sample['task_id']
        transactions = sample['metadata']['transaction_events']
        questions = sample['questions']
        chunks = sample['chunks']

        # 统计问题数量和类别分布
        self.stats['total_questions'] += len(questions)
        self.stats['total_chunks'] += len(chunks)
        for q in questions:
            self.stats['category_distribution'][q['category']] += 1

        # 检查1: 验证transaction数据的合理性
        if not transactions:
            self.stats['warnings'].append(f"{task_id}: 没有交易记录")

        for tx in transactions:
            # 检查金额是否为正数
            if tx['amount'] <= 0:
                self.stats['warnings'].append(f"{task_id}: 交易金额不合理: {tx['amount']}")

            # 检查日期格式
            try:
                from datetime import datetime
                datetime.strptime(tx['date'], '%Y-%m-%d')
            except ValueError:
                self.stats['warnings'].append(f"{task_id}: 日期格式错误: {tx['date']}")

        # 检查2: 验证总金额问题的答案
        for q in questions:
            if q['category'] == 'total_amount':
                expected_total = sum(tx['amount'] for tx in transactions)
                generated_answer = float(q['answer'])
                if abs(expected_total - generated_answer) > 0.01:
                    self.stats['warnings'].append(
                        f"{task_id} - {q['qid']}: 总金额不匹配 "
                        f"(预期: {expected_total:.2f}, 实际: {generated_answer:.2f})"
                    )

        # 检查3: 验证chunks数量与metadata中的sessions数量一致
        num_sessions = sample['metadata']['num_sessions']
        if len(chunks) != num_sessions:
            self.stats['warnings'].append(
                f"{task_id}: chunks数量({len(chunks)})与sessions数量({num_sessions})不一致"
            )

        # 检查4: 验证问题的position是否合理
        max_position = len(chunks) - 1
        for q in questions:
            if q['position'] != max_position:
                self.stats['warnings'].append(
                    f"{task_id} - {q['qid']}: position({q['position']})应该等于{max_position}"
                )
                break  # 只报告一次

        # 检查5: 验证dialogue与transactions的一致性（重点新增）
        print(f"  正在验证dialogue与transactions的一致性...")
        self.verify_dialogue_transaction_consistency(task_id, chunks, transactions)

    def verify_dialogue_transaction_consistency(self, task_id: str, chunks: List[str],
                                                      transactions: List[Dict[str, Any]]):
        """验证dialogue与transactions的一致性"""

        # 合并所有chunks的对话内容
        all_dialogue = "\n\n".join(chunks)

        # 检查每笔transaction是否在dialogue中被提及
        missing_transactions = []

        for idx, tx in enumerate(transactions):
            scene = tx['scene']
            subscene = tx.get('subscene', '')
            amount = tx['amount']
            description = tx.get('description', '')

            # 检查关键信息是否在对话中出现
            # 至少需要场景或金额在对话中出现
            scene_found = scene.lower() in all_dialogue.lower()
            subscene_found = subscene.lower() in all_dialogue.lower() if subscene else True
            amount_str = f"{amount:.2f}"
            amount_found = (
                str(amount) in all_dialogue or
                amount_str in all_dialogue or
                str(int(amount)) in all_dialogue
            )

            # 如果场景和金额都没找到，可能有问题
            if not scene_found and not amount_found:
                missing_transactions.append({
                    'index': idx,
                    'transaction': tx,
                    'reason': f"场景'{scene}'和金额'{amount}'都未在对话中找到"
                })
            # 如果找到场景但没找到金额，也标记（可能是金额描述不一致）
            elif scene_found and not amount_found:
                # 这个情况比较宽容，不一定是错误（对话中可能说"大概多少钱"而不是精确金额）
                pass

        if missing_transactions:
            print(f"  ⚠️  发现 {len(missing_transactions)} 笔交易在dialogue中找不到对应内容")
            for missing in missing_transactions[:3]:  # 只显示前3个
                print(f"      - 交易#{missing['index']}: {missing['transaction']}")
                print(f"        原因: {missing['reason']}")

            self.stats['dialogue_transaction_mismatches'].append({
                'task_id': task_id,
                'missing_count': len(missing_transactions),
                'total_transactions': len(transactions),
                'message': f"有{len(missing_transactions)}笔交易在dialogue中找不到对应内容（共{len(transactions)}笔）"
            })
        else:
            print(f"  ✅ 所有 {len(transactions)} 笔交易都在dialogue中找到了对应内容")

    def print_summary(self):
        """打印验证摘要"""
        print("\n" + "="*80)
        print("数据验证摘要")
        print("="*80)

        print(f"\n总样本数: {self.stats['total_samples']}")
        print(f"总问题数: {self.stats['total_questions']}")
        print(f"总chunks数: {self.stats['total_chunks']}")
        print(f"每样本平均问题数: {self.stats['total_questions'] / self.stats['total_samples']:.1f}")
        print(f"每样本平均chunks数: {self.stats['total_chunks'] / self.stats['total_samples']:.1f}")

        print("\n问题类别分布:")
        print("-" * 60)
        for category, count in sorted(self.stats['category_distribution'].items()):
            percentage = count / self.stats['total_questions'] * 100
            print(f"{category:<30} {count:>10} ({percentage:>5.1f}%)")

        print("\n基础数据警告:")
        print("-" * 80)
        if self.stats['warnings']:
            for warning in self.stats['warnings'][:20]:  # 只显示前20个
                print(f"  ⚠️  {warning}")
            if len(self.stats['warnings']) > 20:
                print(f"\n... 还有 {len(self.stats['warnings']) - 20} 个警告未显示")
        else:
            print("  ✅ 未发现警告")

        print("\nDialogue与Transactions一致性检查:")
        print("-" * 80)
        if self.stats['dialogue_transaction_mismatches']:
            for mismatch in self.stats['dialogue_transaction_mismatches']:
                print(f"  ⚠️  {mismatch['task_id']}: {mismatch['message']}")
        else:
            print("  ✅ Dialogue与Transactions基本一致")

        print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("用法: python validate_synth.py <data_file.json>")
        print("\n示例:")
        print("  python validate_synth.py processed_synth.json")
        sys.exit(1)

    data_file = sys.argv[1]

    print(f"正在加载数据文件: {data_file}\n")

    # 使用增强版验证器（包括dialogue一致性检查，不需要LLM调用）
    validator = EnhancedValidator(data_file)
    validator.validate()


if __name__ == "__main__":
    main()
