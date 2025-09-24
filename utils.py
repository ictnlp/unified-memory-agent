import numpy as np
import regex
import string
from collections import Counter

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def normalize_answer(s):
    """Normalize answer text for evaluation"""
    s = s.replace(',', "")
    s = regex.sub(r'\b(a|an|the|and)\b', ' ', s)
    s = ' '.join(s.split())
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return s.lower()

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth"""
    if ps:
        pred_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
        gt_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    else:
        pred_tokens = normalize_answer(prediction).split()
        gt_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def f1_multi_answer(prediction, ground_truth):
    """Calculate F1 for multi-answer questions"""
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([f1_score(pred, gt) for pred in predictions]) for gt in ground_truths])
