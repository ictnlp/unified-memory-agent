import re
from collections import Counter
import regex
import string
# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s: str) -> str:
    assert isinstance(s, str), f"Expected string, got {type(s)}"
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    # remove spaces
    string = string.replace(" ", "")

    return string

def _extract_answer_from_response(predicted_answer):
    """
    Extracts content within <answer></answer> tags or from \\box{}/\\boxed{} format using regex.
    """
    # Attempt to match <answer></answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, predicted_answer, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Try to match \\boxed{} format (preferred for math/classification)
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    boxed_match = re.search(boxed_pattern, predicted_answer)

    if boxed_match:
        return boxed_match.group(1).strip()

    # If \\boxed{} not found, try to match \\box{} format for backwards compatibility
    box_pattern = r'\\box\{([^}]*)\}'
    box_match = re.search(box_pattern, predicted_answer)

    if box_match:
        return box_match.group(1).strip()

    # If none of the above patterns found, return the original answer
    return predicted_answer.strip()

def normalize_answer(s):
    """Normalize answer text for evaluation"""
    s = s.replace(',', "")
    def remove_articles(text):
        return regex.sub(r'\\b(a|an|the|and)\\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Combine Memagent and Memalpha reward functions
def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    tool_reward = 0.0
    if extra_info and 'tool_rewards' in extra_info:
        tool_rewards = extra_info['tool_rewards']
        num_tools = extra_info['num_tools']
        if tool_rewards.size:
            tool_reward = tool_rewards.sum() / num_tools.item()
            tool_reward = 0.5 * tool_reward
    outcome_reward = 0.0
    format_reward = 0.0
    if extra_info['is_final']:
        # Handle ground_truth as list for multi-query cases
        # print(f'extra_info["is_final"]: {extra_info["is_final"]}')
        # print(f'type(extra_info["is_final"]): {type(extra_info["is_final"])}')
        # print(f'len(ground_truth): {len(ground_truth)}')
        ground_truth = ground_truth[extra_info['is_final'][0] - 1]
        
        solution_str = solution_str.split("</tool_responses>assistant")[-1].strip()
        solution_str = solution_str[-300:].lower()
        try:
            solution_str = last_boxed_only_string(solution_str)
            solution_str = remove_boxed(solution_str)
            format_reward = 0.5
        except:
            solution_str = None

        if solution_str is not None:
            if data_source == 'hotpotqa':
                # outcome_reward = compute_score(solution_str, ground_truth)
                outcome_reward = 1.0 if is_equiv(solution_str, ground_truth.lower()) else 0.0
            elif data_source == 'locomo':
                pred_norm = normalize_answer(solution_str)
                gold_norm = normalize_answer(ground_truth)
                pred_tokens = pred_norm.split()
                gold_tokens = gold_norm.split()
                if not pred_tokens or not gold_tokens:
                    f1_score = 0.0
                    precision = 0.0
                    recall = 0.0
                else:
                    common = Counter(pred_tokens) & Counter(gold_tokens)
                    num_same = sum(common.values())
                    
                    if num_same == 0:
                        precision = 0.0
                        recall = 0.0
                        f1_score = 0.0
                    else:
                        precision = 1.0 * num_same / len(pred_tokens)
                        recall = 1.0 * num_same / len(gold_tokens)
                        f1_score = (2 * precision * recall) / (precision + recall)
                outcome_reward = f1_score
            elif data_source == 'memalpha_booksum':
                keywords = ground_truth.split(",")
                keywords = [x.strip() for x in keywords]
                hit = 0
                for keyword in keywords:
                    if keyword.lower() in solution_str.lower():
                        hit += 1
                outcome_reward = hit / len(keywords)
            elif data_source == 'memalpha_hotpotqa' or data_source == 'memalpha_squad':
                answer_text = ground_truth.get('text', ground_truth) if isinstance(ground_truth, dict) else str(ground_truth)
                outcome_reward =  1.0 if answer_text.lower() in solution_str.lower() else 0.0
            elif data_source.startswith('memalpha_icl') or data_source == 'memalpha_pubmed-rct':
                # PUBMED dataset evaluation: MUST be ONLY a single digit
                extracted_answer = _extract_answer_from_response(solution_str)
                # Remove quotes and strip whitespace
                extracted_answer = extracted_answer.strip('"\'').strip()
                # STRICT pattern: must be EXACTLY a single digit with nothing else
                single_digit_pattern = r'^\d+$'
                if not re.match(single_digit_pattern, extracted_answer):
                    outcome_reward =  0.0
                else:
                    gold_num = str(ground_truth).strip('"\'').strip()
                    outcome_reward = 1.0 if extracted_answer == gold_num else 0.0
            elif data_source == 'memalpha_lme_train':
                raise NotImplementedError("Memalpha LME train reward function not implemented yet.")
            elif data_source == 'memalpha_perltqa':
                if ";" in ground_truth:
                    ground_truth = ground_truth.split(";")
                    total_hit = 0
                    for answer in ground_truth:
                        if answer.lower().strip() in solution_str:
                            total_hit += 1
                    outcome_reward = total_hit / len(ground_truth)
                else:
                    outcome_reward = 1.0 if ground_truth.lower() in solution_str.lower() else 0.0
            elif data_source == 'synth':
                outcome_reward = 1.0 if ground_truth.lower() in solution_str.lower() else 0.0
            else:
                raise NotImplementedError(f"Reward function for data source {data_source} not implemented.")

    outcome_reward += format_reward
    total_score = outcome_reward + tool_reward
    return {
        "score": total_score,
        "outcome_reward": outcome_reward,
        "tool_reward": tool_reward,
        "format_reward": format_reward
    }