import re
import torch
import json
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

from my_python_engine import run_python_code

TIMEOUT = 10


post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
    'mathqa-numeric': lambda x: float(x),
}

def floatify(s):
    try:
        return float(s)
    except:
        return None

### the answer_cot is a list of answer_cot

post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot, answer_trigger: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'svamp'): lambda answer_cot, answer_trigger: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa'): lambda answer_cot, answer_trigger: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa-numeric'): lambda answer_cot, answer_trigger: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('nl', 'gsm8k'): lambda answer_cot, answer_trigger: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'svamp'): lambda answer_cot, answer_trigger: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'mathqa'): lambda answer_cot, answer_trigger: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
    ('nl', 'mathqa-numeric'): lambda answer_cot, answer_trigger: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
}
compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
    'mathqa-numeric': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
}

def extract_pred_values(pred_list, args):
    pred_cot_list = [
        pred.strip().split(args.cot["cot_trigger"])[-1].strip()
        for pred in pred_list
    ]
    execute_fn = post_process_answer_cot_fn_mapper[(args.engine, args.dataset_name)]
    pred_value_list = execute_fn(pred_cot_list, args.cot["answer_trigger"])

    return pred_value_list

def read_raw_dataset(train_path, eval_path):
    raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(train_path,'r'))),
            'test': Dataset.from_list(json.load(open(eval_path,'r'))),
    })
    return raw_dataset



def compute_acc(pred_list, ans_list, args):
    corr_cnt, total_cnt = 0, 0

    for pred, ans in zip(pred_list, ans_list):
        # print('pred and ans:', pred, ans)
        total_cnt += 1
        if pred is not None:
            is_corr = compare_answer_fn_mapper[args.dataset_name](pred, ans)
            corr_cnt += is_corr
    return corr_cnt, total_cnt
    #return 1.0 * corr_cnt / max(1.0, total_cnt)
        
from torch.distributed import all_reduce, ReduceOp
def do_gather(var):
    var = torch.FloatTensor(var).cuda()
    all_reduce(var, op=ReduceOp.SUM)
    var = var.cpu().numpy().tolist()
    return var

def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """
    if group is None:
        group = torch.distributed.group.WORLD
    group_size = 1 if group is None else group.size()
    allgather_tensor = [torch.zeros_like(tensor) for _ in range(group_size)]
    torch.distributed.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)
    return allgather_tensor

from trl.core import masked_mean, masked_var
def allgather_masked_whiten(values, mask, shift_mean=False):
    """Whiten values with all-gathered masked values.

    Args:
        values: (bs, ...)
        mask: (bs, ...)
        shift_mean: bool

    Returns:
        whitened values, (bs, ...)
    """
    allgather_values = allgather(values)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_values {allgather_values.shape}, {allgather_values[0, 0:3]}')

    allgather_mask = allgather(mask)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_mask {allgather_mask.shape}, {allgather_mask[0, 0:3]}')

    global_mean = masked_mean(allgather_values, allgather_mask)
    global_var = masked_var(allgather_values, allgather_mask)
    whitened = (values - global_mean) * torch.rsqrt(global_var + 1e-8)
    if shift_mean:
        whitened += global_mean
    return whitened


import scipy.signal as scipy_signal
def discount_cumsum(rewards, discount):
    return scipy_signal.lfilter([1], [1, -discount], x=rewards[::-1])[::-1]


def extract_numbers(s):
    pattern = r'\d+'
    numbers = re.findall(pattern, s)
    return numbers

if __name__ == '__main__':
    input_string = "abc123def456ghi789"
    result = extract_numbers(input_string)
    print("抽取到的数字为:", result)
    
