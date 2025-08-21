# encoding=utf-8
import copy
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


from tok import batch_decode_to_tokens
from modeling_hidden_head import AutoModelForCausalLMWithValueHead
from models import get_tok


def get_cl_model_and_tok(model_args):
    # print("model_args: ", model_args)
    model_args_copied = copy.copy(model_args)
    model_path = model_args.pop('model_name_or_path', 'unknown')
    model_name = model_args.pop('model_name', 'unknown')
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, **model_args)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, **model_args)

    tokenizer = get_tok(model_path, model_name)

    # model.resize_token_embeddings(len(tokenizer))
        
    return model, ref_model, tokenizer


def check_string(s):
    pattern = r'^[^a-zA-Z]*\d+[^a-zA-Z]*$'
    return bool(re.match(pattern, s))

from utils import (
    compare_answer_fn_mapper,
    post_process_answer_cot_fn_mapper,
    post_process_final_answer_fn_mapper)

def get_cur_list_value(mini_batch_inds, correctness):
    ret_list = []
    for idx in mini_batch_inds:
        ret_list.append(correctness[idx])
    return ret_list

#cur_0base_number_idx, cur_model_number_idx = common_sub_seq(a
#            cur_base_token_list, cur_model_token_list)

def check_string(s):
    pattern = r'^[^a-zA-Z]*\d+[^a-zA-Z]*$'
    return bool(re.match(pattern, s))

def find_num_idx(token_list):
    ret_list = []
    for i, token in enumerate(token_list):
        if check_string(token):
            #print("token is : ", token)
            ret_list.append(i)
    return ret_list

def find_num_and_idx(token_list):
    num_ret_list, word_ret_list = [], []
    for i, token in enumerate(token_list):
        if check_string(token):
            num_ret_list.append((token, i))
        else:
            word_ret_list.append((token, i))
    return num_ret_list, word_ret_list
    #return sorted(num_ret_list, key: lambda x, x[0]), \
    #    sorted(word_ret_list, key: lambda x, x[0])

def token_and_idx(token_list):
    return [(token, idx)
            for idx, token in enumerate(token_list) ]

def longest_common_subsequence(X, Y):
    # print("x and y: ", X, Y)
    m = len(X)
    n = len(Y)
    # 创建一个二维数组来存储子问题的解
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # 构建 L[m+1][n+1] 自底向上
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1][0] == Y[j - 1][0]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    index = L[m][n]
    lcs_a, lcs_b = [None] * (index), [None] * (index)

    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1][0] == Y[j - 1][0]:
            lcs_a[index - 1] = X[i - 1]
            lcs_b[index - 1] = Y[j - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs_a, lcs_b
    #print("lcs: ", lcs_a, lcs_b)
    #return ''.join(lcs)a

def _common_num_sub_seq(cur_base_token_list, cur_model_tokens_list):
    cur_base_num_tok_and_idx, cur_base_word_tok_and_idx = \
        find_num_and_idx(cur_base_token_list)
    cur_model_num_tok_and_idx, cur_model_word_tok_and_idx = \
        find_num_and_idx(cur_model_token_list)

    base_num_tok_and_idx, model_num_tok_and_idx = \
        longest_common_subsequence(cur_base_num_tok_and_idx,
                                   cur_model_num_tok_and_idx)

    if len(base_num_tok_and_idx) != 0:
        return base_num_tok_and_idx, model_num_tok_and_idx
    elif len(cur_base_num_tok_and_idx) != 0:
        return cur_base_num_tok_and_idx[:1], cur_base_num_tok_and_idx[:1]
    else:
        return cur_base_word_tok_and_idx[:1], cur_base_word_tok_and_idx[:1]

def _common_sub_seq(cur_base_token_list, cur_model_token_list):
    base_token_and_idx = token_and_idx(cur_base_token_list)
    model_token_and_idx = token_and_idx(cur_model_token_list)

    common_base_tok_and_idx, common_model_tok_and_idx = \
        longest_common_subsequence(base_token_and_idx,
                                   model_token_and_idx)

    return common_base_tok_and_idx, common_model_tok_and_idx

def _common_sub_seq_idx_only(cur_base_token_list,
                            cur_model_token_list):
    base_token_and_idx = token_and_idx(cur_base_token_list)
    model_token_and_idx = token_and_idx(cur_model_token_list)

    common_base_tok_and_idx, common_model_tok_and_idx = \
        longest_common_subsequence(base_token_and_idx,
                                   model_token_and_idx)

    common_base_idx = [
        x[1] for x in common_base_tok_and_idx
    ]
    common_model_idx = [
        x[1] for x in common_model_tok_and_idx
    ]
    return common_base_idx, common_model_idx

def batch_common_sub_seq_idx_only(cur_base_token_seq_list,
                            cur_model_token_seq_list):
    ret_base_seq_idx_list, ret_model_seq_idx_list = [], []

    for cur_base_token_seq, cur_model_token_seq in zip(
            cur_base_token_seq_list,
            cur_model_token_seq_list):
        base_token_and_idx = token_and_idx(cur_base_token_seq)
        model_token_and_idx = token_and_idx(cur_model_token_seq)

        common_base_tok_and_idx, common_model_tok_and_idx = \
            longest_common_subsequence(base_token_and_idx,
                                   model_token_and_idx)

        common_base_idx = [
            x[1] for x in common_base_tok_and_idx
        ]
        common_model_idx = [
            x[1] for x in common_model_tok_and_idx
        ]
        ret_base_seq_idx_list.append(common_base_idx)
        ret_model_seq_idx_list.append(common_model_idx)

    return ret_base_seq_idx_list, ret_model_seq_idx_list
    



if __name__ == '__main__':
    ss_list = ["134ADADAFADFas134", "134", " 134", "9 6", "^134",
               "abc134abc", "abcd", "12a13b13c"
               ]
               
    for ss in ss_list:
        print("check: ", check_string(ss))

    X = [("A", 1), ("G", 2), ("T", 3), ("A", 4), ("B", 5)]
    Y = [("G", 1), ("X", 2), ("T", 3), ("X", 4), ("A", 5), ("Y", 6)]
    print("最长公共子序列是", longest_common_subsequence(X, Y))    
        
    X = [("A", 1)]
    Y = [("G", 1)]
    print("最长公共子序列是", longest_common_subsequence(X, Y))

    X = ["A", "G", "T", "A", "B",]
    Y = ["G","X", "T", "X", "A", "Y"]

    print("run1: ", _common_sub_seq(X, Y))
    print("run2: ", _common_sub_seq_idx_only(X, Y))

    X_list = [X,X,X,X]
    Y_list = [Y,Y,Y,[]]
    

    print("batch run: ", batch_common_sub_seq_idx_only(X_list, Y_list))
