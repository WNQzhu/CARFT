import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def parse_gen_log(log_path):
    ret = []
    pat = '{"qr"'
    with open(log_path, 'r') as f:
        for line in f:
            if line is not None:
                if pat in line:
                    idx = line.find(pat)
                    left = line[idx:]
                    print("left is: ", left)

                    print("--"* 40)
                    try:
                        d = eval(left.strip())
                    except Exception as e:
                        print(f"err: {e}")
                    else:
                        ret.append(d)
    return ret

def extract(d_list):
    ret_list = []
    cot_trigger = "\nAnswer reasoning:\n"
    for d in d_list:
        tmp_list = []
        for qr, pred, ans in zip(d['qr'], d['val'], d['label']):
            idx = qr.find(cot_trigger)
            if idx == -1:
                continue
            query = qr[:idx]
            cot = qr[idx + len(cot_trigger):]
            if pred == "None":
                pred = None
            if ans == "None":
                ans = None
            tmp_list.append((query, cot, pred, ans))
        ret_list.append(tmp_list)
    return ret_list    

if __name__ == '__main__':
    #log_path = "/home/wnq/pp1/code/dpo.log"
    
    log_path = "/home/wnq/bank/bundle.rollout/code/vae/t.log"

    #ret = parse_train_log(log_path)
    ret = parse_gen_log(log_path)

    print('ret: ', ret)
    ret_list = extract(ret)

    # print("ret_list: ", ret_list[0][0])
 
    sample = ret_list[0][0]
    
    print("sample[0]: ", sample[0])
    print("--" * 40)
    print("sample[1]: ", sample[1])
    print("--" * 40)    
    print("sample[2]: ", sample[2])
    print("--" * 40)    
    print("sample[3]: ", sample[3])

    print("==" * 40)
    sample = ret_list[0][1]    
    print("sample[0]: ", sample[0])
    print("--" * 40)
    print("sample[1]: ", sample[1])
    print("--" * 40)    
    print("sample[2]: ", sample[2])
    print("--" * 40)    
    print("sample[3]: ", sample[3])
