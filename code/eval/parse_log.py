import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json

def parse_train_log(log_path):
    ret = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line is not None:
                if "{'loss'" in line:
                    ret.append(eval(line.strip()))
    return ret

def parse_myacc_log(log_path):
    ret = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line is not None:
                pat = "{'eval_myacc'"
                if pat in line:
                    _line = line.strip()
                    start_idx = _line.find(pat)
                    ret.append(eval(_line[start_idx:]))
    return ret


def parse_reft_loss(log_path):
    ret = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line is not None:
                if "{'loss/policy_avg'" in line:
                    try:
                        val = eval(line.strip())
                    except Exception as e:
                        pass
                    else:
                        ret.append(val)
    return ret

def draw_reft(dict_list):
    policy_avg_list = []
    value_avg_list = []
    epochs = []
    for d in dict_list:
        policy_avg_list.append(d['loss/policy_avg'])
        value_avg_list.append(d['loss/value_avg'])
        epochs.append(d['epoch'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, policy_avg_list, label="policy_avg",  color='tab:blue')
    ax.plot(epochs, value_avg_list,  label="value_avg", color='tab:orange')
    plt.legend()


    for e, c in zip(epochs, policy_avg_list):
        print(f"({e}, {c})", end=" ")
    print('\n--------------')
    for e, r in zip(epochs, value_avg_list):
        print(f"({e}, {r})", end=" ")
    
    plt.show()    


def draw_reward_and_margin(dict_list):
    chosen_reward = []
    rejected_reward = []
    margin = []
    epochs = []
    for d in dict_list:
        chosen_reward.append(d['rewards/chosen'])
        rejected_reward.append(d['rewards/rejected'])
        margin.append(d['rewards/margins'])
        epochs.append(d['epoch'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, chosen_reward, color='tab:blue')
    ax.plot(epochs, rejected_reward, color='tab:orange')
    ax.plot(epochs, margin, color='tab:red')

    for e, c in zip(epochs, chosen_reward):
        print(f"({e}, {c})", end=" ")
    print('\n--------------')
    for e, r in zip(epochs, rejected_reward):
        print(f"({e}, {r})", end=" ")
    print('\n--------------')
    for e, m in zip(epochs, margin):
        print(f"({e}, {m})", end=" ")
    

    plt.show()    

def draw_loss(dict_list):
    losses = []
    margin = []
    epochs = []
    for d in dict_list:
        losses.append(d['loss'])
        epochs.append(d['epoch'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, losses, color='tab:blue')

    for e, c in zip(epochs, losses):
        print(f"({e}, {c})", end=" ")    

    plt.show()    

def draw_myacc(dict_list):
    myaccs = []
    margin = []
    epochs = []
    for d in dict_list:
        myaccs.append(d['eval_myacc'])
        epochs.append(round(d['epoch']))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, myaccs, color='tab:blue')

    for e, c in zip(epochs, myaccs):
        print(f"({e}, {c})", end=" ")    

    plt.show()

def main(args):

    log_path = args.log_path

    ret = parse_myacc_log(log_path)

    #draw_reward_and_margin(ret)
    #draw_loss(ret)

    draw_myacc(ret)

    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='hot map')
    parser.add_argument('--log_path', type=str, help='parser input')

    args = parser.parse_args() 
    main(args)
