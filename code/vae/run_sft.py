import os
import sys
import argparse
import subprocess

cur_path = os.path.dirname(os.path.abspath(__file__))

def main(args, additional_args):
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '8'
    
    cmd = list()
    cmd.extend(["accelerate", "launch"])
    cmd.append("--config_file")
    if args.world_size == 8:
        cmd.append("${VAE_ROOT}/accelerate_configs/deepspeed_zero3_8gpus.yaml")
    elif args.world_size == 4:
        cmd.append("${VAE_ROOT}/accelerate_configs/deepspeed_zero3_4gpus.yaml")
    elif args.world_size == 1:
        cmd.append("${VAE_ROOT}/accelerate_configs/deepspeed_zero3_1gpus.yaml")
    else:
        raise ValueError(f"world size:  {args.world_size} not supported")


    if args.train_mode == 'train':
        cmd.append("scripts/paas_my_sft.py")
    else:
        cmd.append("scripts/paas_my_sft_test.py")
    # cmd.extend(["--world_size", f"{args.world_size}"])
    cmd.extend(additional_args)
    
    print(f"cmd: {cmd}")
    #result = subprocess.Popen(cmd, cwd=cur_path, env=env,
    #                          stdout=subprocess.PIPE,
    #                          stderr=subprocess.PIPE)
    expanded_cmd = [
        os.path.expandvars(args) for args in cmd
    ]
    
    print(f"expanded_cmd: {expanded_cmd}")
    
    result = subprocess.Popen(expanded_cmd, cwd=cur_path, env=env)
    result.wait()
    

if __name__ == '__main__':
    main_args = []
    additional_args = []

    is_main = False
    for v in sys.argv[1:]:
        if v == '--mode' or v == '--world_size' or v == '--train_mode':
            is_main = True
        elif v.startswith('--'):
            is_main = False
        if is_main:
            main_args.append(v)
        else:
            additional_args.append(v)
    
    del sys.argv[1:]
    sys.argv.extend(main_args)
    
    parser = argparse.ArgumentParser(description='pp2')
    # parser.add_argument('--mode', type=str, help='sft, reft, clft')
    parser.add_argument('--world_size', type=int, default=8, help='world_size')
    parser.add_argument('--train_mode', type=str, default='train', help='test')

    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--attn_implementation', type=str,
                        default="flash_attention_2",
                        help='attn_implementation')    
    parser.add_argument('--train_path', type=str, help='train_path')
    parser.add_argument('--eval_path', type=str, help='eval_path')
    parser.add_argument('--dataset_name', type=str, help='dataset_name')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='max_length')
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='gradient_accumulation_steps')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--logging_step', type=int,
                        default=500, help='logging_step')    
    parser.add_argument('--num_train_epochs', type=int, default=100,
                        help='num_train_epochs')
    parser.add_argument('--output_dir', type=str, help='output_dir')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='per_device_train_batch_size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32,
                        help='per_device_eval_batch_size')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='save_total_limit')
    parser.add_argument('--cl_coef', type=float, default=0.001, help='cl_coef')    
    
    
    # 解析参数
    args = parser.parse_args()

    main(args, additional_args)
