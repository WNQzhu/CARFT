# encoding=utf-8

from utils import (
    compare_answer_fn_mapper,
    post_process_answer_cot_fn_mapper,
    post_process_final_answer_fn_mapper)

def get_rewards(args, programs, answer_values):
    execute_fn = post_process_answer_cot_fn_mapper[
        (args.engine, args.dataset_name)]
    correctness = []
    for i, extracted_ans in enumerate(execute_fn(programs,
                                                 args.cot["answer_trigger"])):
        target_value = post_process_final_answer_fn_mapper[
            args.dataset_name](answer_values[i])
        if extracted_ans is not None:
            if args.engine == 'game24' or args.engine == 'calcn':
                is_correct = extracted_ans
            else:
                if compare_answer_fn_mapper[
                        args.dataset_name](extracted_ans, target_value):
                    is_correct = 1
                else:
                    is_correct = 0.1
                    # for mathqa, even though it can executed, if the results is not within a,b,c,d,xxx, still zero reward
                    # because we want to give some reward for the prediction that able to select one of the answer
                    # for example, the executed answer is "{}" in mathqa.
                    # THIS PART IS TO BE DECIDED.
                    # if src_name == 'mathqa' and not (len(extracted_ans) == 1 and extracted_ans.isalpha()):
                    #     is_correct = 0
        else:
            is_correct = 0
        correctness.append(is_correct)
    return correctness

