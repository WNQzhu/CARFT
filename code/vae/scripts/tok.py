from transformers import AutoTokenizer

def batch_decode_to_tokens(tokenizer, sequences):
    """
    将一批词元索引序列转换为对应的词元字符串列表
    :param tokenizer: 分词器
    :param sequences: 包含多个词元索引序列的列表
    :return: 包含多个词元字符串列表的列表
    """
    token_lists = []
    for sequence in sequences:
        tokens = tokenizer.convert_ids_to_tokens(sequence)
        token_lists.append(tokens)
    return token_lists

if __name__ == '__main__':
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        '/home/wnq/models/EleutherAI/pythia-14m')

    # 多个文本转词元
    texts = ["Hello, how are you?", "I'm fine, thank you.",
             "3 + 3 = 6, so the answer is 6", "\n\n$50 + $45 = $<<50+45=95>>95"]

    batch_tokens = [tokenizer.encode(text) for text in texts]
    print('batch_tokens: ', batch_tokens)
    ans = batch_decode_to_tokens(tokenizer, batch_tokens)
    print('ans: ', ans)
    ans1 = [''.join(e) for e in ans]
    print('ans1 ', ans1)
    # 批量词元转文本
    decoded_texts = tokenizer.batch_decode(batch_tokens)
    for text in decoded_texts:
        print(text)
