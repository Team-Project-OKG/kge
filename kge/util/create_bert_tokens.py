import numpy as np
import pandas as pd
import torch
import transformers
import math
import os
import sys

def create_bert_tokens(input_file, output_file, tokenizer_class=transformers.BertTokenizer, pretrained_weights="prajjwal1/bert-tiny"):
    df = pd.read_csv(input_file, delimiter="\t", header=None)

    batch_size = 5000

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    all_tokens = None

    for batch_number in range(math.ceil(len(df)/batch_size)):

        batch_data = df[batch_number * batch_size:(batch_number + 1) * batch_size]
        print(batch_number, " of ", math.ceil(len(df)/batch_size), ": ", batch_number / math.ceil(len(df)/batch_size) * 100, "%")
        tokenized = batch_data[1].astype(np.str).apply((lambda x: tokenizer.encode(x, add_special_tokens=False)))

        if all_tokens is None:
            all_tokens = tokenized
        else:
            all_tokens = all_tokens.append(tokenized)

    #max_len = 0
    #for i in all_tokens.values:
    #    if len(i) > max_len:
    #        max_len = len(i)

    #padded = np.array([i + [0]*(max_len-len(i)) for i in all_tokens.values])

    #attention_mask = np.where(padded != 0, 1, 0)

    #input_ids = torch.tensor(padded, device=0)
    #attention_mask = torch.tensor(attention_mask, device=0)

    with open(output_file, "w") as file:
        for i, value in all_tokens.items():
            file.write(str(i) + "\t" + " ".join([str(x) for x in value]) + os.linesep)


# give file name as first command line argument
if __name__ == '__main__':
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "/home/alex/PycharmProjects/kge/data/olpbench/relation_ids.del"
        output_file = "../pretrained/bert-tiny_relation_id_tokens_ids_map.del"
    create_bert_tokens(input_file, output_file)
