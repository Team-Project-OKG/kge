import numpy as np
import pandas as pd
import torch
import transformers
import math
import sys
import pathlib

from os import linesep


def create_bert_tokens(input_file, output_file, pretrained_weights="bert_uncased_L-4_H-256_A-4"):
    """
    Generates BERT token id map from a input file of the structure:
    ID\tTextRepresentation
    """
    df = pd.read_csv(input_file, delimiter="\t", header=None)

    batch_size = 5000

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_weights)

    all_tokens = None

    for batch_number in range(math.ceil(len(df)/batch_size)):

        batch_data = df[batch_number * batch_size:(batch_number + 1) * batch_size]
        print(batch_number, " of ", math.ceil(len(df)/batch_size), ": ", batch_number / math.ceil(len(df)/batch_size) * 100, "%")
        tokenized = batch_data[1].astype(np.str).apply((lambda x: tokenizer.encode(x, add_special_tokens=False)))

        if all_tokens is None:
            all_tokens = tokenized
        else:
            all_tokens = all_tokens.append(tokenized)

    with open(output_file, "w") as file:
        for i, value in all_tokens.items():
            file.write(str(i) + "\t" + " ".join([str(x) for x in value]) + linesep)


# give input mapping file name, output file name and the huggingface pretrained weights name as command line arguments
if __name__ == '__main__':
    position = pathlib.Path(__file__)

    if len(sys.argv) > 2:
        input_file = pathlib.Path(sys.argv[1])
        output_file = pathlib.Path(sys.argv[2])
        pretrained_weights = sys.argv[3]
    else:
        input_file = position.resolve().parent.parent / "data" / "olpbench" / "relation_ids.del"
        output_file = position.resolve().parent.parent / "data" / "olpbench" / "bert_relation_id_tokens_ids_map.del"
        pretrained_weights = "bert_uncased_L-4_H-256_A-4"
    create_bert_tokens(input_file, output_file, pretrained_weights)
