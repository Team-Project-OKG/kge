import numpy as np
import pandas as pd
import torch
import transformers
import gc
import math

from gensim.models import KeyedVectors


df = pd.read_csv("/home/alex/PycharmProjects/kge/data/olpbench/entity_token_id_map.del", delimiter="\t", header=None)

if df.loc[0][1] == "[unmapped]":
    df.drop(0, inplace=True)

batch_size = 5000

model_class, tokenizer_class, pretrained_weights = (
transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights).to(0)

kv_model = KeyedVectors(model.config.dim)

for batch_number in range(math.ceil(len(df)/batch_size)):

    batch_data = df[batch_number * batch_size:(batch_number + 1) * batch_size]
    print(batch_number)
    tokenized = batch_data[1].astype(np.str).apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded, device=0)
    attention_mask = torch.tensor(attention_mask, device=0)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].cpu().numpy()

    del input_ids, attention_mask, last_hidden_states
    gc.collect()
    torch.cuda.empty_cache()

    kv_model.add(batch_data[1].astype(np.str).values, features)


kv_model.save_word2vec_format("../pretrained/bert-tokens.txt")
kv_model.save("../pretrained/bert-tokens")
print("test")