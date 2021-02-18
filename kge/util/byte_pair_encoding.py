import re
from kge import Dataset
import torch
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Algorithm:
# Stop if number of max. iteration or desired number of tokens are reached
    # Count frequency of each word
    # Split words into characters
    # Count frequency of each consecutive byte-pair
    # Find most frequent byte-pair
    # Merge most frequent byte-pair into new token
    # Replace individual character pairs by new, most frequent token
# output: vocabulary

class BytePairEncodingVocab:

    def __init__(self,
                 olp_dataset: Dataset,
                 iterations_entities,
                 iterations_relations
                 ):

        self.entity_sub_token_ids = None
        self.relation_sub_token_ids = None
        self.num_entity_sub_tokens = None
        self.num_relation_sub_tokens = None
        self.entity_sub_token_lookup = None
        self.relation_sub_token_lookup = None

        # run bpe and init vocab
        self.create_sub_token_vocabs(olp_dataset, iterations_entities, iterations_relations)

    def get_bigrams(self, vocab):
        split_vocab = [x.split(' ') for x in vocab]
        bigrams = np.array([' '.join(sub_token[i:i + 2]) for sub_token in split_vocab for i in range(len(sub_token) - 1)])
        return bigrams

    # TODO: improve performance
    #'</w>'.join(''.join(entity_token_vocab).split('</w>'))
    #if best_bigram[0] in x and best_idx[1] in
    def merge(self, best_bigram, vocab):
        first_half, second_half = best_bigram.split(" ")
        begin = '^' + first_half + '\s' + second_half + '\s'
        mid = '\s' + first_half + '\s' + second_half + '\s'
        end = '\s' + first_half + '\s' + second_half + '$'
        begin_end = '^' + first_half + '\s' + second_half + '$'
        pattern = re.compile('|'.join([begin, mid, end, begin_end]))
        vocab = [pattern.sub(' ' + best_bigram.replace(" ", "") + ' ', x).strip() for x in vocab]
        # entity_token_vocab = [x.replace(best_bigram, best_bigram.replace(" ", "")) for x in entity_token_vocab]
        # ' '.join(entity_token_vocab).replace(best_bigram, best_bigram.replace(" ", ""))
        return vocab

    def run_bpe(self, num_iters, token_vocab):
        t1 = time.time()
        for i in range(num_iters):
            bigrams = self.get_bigrams(token_vocab)
            if len(bigrams) == 0:
                print("Could not find any bigrams, aborting...")
                break
            # Count frequency of each consecutive byte-pair
            unique_bigrams, counts = np.unique(bigrams, return_counts=True)
            # Find most frequent byte-pair
            best_idx = np.argmax(counts)
            best_bigram = unique_bigrams[best_idx]
            # Merge most frequent byte-pairs in token vocabulary
            token_vocab = self.merge(best_bigram, token_vocab)

            # OPTIONAL:
            # count all sub-tokens in token vocabulary
            sub_token_vocab = np.unique(np.array(' '.join(token_vocab).split()))
            print("Number of sub-tokens: {} after iteration {}".format(len(sub_token_vocab), i + 1))

            if i > 0 and (i + 1) % 10 == 0:
                print("Number of sub-tokens: {} after iteration {}".format(len(sub_token_vocab), i + 1))
                t2 = time.time()
                print("{}s per {} iterations".format(t2-t1, 10))
                t1 = time.time()
        t2 = time.time()
        print("Total runtime: ", t2-t1)
        return token_vocab

    # Map from entity/relation token id's to sub-token id's
    # Input: token sequences, e.g. 1 4 15 9 8 2 0 0 0  -> text:  1 the film was very exciting for the viewer 2 0 0 0
    # Output: byte-pair encoded subword token sequence, e.g. 1 13 22 11 5 4 3 7 8 9 32 55 89 2 0 0 0

    # Starts with character-level sub-tokens. If no bigrams are found -> token-level sub_tokens
    # e.g.: 'h o h e n z o l l e r n s </w>' -> 'ho hen zoll ern s</w>' -> 'hohenzollerns</w>'

    # Define following order:
    # entity -> token -> sub-token -> character
    def create_sub_token_vocabs(self, olp_dataset, iterations_entities, iterations_relations):
        entity_tokens_str = olp_dataset._meta['entity_token_ids']
        relation_tokens_str = olp_dataset._meta['relation_token_ids']

        # get vocabulary, add stop word '</w>' at the end of each token
        entity_tokens = [' '.join(x) + ' </w>' for x in entity_tokens_str[4:]]
        total_entity_tokens = len(entity_tokens)
        total_entity_sub_tokens = len(np.unique(np.array(' '.join(entity_tokens).split())))
        print("Number of sub-tokens at the beginning: {}/{}".format(total_entity_sub_tokens, total_entity_tokens))
        entity_tokens_bpe = self.run_bpe(iterations_entities, entity_tokens)
        #entity_token_vocab = {entity_tokens_str[4:][idx]: x.split() for idx, x in enumerate(entity_tokens_bpe)}  # Optional: add  + '</w>' to key
        #entity_token_vocab_ = {idx + 4: x.split() for idx, x in enumerate(entity_tokens_bpe)}

        entity_sub_tokens = np.unique(np.array(' '.join(entity_tokens_bpe).split()))
        self.num_entity_sub_tokens = len(entity_sub_tokens)
        self.entity_sub_token_ids = {x: idx for idx, x in enumerate(entity_sub_tokens)}

        self.entity_sub_token_lookup = {idx + 4: np.array([self.entity_sub_token_ids[y]
                                                    for y in x.split()])
                                                    for idx, x in enumerate(entity_tokens_bpe)
                                          }
        self.entity_sub_token_lookup[3] = np.array([-1])  # keep end of tokens

        relation_tokens = [' '.join(x) + ' </w>' for x in relation_tokens_str[4:]]
        total_relation_tokens = len(relation_tokens)
        total_relation_sub_tokens = len(np.unique(np.array(' '.join(relation_tokens).split())))
        print("Number of sub-tokens at the beginning: {}/{}".format(total_relation_sub_tokens, total_relation_tokens))
        relation_tokens_bpe = self.run_bpe(iterations_relations, relation_tokens)
        self.relation_token_vocab = {relation_tokens_str[3:][idx]: x.split() for idx, x in enumerate(relation_tokens_bpe)}
        #self.relation_token_vocab_ = {idx + 4: x.split() for idx, x in enumerate(relation_tokens_bpe)}
        relation_sub_tokens = np.unique(np.array(' '.join(relation_tokens_bpe).split()))
        self.num_relation_sub_tokens = len(relation_sub_tokens)
        self.relation_sub_token_ids = {x: idx for idx, x in enumerate(relation_sub_tokens)}
        self.relation_sub_token_lookup = {idx + 4: np.array([self.relation_sub_token_ids[y]
                                                    for y in x.split()])
                                                    for idx, x in enumerate(relation_tokens_bpe)
                                          }
        self.relation_sub_token_lookup[3] = np.array([-1])


# Todo: encode unseen words
class BPESubTokenEmbedder:
    """Byte-Pair Encoding Sub-Token Embedder takes in token index sequence.
       Splits tokens into sub-tokens, embeds sub-tokens into tensors and
       returns sub-token embeddings.
    """
    def __init__(self,
                 bpe_vocab: BytePairEncodingVocab,
                 configuration_key: str,
                 ):

        if "entity" in configuration_key:
            self.sub_token_lookup = bpe_vocab.entity_sub_token_lookup

        if "relation" in configuration_key:
            self.sub_token_lookup = bpe_vocab.relation_sub_token_lookup

        self.bpe_vocab = bpe_vocab


    def get_sub_tokens_from_tokens(self, token_indexes):
        import time
        t1 = time.time()
        # Create sub-word/token embeddings
        token_indexes = token_indexes[:, 1:]
        # entity_token_str idx 2 -> 2 (end), 3 -> [x, x , x]
        # token_idx_sequence: 2 x x x 3 0 0 0 -> lookup -> 1d sub-token sequence -> split at 3 -> tensor with padding
        t2 = time.time()
        # convert token to sub-tokens (dict lookup enforces cpu)
        #token_indexes[torch.where(token_indexes == 3)] = -1   # -1 is unique end of sequence token
        token_idx_sequence = token_indexes[token_indexes > 2].cpu().detach().numpy() # split at 3
        t3 = time.time()
        # [x for x in np.split(sub_tokens_np, offsets_np)[:-1] if x[-1] == 3]
        # -------- Solution A -----------:
        # np.where(np.array([x.tolist() for x in sub_tokens]) == 3)
        sub_tokens_np = np.concatenate([self.sub_token_lookup[token_idx] for token_idx in token_idx_sequence])[:-1]
        t4 = time.time()
        sub_tokens_np = np.insert(sub_tokens_np, 0, -1)  # insert 3 at idx 0 -> cut token 3 at last step
        t5 = time.time()
        offsets_np = np.where(sub_tokens_np == -1)[0] # + 1 # solution for self.sub_token_lookup[3] = np.array([3])
        t6 = time.time()
        # Todo: check if we can remove  [:-1] from sub_tokens_np and delete appending len(sub_tokens_np)
        offset_by_len = np.diff(np.append(np.insert(offsets_np, 0, 0), len(sub_tokens_np)))
        t7 = time.time()
        sub_token_tensor = torch.tensor(sub_tokens_np, device=token_indexes.device)
        t8 = time.time()
        sub_tokens_split = torch.split(sub_token_tensor, offset_by_len.tolist())[1:]
        t9 = time.time()
        #sts_np = [x.cpu().detach().numpy() for x in sub_token_tensor_single]         # Debug
        # Todo: check if this is correct for olpbench big
        output = (pad_sequence(sub_tokens_split).transpose(0, 1))[:,1:] #.cpu().detach().numpy()
        t10 = time.time()
        # Todo: option to keep or drop start and end of sub-token sequence
        #if keep_start_end_token
        #o1 = output.cpu().detach().numpy()
        #print("Total: ", t10-t1)
        '''
        print("1,2", t2 - t1)
        print("2,3", t3 - t2)
        print("3,4", t4 - t3)
        print("4,5", t5 - t4)
        print("5,6", t6 - t5)
        print("6,7", t7 - t6)
        print("7,8", t8 - t7)
        print("8,9", t9 - t8)
        print("9,10", t10 - t9)
        '''

        # -------- /Solution A -----------:

        t = 0

        '''
        # -------- Solution B -----------
        #sub_tokens = np.array([self.sub_token_lookup_[token_idx] for token_idx in token_idx_sequence])
        # offsets = np.where(np.array([len(x.shape) for x in sub_tokens]) == 0) # solution for self.sub_token_lookup[3] = np.array(3)
        #offsets = np.where(np.array([len(x) for x in sub_tokens]) == 1)  # solution for self.sub_token_lookup[3] = np.array([3])
        #sub_tokens_np = np.concatenate([self.sub_token_lookup[token_idx] for token_idx in token_idx_sequence])[:-1]

        # -------- /Solution B -----------:
        t = 0

        #test = np.split(sub_tokens_np, offsets_np[0])
        # 14576 -> 162, 164, 587, 589, 1169, 1195, 1197, 1429, 1447, 1449, 1993, 1996
        # length of boxes of 3


        #torch.split(sub_tokens_tensor, offsets.tolist())
        #sub_tokens_tensor = torch.tensor(np.concatenate(sub_tokens), device=token_indexes.device)

        #offsets_tensor = torch.where(sub_tokens_tensor == 3) #np.where(np.array([x.tolist() for x in sub_tokens]) == 3)
        #sub_token_sequence_tensor = torch.split(sub_tokens_tensor, offsets_tensor)
        sub_tokens = [self.sub_token_lookup[token_idx] for token_idx in token_idx_sequence]

        # split list at 3 into 2D list -> pad and create tensor
        #offsets = np.where(np.asarray([x.tolist() for x in sub_tokens], dtype="object") == 3)
        #offsets = np.where(np.array([x.tolist() for x in sub_tokens]) == 3)
        #offsets = np.where(np.array([x.tolist() for x in sub_tokens]) == 3)
        t6 = time.time()
        #offsets = np.where(sub_tokens_np == 3)[0] + 1
        offsets_ = np.where(np.array([len(x) for x in sub_tokens]) == 1)
        debug = [idx for idx, x in enumerate(sub_tokens) if -1 in x.tolist()]
        sub_token_sequence = np.split(sub_tokens, offsets_[0] + 1)#[:-1]  gives same result as keeping [:-1] -> filtered later

        #[np.concatenate(row) for row in sub_tokens if len(row.shape) > 0 and row.size > 0]
        #for idx, row in enumerate(sub_tokens):
        #    test = np.concatenate(row)
            #if len(row.shape) > 0 and row.size > 0]:

        #sub_token_sequence = torch.tensor_split(sub_tokens, offsets[0] + 1)[:]
        t7 = time.time()
        # without padding
        #sub_token_sequence_arr = [x.tolist() for row in sub_token_sequence for x in row]
        #sub_token_sequence_t = [[torch.tensor(x) if idx < len(row) - 1 for idx, x in enumerate(row)] for row in sub_token_sequence]

        # slow
        #sub_token_sequence = [[torch.tensor(x, device=token_indexes.device) for x in row if x.size > 0 and len(x.shape) > 0] for row in sub_token_sequence]

        # faster
        #sub_token_sequence_tensor = [
        #    [torch.tensor(x) for x in row if x.size > 0 and len(x.shape) > 0] for row in
        #    sub_token_sequence]

        #sub_token_sequence_tensor = [[torch.tensor(x) for x in row[:-1]] for row in sub_token_sequence]

        t8 = time.time()
        #sub_token_tensor = [torch.cat(row) for row in sub_token_sequence_tensor if len(row) > 1]

        # fastest
        #sub_token_tensor = [torch.tensor(np.concatenate(row[:-1], axis=None)) for row in sub_token_sequence]
        sub_token_tensor = [torch.tensor(np.concatenate(row[:-1], axis=None)) for row in sub_token_sequence if len(row) > 1]
        sts_2 = [np.concatenate(row[:-1]) for row in sub_token_sequence if len(row) > 1]
        t9 = time.time()

        #padded_sub_token_tensor = (pad_sequence(sub_token_tensor).view(1, 0)).to(token_indexes.device)
        #padded_sub_token_tensor = (pad_sequence(sub_token_tensor).to(token_indexes.device).permute(1,0))
        padded_sub_token_tensor = (pad_sequence(sub_token_tensor).to(token_indexes.device).transpose(0, 1))
        o2 = padded_sub_token_tensor.cpu().detach().numpy()
        diff_np = [(x - padded_sub_token_tensor[idx]).cpu().detach().numpy() if idx < padded_sub_token_tensor.shape[
            0] else np.ones(padded_sub_token_tensor.shape[1], dtype='int64') for idx, x in enumerate(output)]
        t10 = time.time()
        diff = np.sum(diff_np, axis=1)
        diff_idxs = np.where(diff != 0)
        print("RTT2: ", t10-t5)
        from torch.nn.functional import pad

        print("Total: ", t2-t1)
        print("3,4", t4 - t3)
        print("4,5", t5 - t4)
        print("5,6", t6 - t5)
        print("6,7", t7 - t6)
        print("7,8", t8 - t7)
        print("8,9", t9 - t8)
        print("9,10", t10 - t9)
        print("6-9", t9-t6)
        '''

        t2 = time.time()


        padded_sub_token_tensor = output
        return padded_sub_token_tensor


        # Backup
        '''
        first_sequence = first_sequence[0:num_tokens[0]]  # remove padding
        entity_tokens = np.array(self.entity_tokens)
        # lookup to text
        sequence_text = [self.entity_tokens_str[int(x)] for x in first_sequence[1:-1].cpu().detach().numpy()]
        # split into sub-tokens with BPE vocab:
        sub_token_sequence_str = [self.entity_token_vocab[x] for x in sequence_text]
        #[[entity_sub_token_ids[y] for y in x] for x in sub_token_sequence]
        entity_sub_token_sequence = [self.entity_sub_token_ids[y] for x in sub_token_sequence_str for y in x]
        first_entity_sub_tokens = torch.tensor(entity_sub_token_sequence)
        first_entity_emb = sub_token_entity_emb(first_entity_sub_tokens)
        tst = 0
        '''