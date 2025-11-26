import regex
from collections import Counter
from typing import List, Dict, Tuple


# hint: https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution
class BPE:

    def __init__(self):
        self.special_tokens: List[str] = ['<|endoftoken|>']
        self.vocab: Dict[int, bytes] = {}
        self.reverse_vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _pretokenize(self, text: str) -> List[str]:
        """Pretokenize text into words and symbols"""
        pretokens = []
        pre_pretokens = []
        if self.special_tokens:
            regex_split_pattern = "|".join(map(lambda x: regex.escape(x), self.special_tokens))
            pre_pretokens = regex.split(regex_split_pattern, text)
            pre_pretokens = [pre_pretoken for pre_pretoken in pre_pretokens if pre_pretoken]
        else:
            pre_pretokens = [text]
        for pre_pretoken in pre_pretokens:
            tmp_pretokens = regex.findall(self.PAT, pre_pretoken)
            pretokens += tmp_pretokens
        return pretokens
    
    def _tokenize(self, pretoken: str) -> Tuple[bytes]:
        """Tokenize pretoken into tokens"""
        pretoken_bytes = pretoken.encode('utf-8')
        tokens = []
        i = 0
        n = len(pretoken_bytes)

        while i < n:
            max_len = min(20, n - i)
            found = False
            for l in range(max_len, 0, -1):
                if pretoken_bytes[i:i + l] in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[pretoken_bytes[i:i + l]])
                    i += l
                    found = True
                    break
            if not found:
                raise ValueError(f"Byte sequence {pretoken_bytes[i]} not found in vocab")
        return tuple(tokens)
    
    def _get_pairs(self, pretoken_idx):
        pairs = []
        for i in range(len(pretoken_idx) - 1):
            pairs.append((pretoken_idx[i], pretoken_idx[i + 1]))
        return pairs
    
    def _count_pairs(self, pretoken_idx_freq):
        pair_counts = Counter()
        for pretoken_idx, freq in pretoken_idx_freq.items():
            pretoken_idx_pairs = self._get_pairs(pretoken_idx)
            for pair in pretoken_idx_pairs:
                pair_counts[pair] += freq
        return pair_counts
    
    def _merge_pair(self, pretoken_idx, max_idx_pair, next_vocab_id):
        new_pretoken_idx = []
        i = 0
        while i < len(pretoken_idx):
            if i < len(pretoken_idx) - 1 and (pretoken_idx[i], pretoken_idx[i + 1]) == max_idx_pair:
                new_pretoken_idx.append(next_vocab_id)
                i += 2
            else:
                new_pretoken_idx.append(pretoken_idx[i])
                i += 1
        return tuple(new_pretoken_idx)

    def train(self, input_path: str, vocab_size: int = 270, special_tokens = ['<|endoftoken|>']) \
        -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        # step 0: init
        self.special_tokens = special_tokens
        self.vocab = {idx: special_token.encode("utf-8") for idx, special_token in enumerate(self.special_tokens)}
        self.vocab.update({idx + len(self.special_tokens): bytes([idx]) for idx in range(256)})
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        next_vocab_id = len(self.vocab.keys())
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.read()
        # print(self.vocab, next_vocab_id)
        # step 1: pretokenize
        pretokens_str = self._pretokenize(data)
        pretoken_idx_freq = Counter()
        # step 2: tokenize
        for pretoken_str in pretokens_str:
            pretoken_idx_freq[self._tokenize(pretoken_str)] += 1

        # step 3: count byte pairs
        num_merges = vocab_size - len(self.vocab.keys())
        for idx in range(num_merges):
            pair_counts = self._count_pairs(pretoken_idx_freq)
            if not pair_counts:
                break
            max_idx_pair = max(pair_counts.items(), key=lambda x: (x[1], (self.vocab[x[0][0]], self.vocab[x[0][1]])))[0]
            # step 4: merge
            # print(self.vocab)
            new_pretoken_idx_freq = Counter()
            for pretoken_idx, freq in pretoken_idx_freq.items():
                new_pretoken_idx = self._merge_pair(pretoken_idx, max_idx_pair, next_vocab_id)
                new_pretoken_idx_freq[new_pretoken_idx] += freq
            pretoken_idx_freq = new_pretoken_idx_freq
            self.merges.append((self.vocab[max_idx_pair[0]], self.vocab[max_idx_pair[1]]))
            # step 5: update vocab
            new_token = self.vocab[max_idx_pair[0]] + self.vocab[max_idx_pair[1]]
            self.vocab[next_vocab_id] = new_token
            self.reverse_vocab[new_token] = next_vocab_id
            next_vocab_id += 1
        return self.vocab, self.merges
