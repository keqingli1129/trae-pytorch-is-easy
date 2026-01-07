import torch

# A minimal tokenizer for our example
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            '<pad>': 0, 'The': 1, 'quick': 2, 'brown': 3, 'fox': 4,
            'jumps': 5, 'over': 6, 'lazy': 7, 'dog': 8,
            '<|user|>': 9, '<|assistant|>': 10, '<|end|>': 11,
            'What': 12, 'is': 13, 'a': 14, '?': 15
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        # A simple split-based tokenizer, robust to punctuation
        tokens = text.replace('?', ' ?').split()
        return [self.vocab[t] for t in tokens]

    def decode(self, tensor):
        return " ".join([self.inv_vocab[i] for i in tensor.tolist()])