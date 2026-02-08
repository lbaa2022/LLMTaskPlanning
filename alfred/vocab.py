"""
Minimal Vocab class for ALFRED preprocessing.
"""

class Vocab:
    """Simple vocabulary class for word-to-index mapping."""

    def __init__(self, initial_tokens=None):
        self.word2idx = {}
        self.idx2word = {}
        self.next_idx = 0

        if initial_tokens:
            for token in initial_tokens:
                self.word2index(token, train=True)

    def word2index(self, words, train=True):
        """Convert word(s) to index/indices."""
        if isinstance(words, str):
            return self._get_or_add_word(words, train)
        else:
            return [self._get_or_add_word(w, train) for w in words]

    def _get_or_add_word(self, word, train):
        """Get index for a word, optionally adding it if training."""
        word = word.strip().lower()
        if word in self.word2idx:
            return self.word2idx[word]
        elif train:
            idx = self.next_idx
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.next_idx += 1
            return idx
        else:
            return self.word2idx.get('<<pad>>', 0)

    def index2word(self, idx):
        """Convert index to word."""
        return self.idx2word.get(idx, '<<unk>>')

    def __len__(self):
        return self.next_idx
