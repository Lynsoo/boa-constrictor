import torch
import torch.nn as nn
import numpy as np

def BoaConstrictor_MinGRU(d_model=256, num_layers=2, vocab_size=256, device="cuda"):
    """Construct a MinGRU-based backbone compatible with BOA."""
    IS_CUDA = torch.cuda.is_available() and device == "cuda"
    device = "cuda" if IS_CUDA else "cpu"

    class MinGRUBlock(nn.Module):
        """Single GRU block with feedforward and layernorm."""
        def __init__(self, d_model: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.gru = nn.GRU(d_model, d_model, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4*d_model),
                nn.GELU(),
                nn.Linear(4*d_model, d_model)
            )
        def forward(self, x, hidden=None):
            y = self.ln1(x)
            y, hidden = self.gru(y, hidden)
            y = self.ln2(y)
            y = self.ff(y)
            return x + y, hidden

        def init_cache(self, batch_size: int, device):
            # GRU hidden state initialization for streaming
            return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)

        def step(self, x, hidden):
            y = self.ln1(x.unsqueeze(1))  
            y, hidden = self.gru(y, hidden)
            y = self.ln2(y)
            y = self.ff(y)
            return x + y.squeeze(1), hidden 

    class MinGRUBytePredictor(nn.Module):
        """Predict next byte using stacked MinGRU blocks."""
        def __init__(self, d_model=256, num_layers=2, vocab_size=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([MinGRUBlock(d_model) for _ in range(num_layers)])
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size)
            )

        def forward(self, x, hidden_states=None):
            h = self.embedding(x) 
            new_hidden = []
            if hidden_states is None:
                hidden_states = [None] * len(self.blocks)

            for blk, hidden in zip(self.blocks, hidden_states):
                h, new_h = blk(h, hidden)
                new_hidden.append(new_h)
            return self.head(h), new_hidden

        # Streaming / step-by-step inference
        @torch.inference_mode()
        def init_stream(self, batch_size=1, device=None):
            device = device or next(self.parameters()).device
            return [blk.init_cache(batch_size, device) for blk in self.blocks]

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, hidden_states):
            x = self.embedding(byte_t) 
            new_hidden = []
            h = x
            for blk, hidden in zip(self.blocks, hidden_states):
                h, new_h = blk.step(h, hidden)
                new_hidden.append(new_h)
            logits_next = self.head(h) 
            return logits_next  

    # Instantiate model
    model = MinGRUBytePredictor(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
    return model

def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    block = seq_len * batch_size
    return (n_bytes // block) * block

def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    train_bytes = buf[i0:i1].tobytes()
    val_bytes   = buf[i1:i2].tobytes()
    test_bytes  = buf[i2:i2+n_test].tobytes()

    return train_bytes, val_bytes, test_bytes

class ByteDataloader:
    """ Simple dataloader that yields batches of bytes. """
    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0
        self.device = device
    def __len__(self):
        """ Returns the total number of batches in the dataset. """
        return len(self.data_bytes) // (self.seq_len * self.batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos + self.seq_len * self.batch_size > len(self.data_bytes):
            self.pos = 0  # reset for simplicity
            raise StopIteration
        
        batch_indices = np.arange(self.pos, self.pos + self.seq_len * self.batch_size)
        batch_indices = batch_indices.reshape(self.batch_size, self.seq_len)
        self.pos += self.seq_len * self.batch_size
        
        batch = self.data_bytes[batch_indices]
        return torch.tensor(batch, dtype=torch.long).to(self.device)
    
