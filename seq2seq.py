# nmt_seq2seq.py
# Urdu → Roman Urdu NMT (2-layer Encoder, 4-layer Decoder, 20 units)
# Dr. Usama guideline implementation

import torch
import torch.nn as nn
import torch.optim as optim
import random

# -------------------------
# Encoder: 2-layer LSTM
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=20, n_layers=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True)
    
    def forward(self, src):
        embedded = self.embedding(src)  # [B, Tx, emb]
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)

# -------------------------
# Decoder: 4-layer LSTM
# -------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=20, n_layers=4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
    
    def forward(self, tgt, hidden, cell):
        embedded = self.embedding(tgt)   # [B, Ty, emb]
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.fc_out(outputs)    # [B, Ty, vocab]
        return logits, (hidden, cell)

# -------------------------
# Seq2Seq Model
# -------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features

        # Encode
        _, (hidden, cell) = self.encoder(src)

        # Initialize decoder states:
        # Take encoder last layer, repeat to match decoder's 4 layers
        hidden = hidden[-1].unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)
        cell   = cell[-1].unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)

        # Prepare outputs
        outputs = torch.zeros(batch_size, tgt_len-1, vocab_size).to(self.device)

        # First input = <sos>
        input_tok = tgt[:,0]

        for t in range(1, tgt_len):
            embedded = self.decoder.embedding(input_tok).unsqueeze(1)  # [B,1,emb]
            out, (hidden, cell) = self.decoder.rnn(embedded, (hidden, cell))
            logits = self.decoder.fc_out(out.squeeze(1))  # [B,vocab]
            outputs[:,t-1,:] = logits

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = tgt[:,t] if teacher_force else top1

        return outputs

# -------------------------
# Example Training Loop
# -------------------------
def train_example():
    # Toy vocab (replace with SentencePiece or your dataset vocab)
    vocab_size = 50
    pad_idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = Encoder(vocab_size, emb_dim=128, hid_dim=20, n_layers=2, pad_idx=pad_idx)
    dec = Decoder(vocab_size, emb_dim=128, hid_dim=20, n_layers=4, pad_idx=pad_idx)
    model = Seq2Seq(enc, dec, pad_idx, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Dummy data: (batch=2, seq_len=6)
    src_batch = torch.randint(2, vocab_size, (2,6)).to(device)
    tgt_batch = torch.randint(2, vocab_size, (2,6)).to(device)
    tgt_batch[:,0] = 1  # <sos>

    for epoch in range(1, 6):
        model.train()
        optimizer.zero_grad()
        logits = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)
        # logits: [B, Ty-1, vocab], targets: [B, Ty-1]
        loss = criterion(logits.reshape(-1, vocab_size), tgt_batch[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

if __name__ == "__main__":
    train_example()
