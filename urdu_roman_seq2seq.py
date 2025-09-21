#!/usr/bin/env python3
"""
Complete Seq2Seq Model for Urdu to Roman Urdu Translation
Using your cleaned dataset with attention mechanism
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter, defaultdict
import re
import json
import pickle
from typing import List, Dict, Tuple, Set
import random
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class UrduRomanDataset(Dataset):
    """Dataset class for Urdu-Roman Urdu pairs"""
    
    def __init__(self, pairs, urdu_vocab, roman_vocab, max_length=50):
        self.pairs = pairs
        self.urdu_vocab = urdu_vocab
        self.roman_vocab = roman_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        urdu_text, roman_text = self.pairs[idx]
        
        # Tokenize and convert to IDs
        urdu_ids = self.text_to_ids(urdu_text, self.urdu_vocab)
        roman_ids = self.text_to_ids(roman_text, self.roman_vocab)
        
        return torch.tensor(urdu_ids, dtype=torch.long), torch.tensor(roman_ids, dtype=torch.long)
    
    def text_to_ids(self, text, vocab):
        """Convert text to token IDs"""
        tokens = text.split()
        ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Add SOS and EOS tokens
        ids = [vocab['<SOS>']] + ids + [vocab['<EOS>']]
        
        # Pad or truncate to max_length
        if len(ids) > self.max_length:
            ids = ids[:self.max_length-1] + [vocab['<EOS>']]
        else:
            ids.extend([vocab['<PAD>']] * (self.max_length - len(ids)))
        
        return ids

class Attention(nn.Module):
    """Attention mechanism for seq2seq model"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden for each encoder output
        decoder_hidden_repeated = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate decoder hidden with encoder outputs
        energy = torch.tanh(self.attention(torch.cat((decoder_hidden_repeated, encoder_outputs), dim=2)))
        
        # Calculate attention weights
        attention_weights = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Calculate context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights

class Encoder(nn.Module):
    """Bidirectional LSTM Encoder with 2 layers"""
    
    def __init__(self, vocab_size, emb_dim=128, hid_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Combine forward and backward hidden states
        # hidden shape: [n_layers * 2, batch_size, hid_dim]
        hidden = hidden.view(self.n_layers, 2, hidden.size(1), hidden.size(2))
        hidden = hidden.mean(dim=1)  # Average forward and backward
        
        cell = cell.view(self.n_layers, 2, cell.size(1), cell.size(2))
        cell = cell.mean(dim=1)
        
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    """LSTM Decoder with 4 layers and attention"""
    
    def __init__(self, vocab_size, emb_dim=128, hid_dim=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers=n_layers, 
                          batch_first=True, dropout=dropout)
        self.attention = Attention(hid_dim)
        self.fc_out = nn.Linear(hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        
    def forward(self, tgt, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(tgt))
        
        # Get attention context
        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        
        # Concatenate embedded input with context vector
        rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)
        
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Concatenate output with context vector for final prediction
        output_with_context = torch.cat((outputs, context_vector.unsqueeze(1)), dim=2)
        logits = self.fc_out(output_with_context)
        
        return logits, (hidden, cell), attention_weights

class Seq2SeqModel(nn.Module):
    """Complete Seq2Seq model with attention"""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc_out.out_features
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Initialize decoder input
        decoder_input = tgt[:, 0:1]  # First token (SOS)
        
        # Prepare outputs
        outputs = torch.zeros(batch_size, tgt_len-1, vocab_size).to(self.device)
        attention_weights = torch.zeros(batch_size, tgt_len-1, src.size(1)).to(self.device)
        
        for t in range(1, tgt_len):
            # Decode
            logits, (hidden, cell), attn_weights = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            
            outputs[:, t-1, :] = logits.squeeze(1)
            attention_weights[:, t-1, :] = attn_weights.squeeze(1)
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(2)
            decoder_input = tgt[:, t:t+1] if teacher_force else top1
            
        return outputs, attention_weights

class VocabularyBuilder:
    """Build vocabulary from text data"""
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_counts.update(words)
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        # Build vocabulary
        idx = 0
        for token in special_tokens:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1
        
        # Add words that meet frequency threshold
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Words with freq >= {self.min_freq}: {len(self.word2idx) - len(special_tokens)}")
        
        return self.word2idx, self.idx2word

def load_data(file_path):
    """Load Urdu-Roman Urdu pairs from file"""
    print(f"Loading data from {file_path}...")
    
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                urdu, roman = line.split('\t', 1)
                pairs.append((urdu.strip(), roman.strip()))
    
    print(f"Loaded {len(pairs)} pairs")
    return pairs

def preprocess_data(pairs, max_pairs=10000):
    """Preprocess and filter data"""
    print("Preprocessing data...")
    
    # Filter pairs by length and content
    filtered_pairs = []
    for urdu, roman in pairs:
        # Skip if too long or contains non-printable characters
        if (len(urdu.split()) <= 30 and len(roman.split()) <= 30 and
            len(urdu) > 5 and len(roman) > 5):
            filtered_pairs.append((urdu, roman))
    
    # Limit dataset size for faster training
    if len(filtered_pairs) > max_pairs:
        filtered_pairs = filtered_pairs[:max_pairs]
    
    print(f"Filtered to {len(filtered_pairs)} pairs")
    return filtered_pairs

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the seq2seq model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(src, tgt, teacher_forcing_ratio=0.5)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                outputs, _ = model(src, tgt, teacher_forcing_ratio=0.0)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print('-' * 50)
    
    return train_losses, val_losses

def translate(model, text, urdu_vocab, roman_vocab, max_length=50):
    """Translate a single sentence"""
    device = next(model.parameters()).device
    
    # Convert text to IDs
    tokens = text.split()
    src_ids = [urdu_vocab.get(token, urdu_vocab['<UNK>']) for token in tokens]
    src_ids = [urdu_vocab['<SOS>']] + src_ids + [urdu_vocab['<EOS>']]
    
    # Pad to max_length
    if len(src_ids) > max_length:
        src_ids = src_ids[:max_length-1] + [urdu_vocab['<EOS>']]
    else:
        src_ids.extend([urdu_vocab['<PAD>']] * (max_length - len(src_ids)))
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Encode
    model.eval()
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
        
        # Initialize decoder
        decoder_input = torch.tensor([[roman_vocab['<SOS>']]], dtype=torch.long).to(device)
        translated_ids = []
        
        for _ in range(max_length):
            logits, (hidden, cell), _ = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            top1 = logits.argmax(2)
            translated_ids.append(top1.item())
            
            if top1.item() == roman_vocab['<EOS>']:
                break
                
            decoder_input = top1
    
    # Convert IDs back to text
    translated_tokens = [roman_vocab.get(idx, '<UNK>') for idx in translated_ids]
    translated_text = ' '.join([token for token in translated_tokens if token not in ['<SOS>', '<EOS>', '<PAD>']])
    
    return translated_text

def main():
    """Main training function"""
    print("Starting Urdu to Roman Urdu Translation Training")
    print("=" * 60)
    
    # Load and preprocess data
    pairs = load_data('final_cleaned_urdu_roman_urdu_pairs.txt')
    pairs = preprocess_data(pairs, max_pairs=5000)  # Use subset for faster training
    
    # Split data
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=42)
    
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    
    # Build vocabularies
    urdu_texts = [pair[0] for pair in pairs]
    roman_texts = [pair[1] for pair in pairs]
    
    urdu_vocab_builder = VocabularyBuilder(min_freq=2)
    roman_vocab_builder = VocabularyBuilder(min_freq=2)
    
    urdu_vocab, urdu_idx2word = urdu_vocab_builder.build_vocab(urdu_texts)
    roman_vocab, roman_idx2word = roman_vocab_builder.build_vocab(roman_texts)
    
    # Create datasets
    train_dataset = UrduRomanDataset(train_pairs, urdu_vocab, roman_vocab)
    val_dataset = UrduRomanDataset(val_pairs, urdu_vocab, roman_vocab)
    test_dataset = UrduRomanDataset(test_pairs, urdu_vocab, roman_vocab)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = Encoder(len(urdu_vocab), emb_dim=128, hid_dim=64, n_layers=2)
    decoder = Decoder(len(roman_vocab), emb_dim=128, hid_dim=64, n_layers=4)
    model = Seq2SeqModel(encoder, decoder, device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                          num_epochs=10, learning_rate=0.001)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()
    
    # Test translation
    print("\nTesting translation on sample sentences:")
    print("=" * 50)
    
    for i in range(5):
        urdu_text, expected_roman = test_pairs[i]
        translated = translate(model, urdu_text, urdu_vocab, roman_vocab)
        
        print(f"\nExample {i+1}:")
        print(f"Urdu: {urdu_text}")
        print(f"Expected Roman: {expected_roman}")
        print(f"Translated: {translated}")
    
    # Save model and vocabularies
    torch.save(model.state_dict(), 'urdu_roman_seq2seq_model.pth')
    
    with open('urdu_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(urdu_vocab, f, ensure_ascii=False, indent=2)
    
    with open('roman_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(roman_vocab, f, ensure_ascii=False, indent=2)
    
    print("\nModel and vocabularies saved!")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

