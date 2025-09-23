#!/usr/bin/env python3

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import random
from sklearn.model_selection import train_test_split
import math
import time

"""
Unigram Tokenizer-based Seq2Seq Model for Urdu to Roman Urdu Translation
Using Unigram tokenization (better than BPE for many languages)
"""

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class UnigramTokenizer:
    """
    Unigram tokenizer implementation based on SentencePiece algorithm
    Better than BPE for many languages including Urdu
    """
    
    def __init__(self, vocab_size=8000, character_coverage=0.9995):
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.vocab = {}
        self.scores = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
    def _get_word_freqs(self, texts):
        """Get word frequencies from texts"""
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            word_freqs.update(words)
        return word_freqs
    
    def _initialize_vocab(self, texts):
        """Initialize vocabulary with characters and frequent substrings"""
        print("Initializing vocabulary...")
        
        # Get all characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Get word frequencies
        word_freqs = self._get_word_freqs(texts)
        
        # Initialize with characters
        vocab = set(chars)
        
        # Add frequent substrings (2-6 characters) - longer for better subwords
        for length in range(2, 7):
            substring_freqs = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - length + 1):
                    substring = word[i:i+length]
                    substring_freqs[substring] += freq
            
            # Add top frequent substrings with higher threshold
            for substring, freq in substring_freqs.most_common(200):
                if freq > 5 and len(substring) >= 2:  # Lower threshold, longer substrings
                    vocab.add(substring)
        
        # Add common words as complete tokens
        for word, freq in word_freqs.most_common(100):
            if freq > 20 and len(word) >= 2:  # Add frequent words
                vocab.add(word)
        
        # Add special tokens
        vocab.update(self.special_tokens)
        
        print(f"Initial vocabulary size: {len(vocab)}")
        return vocab
    
    def _compute_likelihood(self, text, vocab):
        """Compute likelihood of text given vocabulary"""
        words = text.split()
        total_log_prob = 0
        
        for word in words:
            word_log_prob = self._get_word_log_prob(word, vocab)
            total_log_prob += word_log_prob
        
        return total_log_prob
    
    def _get_word_log_prob(self, word, vocab):
        """Get log probability of a word given vocabulary"""
        if word in vocab:
            token_score = self.scores.get(word, 1.0)
            return math.log(max(token_score, 1e-10))
        
        # Try to segment the word
        segments = self._segment_word(word, vocab)
        if segments:
            log_prob = sum(math.log(max(self.scores.get(seg, 1.0), 1e-10)) for seg in segments)
            return log_prob
        
        # If can't segment, return log of UNK probability
        unk_score = self.scores.get('<UNK>', 0.001)
        return math.log(max(unk_score, 1e-10))
    
    def _segment_word(self, word, vocab):
        """Segment word into vocabulary tokens"""
        if not word:
            return []
        
        # Dynamic programming to find best segmentation
        n = len(word)
        dp = [-float('inf')] * (n + 1)
        dp[0] = 0
        parent = [-1] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i):
                substring = word[j:i]
                if substring in vocab:
                    token_score = self.scores.get(substring, 1.0)
                    # Avoid log(0) or log(negative) by using a small epsilon
                    score = math.log(max(token_score, 1e-10))
                    if dp[j] + score > dp[i]:
                        dp[i] = dp[j] + score
                        parent[i] = j
        
        # Reconstruct segmentation
        if dp[n] == -float('inf'):
            return ['<UNK>']
        
        segments = []
        i = n
        while i > 0:
            j = parent[i]
            segments.append(word[j:i])
            i = j
        
        return segments[::-1]
    
    def _remove_tokens(self, vocab, texts, num_to_remove):
        """Remove least important tokens"""
        print(f"Removing {num_to_remove} tokens...")
        
        # Compute impact of removing each token
        token_impact = {}
        
        for i, token in enumerate(vocab):
            if i % 100 == 0:
                print(f"Computing token impact: {i}/{len(vocab)}")
            
            if token in self.special_tokens:
                continue
                
            # Compute likelihood without this token
            vocab_without_token = vocab - {token}
            likelihood_without = sum(self._compute_likelihood(text, vocab_without_token) for text in texts)
            
            # Compute likelihood with this token
            likelihood_with = sum(self._compute_likelihood(text, vocab) for text in texts)
            
            token_impact[token] = likelihood_with - likelihood_without
        
        # Remove tokens with least impact
        tokens_to_remove = sorted(token_impact.items(), key=lambda x: x[1])[:num_to_remove]
        
        for token, _ in tokens_to_remove:
            vocab.remove(token)
        
        return vocab
    
    def _update_scores(self, vocab, texts):
        """Update token scores using EM algorithm"""
        print("Updating token scores...")
        
        # Count token occurrences
        token_counts = Counter()
        total_tokens = 0
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Counting tokens: {i}/{len(texts)}")
            
            words = text.split()
            for word in words:
                segments = self._segment_word(word, vocab)
                for segment in segments:
                    token_counts[segment] += 1
                    total_tokens += 1
        
        # Update scores (probabilities) - ensure no zero scores
        for token in vocab:
            count = token_counts.get(token, 0)
            if total_tokens > 0:
                self.scores[token] = max(count / total_tokens, 1e-10)
            else:
                self.scores[token] = 1e-10
        
        # Ensure special tokens have reasonable scores
        for token in self.special_tokens:
            if token not in self.scores or self.scores[token] <= 0:
                self.scores[token] = 1e-10
        
        print(f"Total tokens counted: {total_tokens}")
        print(f"Non-zero score tokens: {sum(1 for score in self.scores.values() if score > 1e-10)}")
    
    def train(self, texts, num_iterations=10):
        """Train the unigram tokenizer"""
        print(f"Training Unigram tokenizer with {len(texts)} texts...")
        
        # Initialize vocabulary
        vocab = self._initialize_vocab(texts)
        
        # Initial scores - ensure no zero scores
        for token in vocab:
            self.scores[token] = 1.0
        
        # EM iterations
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Update scores
            self._update_scores(vocab, texts)
            
            # Remove tokens if vocabulary is too large
            if len(vocab) > self.vocab_size:
                num_to_remove = len(vocab) - self.vocab_size
                vocab = self._remove_tokens(vocab, texts, num_to_remove)
            
            print(f"Vocabulary size: {len(vocab)}")
        
        # Final vocabulary
        self.vocab = vocab
        self._update_scores(vocab, texts)
        
        # Create token to id mapping
        self.token_to_id = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        return self.token_to_id, self.id_to_token
    
    def encode(self, text):
        """Encode text to token IDs"""
        words = text.split()
        token_ids = []
        
        for word in words:
            segments = self._segment_word(word, self.vocab)
            for segment in segments:
                token_ids.append(self.token_to_id.get(segment, self.token_to_id['<UNK>']))
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join(tokens)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'scores': self.scores,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size,
            'character_coverage': self.character_coverage
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab = data['vocab']
        self.scores = data['scores']
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        self.vocab_size = data['vocab_size']
        self.character_coverage = data['character_coverage']

class UnigramUrduRomanDataset(Dataset):
    """Dataset class for Urdu-Roman Urdu pairs with Unigram tokenization"""
    
    def __init__(self, pairs, urdu_tokenizer, roman_tokenizer, max_length=50):
        self.pairs = pairs
        self.urdu_tokenizer = urdu_tokenizer
        self.roman_tokenizer = roman_tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        urdu_text, roman_text = self.pairs[idx]
        
        # Tokenize using unigram tokenizers
        urdu_ids = self.urdu_tokenizer.encode(urdu_text)
        roman_ids = self.roman_tokenizer.encode(roman_text)
        
        # Add SOS and EOS tokens
        urdu_ids = [self.urdu_tokenizer.token_to_id['<SOS>']] + urdu_ids + [self.urdu_tokenizer.token_to_id['<EOS>']]
        roman_ids = [self.roman_tokenizer.token_to_id['<SOS>']] + roman_ids + [self.roman_tokenizer.token_to_id['<EOS>']]
        
        # Pad or truncate to max_length
        if len(urdu_ids) > self.max_length:
            urdu_ids = urdu_ids[:self.max_length-1] + [self.urdu_tokenizer.token_to_id['<EOS>']]
        else:
            urdu_ids.extend([self.urdu_tokenizer.token_to_id['<PAD>']] * (self.max_length - len(urdu_ids)))
        
        if len(roman_ids) > self.max_length:
            roman_ids = roman_ids[:self.max_length-1] + [self.roman_tokenizer.token_to_id['<EOS>']]
        else:
            roman_ids.extend([self.roman_tokenizer.token_to_id['<PAD>']] * (self.max_length - len(roman_ids)))
        
        return torch.tensor(urdu_ids, dtype=torch.long), torch.tensor(roman_ids, dtype=torch.long)

class Attention(nn.Module):
    """Attention mechanism for seq2seq model"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attention = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
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
        # RNN input: emb_dim + encoder_hidden_dim (bidirectional = hid_dim * 2)
        self.rnn = nn.LSTM(emb_dim + hid_dim * 2, hid_dim, num_layers=n_layers, 
                          batch_first=True, dropout=dropout)
        # Attention expects encoder_hidden_dim (bidirectional = hid_dim * 2) and decoder_hidden_dim
        self.attention = Attention(hid_dim * 2, hid_dim)
        self.fc_out = nn.Linear(hid_dim + hid_dim * 2, vocab_size)
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
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
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
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
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
        epoch_start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            batch_start_time = time.time()
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            
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
                batch_time = time.time() - batch_start_time
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, GPU Mem: {gpu_mem:.2f}GB, Time: {batch_time:.3f}s')
                else:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {batch_time:.3f}s')
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
                outputs, _ = model(src, tgt, teacher_forcing_ratio=0.0)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Epoch Time: {epoch_time:.2f}s')
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            print(f'  GPU Memory: {gpu_mem:.2f}GB')
        print('-' * 50)
    
    return train_losses, val_losses

def translate(model, text, urdu_tokenizer, roman_tokenizer, max_length=50):
    """Translate a single sentence"""
    device = next(model.parameters()).device
    
    # Convert text to IDs using unigram tokenizer
    src_ids = urdu_tokenizer.encode(text)
    src_ids = [urdu_tokenizer.token_to_id['<SOS>']] + src_ids + [urdu_tokenizer.token_to_id['<EOS>']]
    
    # Pad to max_length
    if len(src_ids) > max_length:
        src_ids = src_ids[:max_length-1] + [urdu_tokenizer.token_to_id['<EOS>']]
    else:
        src_ids.extend([urdu_tokenizer.token_to_id['<PAD>']] * (max_length - len(src_ids)))
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Encode
    model.eval()
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
        
        # Initialize decoder
        decoder_input = torch.tensor([[roman_tokenizer.token_to_id['<SOS>']]], dtype=torch.long).to(device)
        translated_ids = []
        
        for _ in range(max_length):
            logits, (hidden, cell), _ = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            top1 = logits.argmax(2)
            translated_ids.append(top1.item())
            
            if top1.item() == roman_tokenizer.token_to_id['<EOS>']:
                break
                
            decoder_input = top1
    
    # Convert IDs back to text using unigram tokenizer
    translated_text = roman_tokenizer.decode(translated_ids)
    
    # Clean up the output - remove special tokens and extra spaces
    translated_text = translated_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').replace('<UNK>', '')
    translated_text = ' '.join(translated_text.split())  # Remove extra spaces
    
    return translated_text

def get_ngrams(tokens, n):
    """Get n-grams from a list of tokens"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def calculate_bleu_score(reference, candidate, max_n=4):
    """
    Calculate BLEU score without NLTK
    BLEU = BP * exp(sum(w_n * log(p_n)))
    where BP is brevity penalty and p_n is n-gram precision
    """
    # Tokenize reference and candidate
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Calculate brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len)
    
    # Calculate precision for each n-gram
    precisions = []
    weights = [1.0 / max_n] * max_n  # Uniform weights
    
    for n in range(1, max_n + 1):
        if len(cand_tokens) < n:
            precisions.append(0.0)
            continue
            
        # Get n-grams
        cand_ngrams = get_ngrams(cand_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        # Calculate clipped precision
        matches = 0
        for ngram in cand_counts:
            matches += min(cand_counts[ngram], ref_counts.get(ngram, 0))
        
        precision = matches / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
        precisions.append(precision)
    
    # Calculate BLEU score
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
    bleu = bp * math.exp(log_precision)
    
    return bleu

def calculate_perplexity(model, data_loader, criterion, device):
    """
    Calculate perplexity (PPL) of the model on given data
    PPL = exp(cross_entropy_loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            
            # Forward pass without teacher forcing
            outputs, _ = model(src, tgt, teacher_forcing_ratio=0.0)
            
            # Calculate loss (excluding first token which is SOS)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Count non-padding tokens
            non_pad_mask = tgt[:, 1:] != 0
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def evaluate_model(model, test_loader, test_pairs, urdu_tokenizer, roman_tokenizer, device, num_samples=None):
    """
    Evaluate model with BLEU score and perplexity
    """
    print("Evaluating model...")
    print("=" * 50)
    
    # Calculate perplexity
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    ppl = calculate_perplexity(model, test_loader, criterion, device)
    print(f"Perplexity: {ppl:.2f}")
    
    # Calculate BLEU scores
    bleu_scores = []
    samples_to_eval = num_samples if num_samples else min(100, len(test_pairs))
    
    print(f"\nCalculating BLEU scores on {samples_to_eval} samples...")
    
    for i in range(samples_to_eval):
        urdu_text, expected_roman = test_pairs[i]
        translated = translate(model, urdu_text, urdu_tokenizer, roman_tokenizer)
        
        # Calculate BLEU score
        bleu = calculate_bleu_score(expected_roman, translated)
        bleu_scores.append(bleu)
        
        if i < 5:  # Show first 5 examples
            print(f"\nSample {i+1}:")
            print(f"Urdu: {urdu_text}")
            print(f"Expected: {expected_roman}")
            print(f"Translated: {translated}")
            print(f"BLEU: {bleu:.4f}")
    
    # Calculate average BLEU scores
    avg_bleu = np.mean(bleu_scores)
    bleu_1 = np.mean([calculate_bleu_score(test_pairs[i][1], translate(model, test_pairs[i][0], urdu_tokenizer, roman_tokenizer), max_n=1) for i in range(samples_to_eval)])
    bleu_2 = np.mean([calculate_bleu_score(test_pairs[i][1], translate(model, test_pairs[i][0], urdu_tokenizer, roman_tokenizer), max_n=2) for i in range(samples_to_eval)])
    bleu_3 = np.mean([calculate_bleu_score(test_pairs[i][1], translate(model, test_pairs[i][0], urdu_tokenizer, roman_tokenizer), max_n=3) for i in range(samples_to_eval)])
    bleu_4 = np.mean([calculate_bleu_score(test_pairs[i][1], translate(model, test_pairs[i][0], urdu_tokenizer, roman_tokenizer), max_n=4) for i in range(samples_to_eval)])
    
    
    
    return {
        'ppl': ppl,
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4,
        'avg_bleu': avg_bleu
    }

def run_experiments(train_loader, val_loader, urdu_token_to_id, roman_token_to_id, device):
    """Run the experiments and return the best model."""
    best_val_loss = float('inf')
    best_model = None
    best_experiment = None
    
    experiments = [
        ("Experiment 1", {
            "emb_src": 256, "emb_tgt": 256, "enc_hidden": 512, "dec_hidden": 512,
            "enc_layers": 2, "dec_layers": 2, "dropout": 0.3, "batch_size": 64, "lr": 1e-3, "epochs": 5
        }),
        ("Experiment 2", {
            "emb_src": 128, "emb_tgt": 128, "enc_hidden": 256, "dec_hidden": 256,
            "enc_layers": 4, "dec_layers": 4, "dropout": 0.3, "batch_size": 64, "lr": 5e-4, "epochs": 5
        }),
        ("Experiment 3", {
            "emb_src": 512, "emb_tgt": 512, "enc_hidden": 512, "dec_hidden": 512,
            "enc_layers": 4, "dec_layers": 4, "dropout": 0.5, "batch_size": 32, "lr": 1e-4, "epochs": 5
        }),
        ("Experiment 4", {
            "emb_src": 256, "emb_tgt": 256, "enc_hidden": 512, "dec_hidden": 512,
            "enc_layers": 4, "dec_layers": 4, "dropout": 0.1, "batch_size": 32, "lr": 1e-3, "epochs": 5
        }),
        ("Experiment 5", {
            "emb_src": 128, "emb_tgt": 128, "enc_hidden": 256, "dec_hidden": 256,
            "enc_layers": 2, "dec_layers": 2, "dropout": 0.5, "batch_size": 128, "lr": 5e-4, "epochs": 5
        }),
    ]
    
    # Loop through experiments
    for experiment_name, params in experiments:
        print(f"\nStarting {experiment_name}")
        print("=" * 50)
        
        # Initialize the model with experiment parameters
        encoder = Encoder(len(urdu_token_to_id), emb_dim=params["emb_src"], hid_dim=params["enc_hidden"], n_layers=params["enc_layers"], dropout=params["dropout"])
        decoder = Decoder(len(roman_token_to_id), emb_dim=params["emb_tgt"], hid_dim=params["dec_hidden"], n_layers=params["dec_layers"], dropout=params["dropout"])
        model = Seq2SeqModel(encoder, decoder, device)
        
        # Move model to device
        model = model.to(device)
        
        # Train model for current experiment
        print(f"Training {experiment_name}...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=params["epochs"], learning_rate=params["lr"])
        
        # Evaluate model on validation set to determine if it's the best
        avg_val_loss = min(val_losses)  # Take the best (lowest) validation loss
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            best_experiment = experiment_name, params
            
            # Save the best model
            torch.save(best_model.state_dict(), f'best_model_{experiment_name}.pth')
            print(f"New best model found and saved from {experiment_name} with validation loss: {best_val_loss:.4f}")
    
    # After all experiments, return the best model
    return best_model, best_experiment, best_val_loss
def check_gpu_utilization():
    """Check and display GPU utilization information"""
    if torch.cuda.is_available():
        print("GPU Information:")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        print(f"  Current Device: {torch.cuda.current_device()}")
        print(f"  Device Name: {torch.cuda.get_device_name()}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print("=" * 50)
    else:
        print("CUDA not available - running on CPU")
        print("=" * 50)

def load_tokenizers_and_model(
    urdu_tokenizer_path='unigram_urdu_tokenizer.pkl',
    roman_tokenizer_path='unigram_roman_tokenizer.pkl',
    model_path='unigram_urdu_roman_seq2seq_model.pth',
    emb_dim=128, hid_dim=64, n_layers=2, device=None
):
    """Load tokenizers and trained model for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizers
    urdu_tokenizer = UnigramTokenizer()
    urdu_tokenizer.load(urdu_tokenizer_path)
    roman_tokenizer = UnigramTokenizer()
    roman_tokenizer.load(roman_tokenizer_path)
    # Build model
    encoder = Encoder(len(urdu_tokenizer.token_to_id), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers)
    decoder = Decoder(len(roman_tokenizer.token_to_id), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers)
    model = Seq2SeqModel(encoder, decoder, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, urdu_tokenizer, roman_tokenizer, device

def streamlit_translate_urdu_to_roman(
    urdu_text,
    model,
    urdu_tokenizer,
    roman_tokenizer,
    device,
    max_length=50
):
    """Translate Urdu text to Roman Urdu for Streamlit app."""
    # Limit input length for safety
    urdu_text = urdu_text[:400]
    # Use the same translate logic as before
    return translate(model, urdu_text, urdu_tokenizer, roman_tokenizer, max_length=max_length)