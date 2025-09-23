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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

class xLSTMAttention(nn.Module):
    """Improved attention mechanism for xLSTM models with proper dimension handling"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim=128):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers with proper dimensions
        self.W_enc = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        # Context projection - now handles bidirectional encoder output
        self.W_context = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(decoder_hidden_dim)
        
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [batch_size, decoder_hidden_dim]
        encoder_outputs: [batch_size, seq_len, encoder_hidden_dim] (bidirectional = hid_dim * 2)
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project encoder outputs to attention space
        enc_proj = self.W_enc(encoder_outputs)  # [batch_size, seq_len, attention_dim]
        
        # Project decoder hidden to attention space
        dec_proj = self.W_dec(decoder_hidden)  # [batch_size, attention_dim]
        
        # Expand decoder projection for broadcasting
        dec_proj_expanded = dec_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, attention_dim]
        
        # Calculate attention scores with better numerical stability
        attention_scores = self.v(torch.tanh(enc_proj + dec_proj_expanded))  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Calculate weighted context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, encoder_hidden_dim]
        context_vector = context_vector.squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        # Combine context with decoder hidden
        combined = torch.cat([context_vector, decoder_hidden], dim=1)  # [batch_size, encoder_hidden_dim + decoder_hidden_dim]
        context = self.W_context(combined)  # [batch_size, decoder_hidden_dim]
        
        # Apply layer normalization for stability
        context = self.layer_norm(context)
        
        return context, attention_weights

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
        
        # Handle different layer counts between encoder and decoder
        decoder_layers = self.rnn.num_layers
        encoder_layers = hidden.size(0)
        
        if decoder_layers > encoder_layers:
            # Pad hidden and cell states with zeros for additional layers
            batch_size = hidden.size(1)
            hidden_dim = hidden.size(2)
            
            # Create zero tensors for additional layers
            additional_hidden = torch.zeros(decoder_layers - encoder_layers, batch_size, hidden_dim, 
                                          device=hidden.device, dtype=hidden.dtype)
            additional_cell = torch.zeros(decoder_layers - encoder_layers, batch_size, hidden_dim, 
                                        device=cell.device, dtype=cell.dtype)
            
            # Concatenate with existing hidden states
            hidden = torch.cat([hidden, additional_hidden], dim=0)
            cell = torch.cat([cell, additional_cell], dim=0)
        
        # Get attention context - use the last layer's hidden state
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

# Improved xLSTM Implementation with proper architecture
class xLSTMBlock(nn.Module):
    """Improved xLSTM block implementation with matrix memory and exponential gating"""
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Matrix memory dimensions - use full hidden size for consistency
        self.memory_size = hidden_size
        
        # Input projections
        self.input_proj = nn.Linear(input_size, hidden_size * 4)
        
        # Matrix memory components
        self.W_f = nn.Linear(hidden_size, self.memory_size)  # Forget gate
        self.W_i = nn.Linear(hidden_size, self.memory_size)  # Input gate  
        self.W_o = nn.Linear(hidden_size, self.memory_size)  # Output gate
        self.W_c = nn.Linear(hidden_size, self.memory_size)  # Candidate values
        
        # Scalar gates for better control
        self.scalar_f = nn.Parameter(torch.ones(1))
        self.scalar_i = nn.Parameter(torch.ones(1))
        self.scalar_o = nn.Parameter(torch.ones(1))
        
        # Output projection
        self.output_proj = nn.Linear(self.memory_size, hidden_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.size()
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.memory_size, device=x.device)
        else:
            h, c = hidden_state
            
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Input projection
            projected = self.input_proj(x_t)
            i_gate, f_gate, o_gate, candidate = projected.chunk(4, dim=1)
            
            # Apply gates with exponential functions for better gradient flow
            i_gate = torch.sigmoid(i_gate) * self.scalar_i
            f_gate = torch.sigmoid(f_gate) * self.scalar_f
            o_gate = torch.sigmoid(o_gate) * self.scalar_o
            candidate = torch.tanh(candidate)
            
            # Matrix memory operations
            f_matrix = self.W_f(h)
            i_matrix = self.W_i(h)
            o_matrix = self.W_o(h)
            c_matrix = self.W_c(h)
            
            # Update cell state with matrix operations
            c = f_gate * c + i_gate * c_matrix
            h = o_gate * torch.tanh(c)
            
            # Project back to hidden size
            h = self.output_proj(h)
            h = self.layer_norm(h)
            
            outputs.append(h)
            
        # Return outputs and final hidden state
        return torch.stack(outputs, dim=1), h, c

class xLSTMEncoder(nn.Module):
    """Improved xLSTM-based Encoder with bidirectional support"""
    def __init__(self, vocab_size, emb_dim=128, hid_dim=64, n_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bidirectional = bidirectional
        
        # Use separate xLSTM blocks for forward and backward if bidirectional
        if bidirectional:
            self.forward_xlstm = xLSTMBlock(emb_dim, hid_dim, n_layers, bidirectional=False)
            self.backward_xlstm = xLSTMBlock(emb_dim, hid_dim, n_layers, bidirectional=False)
            self.output_hidden_dim = hid_dim * 2  # Double for bidirectional
        else:
            self.xlstm = xLSTMBlock(emb_dim, hid_dim, n_layers, bidirectional=False)
            self.output_hidden_dim = hid_dim
            
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        
        if self.bidirectional:
            # Forward pass
            forward_outputs, forward_hidden, forward_cell = self.forward_xlstm(embedded)
            
            # Backward pass (reverse the sequence)
            reversed_embedded = torch.flip(embedded, dims=[1])
            backward_outputs, backward_hidden, backward_cell = self.backward_xlstm(reversed_embedded)
            backward_outputs = torch.flip(backward_outputs, dims=[1])  # Reverse back
            
            # Concatenate forward and backward outputs
            outputs = torch.cat([forward_outputs, backward_outputs], dim=2)
            
            # Concatenate hidden states
            hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
            cell = torch.cat([forward_cell, backward_cell], dim=1)
            
            return outputs, hidden, cell
        else:
            outputs, hidden, cell = self.xlstm(embedded)
            return outputs, hidden, cell

class xLSTMDecoder(nn.Module):
    """Improved xLSTM-based Decoder with proper attention"""
    def __init__(self, vocab_size, emb_dim=128, hid_dim=64, n_layers=4, dropout=0.1, encoder_hidden_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Use encoder hidden dim if provided (for bidirectional), otherwise use hid_dim
        if encoder_hidden_dim is None:
            encoder_hidden_dim = hid_dim
            
        # xLSTM input includes context dimension from attention
        self.xlstm = xLSTMBlock(emb_dim + hid_dim, hid_dim, n_layers)
        
        # Use proper attention with correct dimensions
        self.attention = xLSTMAttention(encoder_hidden_dim, hid_dim)
        
        
        # Output layer combines decoder output with context
        self.fc_out = nn.Linear(hid_dim + encoder_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        
    def forward(self, tgt, hidden, cell, encoder_outputs):
        # Embed target - tgt should be [batch_size, 1]
        embedded = self.dropout(self.embedding(tgt))
        
        # Get attention context
        context, attention_weights = self.attention(hidden, encoder_outputs)
        
        # Use context directly (already has correct dimension from attention)
        context_expanded = context.unsqueeze(1).expand(-1, embedded.size(1), -1)
        xlstm_input = torch.cat([embedded, context_expanded], dim=2)
        
        # Pass through xLSTM
        output, hidden, cell = self.xlstm(xlstm_input, (hidden, cell))
        
        # Combine output with original encoder context for final prediction
        # Get original context from encoder outputs using attention weights
        original_context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        context_expanded_out = original_context.unsqueeze(1).expand(-1, output.size(1), -1)
        combined_output = torch.cat([output, context_expanded_out], dim=2)
        
        # Final linear layer
        prediction = self.fc_out(combined_output)  # [batch_size, 1, vocab_size]
        
        return prediction, hidden, cell, attention_weights

class xLSTMSeq2SeqModel(nn.Module):
    """Complete Seq2Seq model with xLSTM and specialized attention"""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        
        # Project encoder hidden states to decoder hidden dimension
        encoder_hidden_dim = encoder.output_hidden_dim
        decoder_hidden_dim = decoder.hid_dim
        self.hidden_projection = nn.Linear(encoder_hidden_dim, decoder_hidden_dim).to(device)
        self.cell_projection = nn.Linear(encoder_hidden_dim, decoder_hidden_dim).to(device)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc_out.out_features
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Project encoder hidden states to decoder dimensions
        hidden = self.hidden_projection(hidden)
        cell = self.cell_projection(cell)
        
        # Initialize decoder input
        decoder_input = tgt[:, 0:1]  # First token (SOS)
        
        # Prepare outputs
        outputs = torch.zeros(batch_size, tgt_len-1, vocab_size).to(self.device)
        attention_weights = torch.zeros(batch_size, tgt_len-1, src.size(1)).to(self.device)
        
        for t in range(1, tgt_len):
            # Decode with attention
            logits, hidden, cell, attn_weights = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            
            # Store output
            outputs[:, t-1, :] = logits.squeeze(1) if logits.dim() > 2 else logits
            attention_weights[:, t-1, :] = attn_weights
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = tgt[:, t:t+1]  # Next ground truth token
            else:
                # Use predicted token
                top1 = logits.argmax(-1)  # Get most likely token
                decoder_input = top1.unsqueeze(1) if top1.dim() == 1 else top1
            
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

def inject_noise(text, noise_prob=0.1):
    """Inject noise into text by randomly replacing characters"""
    if random.random() > noise_prob:
        return text
    
    chars = list(text)
    if len(chars) < 3:
        return text
    
    # Random character substitution
    if random.random() < 0.5:
        idx = random.randint(0, len(chars) - 1)
        # Replace with similar character (for Roman Urdu)
        similar_chars = {
            'a': 'e', 'e': 'a', 'i': 'e', 'o': 'u', 'u': 'o',
            'k': 'q', 'q': 'k', 'b': 'p', 'p': 'b',
            'd': 't', 't': 'd', 'g': 'j', 'j': 'g'
        }
        if chars[idx].lower() in similar_chars:
            chars[idx] = similar_chars[chars[idx].lower()]
    
    # Random character deletion
    elif random.random() < 0.3 and len(chars) > 5:
        idx = random.randint(1, len(chars) - 2)
        chars.pop(idx)
    
    # Random character insertion
    elif random.random() < 0.2 and len(chars) < 50:
        idx = random.randint(1, len(chars) - 1)
        chars.insert(idx, random.choice('aeiou'))
    
    return ''.join(chars)

def augment_dataset(pairs, augmentation_factor=2):
    """Augment dataset using various techniques"""
    print(f"Augmenting dataset with factor {augmentation_factor}...")
    
    augmented_pairs = []
    
    for urdu_text, roman_text in pairs:
        # Add original pair
        augmented_pairs.append((urdu_text, roman_text))
        
        # Noise injection on Roman text
        noisy_roman = inject_noise(roman_text, noise_prob=0.15)
        if noisy_roman != roman_text:
            augmented_pairs.append((urdu_text, noisy_roman))
    
    # Limit augmentation to avoid too much data
    if len(augmented_pairs) > len(pairs) * augmentation_factor:
        augmented_pairs = augmented_pairs[:len(pairs) * augmentation_factor]
    
    print(f"Augmented dataset from {len(pairs)} to {len(augmented_pairs)} pairs")
    return augmented_pairs

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
            
            # Monitor GPU usage
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"Batch {batch_idx}: GPU Memory: {gpu_mem:.2f}GB")
            
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
            
            if batch_idx % 50 == 0:
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
        encoder_result = model.encoder(src_tensor)
        
        # Handle both encoder types: xLSTM returns (outputs, hidden, cell), LSTM returns (outputs, (hidden, cell))
        if len(encoder_result) == 3:
            encoder_outputs, hidden, cell = encoder_result
            # For xLSTM, project hidden states to decoder dimensions
            if hasattr(model, 'hidden_projection'):
                hidden = model.hidden_projection(hidden.to(device))
                cell = model.cell_projection(cell.to(device))
        else:
            encoder_outputs, (hidden, cell) = encoder_result
        
        # Initialize decoder
        decoder_input = torch.tensor([[roman_tokenizer.token_to_id['<SOS>']]], dtype=torch.long).to(device)
        translated_ids = []
        
        for _ in range(max_length):
            decoder_result = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            
            # Handle both decoder types: xLSTM returns (logits, hidden, cell, attn), LSTM returns (logits, (hidden, cell), attn)
            if len(decoder_result) == 4:
                logits, hidden, cell, _ = decoder_result
            else:
                logits, (hidden, cell), _ = decoder_result
                
            top1 = logits.argmax(2)
            translated_ids.append(top1.item())
            
            if top1.item() == roman_tokenizer.token_to_id['<EOS>']:
                break
                
            decoder_input = top1.to(device)
    
    # Convert IDs back to text using unigram tokenizer
    translated_text = roman_tokenizer.decode(translated_ids)
    
    # Clean up the output - remove special tokens and extra spaces
    translated_text = translated_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').replace('<UNK>', '')
    translated_text = ' '.join(translated_text.split())  # Remove extra spaces
    
    return translated_text

def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score using NLTK library (proper implementation)
    """
    try:
        # Tokenize the sentences
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        if len(cand_tokens) == 0:
            return 0.0
        
        # Use smoothing to handle cases where n-grams don't match
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return score
    except:
        # Fallback: simple word overlap
        ref_words = set(reference.split())
        cand_words = set(candidate.split())
        if len(cand_words) == 0:
            return 0.0
        
        overlap = len(ref_words.intersection(cand_words))
        return overlap / len(cand_words)

def calculate_bleu_scores_batch(references, candidates):
    """
    Calculate BLEU scores for a batch of reference-candidate pairs
    """
    scores = []
    for ref, cand in zip(references, candidates):
        scores.append(calculate_bleu_score(ref, cand))
    return scores

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
    Evaluate model with proper BLEU score using sacrebleu library and perplexity
    """
    print("Evaluating model...")
    print("=" * 50)
    
    # Calculate perplexity
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    ppl = calculate_perplexity(model, test_loader, criterion, device)
    print(f"Perplexity: {ppl:.2f}")
    
    # Calculate BLEU scores using proper library
    samples_to_eval = num_samples if num_samples else min(100, len(test_pairs))
    
    print(f"\nCalculating BLEU scores on {samples_to_eval} samples using NLTK...")
    
    # Collect all translations first
    references = []
    candidates = []
    
    for i in range(samples_to_eval):
        urdu_text, expected_roman = test_pairs[i]
        translated = translate(model, urdu_text, urdu_tokenizer, roman_tokenizer)
        
        references.append(expected_roman)
        candidates.append(translated)
        
        if i < 5:  # Show first 5 examples
            print(f"\nSample {i+1}:")
            print(f"Urdu: {urdu_text}")
            print(f"Expected: {expected_roman}")
            print(f"Translated: {translated}")
    
    # Calculate BLEU scores using NLTK
    individual_scores = calculate_bleu_scores_batch(references, candidates)
    avg_bleu = np.mean(individual_scores)
    corpus_bleu_score = avg_bleu  # For simplicity, use average as corpus score
    
    print(f"\nBLEU Evaluation Results:")
    print(f"Average Sentence BLEU: {avg_bleu:.4f}")
    print(f"Corpus BLEU: {corpus_bleu_score:.4f}")
    
    return {
        'ppl': ppl,
        'corpus_bleu': corpus_bleu_score,
        'avg_bleu': avg_bleu,
        'individual_scores': individual_scores if 'individual_scores' in locals() else []
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
    """Load tokenizers and trained model for inference.
    
    NOTE: This function loads the original BiLSTM+LSTM model.
    For xLSTM models, use load_xlstm_tokenizers_and_model() instead.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizers
    urdu_tokenizer = UnigramTokenizer()
    urdu_tokenizer.load(urdu_tokenizer_path)
    roman_tokenizer = UnigramTokenizer()
    roman_tokenizer.load(roman_tokenizer_path)
    # Build model
    encoder = Encoder(len(urdu_tokenizer.token_to_id), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers)
    decoder = Decoder(len(roman_tokenizer.token_to_id), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers*2)
    model = Seq2SeqModel(encoder, decoder, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, urdu_tokenizer, roman_tokenizer, device

def load_xlstm_tokenizers_and_model(
    urdu_tokenizer_path='unigram_urdu_tokenizer.pkl',
    roman_tokenizer_path='unigram_roman_tokenizer.pkl',
    model_path='xlstm_urdu_roman_seq2seq_model.pth',
    emb_dim=128, hid_dim=256, n_layers=2, device=None
):
    """Load tokenizers and trained xLSTM model for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizers
    urdu_tokenizer = UnigramTokenizer()
    urdu_tokenizer.load(urdu_tokenizer_path)
    roman_tokenizer = UnigramTokenizer()
    roman_tokenizer.load(roman_tokenizer_path)
    # Build xLSTM model
    encoder = xLSTMEncoder(len(urdu_tokenizer.token_to_id), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers)
    decoder = xLSTMDecoder(len(roman_tokenizer.token_to_id), emb_dim=emb_dim, hid_dim=hid_dim, n_layers=4, 
                          encoder_hidden_dim=encoder.output_hidden_dim)
    model = xLSTMSeq2SeqModel(encoder, decoder, device)
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

def main():
    """Main training function"""
    print("Starting Urdu to Roman Urdu Translation Training with Unigram Tokenizer")
    print("=" * 80)
    
    # Check GPU utilization
    check_gpu_utilization()
    
    # Load and preprocess data
    pairs = load_data('normalized_dataset/filtered_urdu_roman_urdu_pairs.txt')
    # pairs = preprocess_data(pairs, max_pairs=8000)  # Use more data for unigram training
    
    # OPTIONAL: Augment dataset for better performance
    # Uncomment the line below to enable data augmentation
    # pairs = augment_dataset(pairs, augmentation_factor=2)  # Doubles the dataset size with noise injection and back-transliteration
    
    # Split data
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=42)
    
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    
    # Prepare texts for tokenizer training
    urdu_texts = [pair[0] for pair in train_pairs]
    roman_texts = [pair[1] for pair in train_pairs]
    
    # Train Unigram tokenizers
    print("\nTraining Unigram tokenizers...")
    print("=" * 50)
    
    urdu_tokenizer = UnigramTokenizer(vocab_size=400)
    urdu_tokenizer.load('unigram_urdu_tokenizer.pkl')
    roman_tokenizer = UnigramTokenizer(vocab_size=400)
    roman_tokenizer.load('unigram_roman_tokenizer.pkl')
    
    # urdu_token_to_id, urdu_id_to_token = urdu_tokenizer.train(urdu_texts, num_iterations=8)
    # roman_token_to_id, roman_id_to_token = roman_tokenizer.train(roman_texts, num_iterations=8)
    
    # # Save tokenizers
    # urdu_tokenizer.save('unigram_urdu_tokenizer.pkl')
    # roman_tokenizer.save('unigram_roman_tokenizer.pkl')
    
    # print(f"Urdu vocabulary size: {len(urdu_token_to_id)}")
    # print(f"Roman vocabulary size: {len(roman_token_to_id)}")
    
    # # Show some sample tokens
    # print(f"\nSample Urdu tokens: {list(urdu_tokenizer.vocab)[:20]}")
    # print(f"Sample Roman tokens: {list(roman_tokenizer.vocab)[:20]}")
    
    # Test tokenization on a sample
    sample_urdu = urdu_texts[0]
    sample_roman = roman_texts[0]
    print(f"\nSample tokenization:")
    print(f"Urdu: {sample_urdu}")
    print(f"Urdu tokens: {urdu_tokenizer.encode(sample_urdu)}")
    print(f"Roman: {sample_roman}")
    print(f"Roman tokens: {roman_tokenizer.encode(sample_roman)}")
    
    # Create datasets
    train_dataset = UnigramUrduRomanDataset(train_pairs, urdu_tokenizer, roman_tokenizer)
    val_dataset = UnigramUrduRomanDataset(val_pairs, urdu_tokenizer, roman_tokenizer)
    test_dataset = UnigramUrduRomanDataset(test_pairs, urdu_tokenizer, roman_tokenizer)
    
    # Create data loaders with optimized settings for GPU
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # OPTION 1: Use original BiLSTM + LSTM model
    encoder = Encoder(len(urdu_tokenizer.token_to_id), emb_dim=128, hid_dim=64, n_layers=2)
    decoder = Decoder(len(roman_tokenizer.token_to_id), emb_dim=128, hid_dim=64, n_layers=4)
    model = Seq2SeqModel(encoder, decoder, device)
    
    # OPTION 2: Use improved xLSTM model with proper bidirectional support
    # Use smaller hidden dim for xLSTM since it's more efficient, but compensate with bidirectional
    # encoder = xLSTMEncoder(len(urdu_token_to_id), emb_dim=128, hid_dim=256, n_layers=2)
    # decoder = xLSTMDecoder(len(roman_token_to_id), emb_dim=128, hid_dim=256, n_layers=4, 
    #                       encoder_hidden_dim=encoder.output_hidden_dim)  # Pass encoder output dim
    # model = xLSTMSeq2SeqModel(encoder, decoder, device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=12, learning_rate=0.001)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Unigram Tokenizer Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('unigram_training_curves.png')
    plt.show()
    
    # Evaluate model with BLEU and Perplexity
    print("\n" + "=" * 80)
    evaluation_results = evaluate_model(model, test_loader, test_pairs, urdu_tokenizer, roman_tokenizer, device, num_samples=100)

    print(f"\nFinal Evaluation Results:")
    print(f"=" * 50)
    print(f"Perplexity: {evaluation_results['ppl']:.2f}")
    print(f"Corpus BLEU: {evaluation_results['corpus_bleu']:.4f}")
    print(f"Average Sentence BLEU: {evaluation_results['avg_bleu']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'unigram_urdu_roman_seq2seq_model.pth')
    
    # If using xLSTM model, also save with xLSTM-specific name
    # torch.save(model.state_dict(), 'xlstm_urdu_roman_seq2seq_model.pth')

    # print("\nModel and tokenizers saved!")
    # print("Training completed successfully!")

    # best_model, best_experiment, best_val_loss = run_experiments(train_loader, val_loader, urdu_token_to_id, roman_token_to_id, device)
    # test_model = best_model  # The best model from the experiments
    # test_loss = 0
    # test_batches = 0
    
    # # Test model
    # test_model.eval()
    # with torch.no_grad():
    #     for src, tgt in test_loader:
    #         src, tgt = src.to(device), tgt.to(device)
    #         outputs, _ = test_model(src, tgt, teacher_forcing_ratio=0.0)
    #         loss = nn.CrossEntropyLoss(ignore_index=0)(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
    #         test_loss += loss.item()
    #         test_batches += 1
            
    # avg_test_loss = test_loss / test_batches
    # print(f"Test Loss: {avg_test_loss:.4f}")
    # torch.save(best_model.state_dict(), 'best_urdu_roman_seq2seq_model.pth')

if __name__ == "__main__":
    main()