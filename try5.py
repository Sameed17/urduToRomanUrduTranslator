# assignment_urdu_translation.py
"""
Assignment: Neural Machine Translation - Urdu to Roman Urdu
Complete implementation meeting all assignment requirements
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from collections import Counter
import math
import re
import json
from typing import List, Dict, Tuple
import random

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

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM Encoder (2 layers as per assignment)"""
    
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Combine forward and backward hidden states
        # hidden shape: [n_layers * 2, batch_size, hid_dim]
        hidden = hidden.view(self.rnn.num_layers, 2, hidden.size(1), hidden.size(2))
        hidden = hidden.mean(dim=1)  # Average forward and backward
        
        cell = cell.view(self.rnn.num_layers, 2, cell.size(1), cell.size(2))
        cell = cell.mean(dim=1)
        
        return outputs, (hidden, cell)

class LSTMDecoder(nn.Module):
    """LSTM Decoder (4 layers as per assignment)"""
    
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, 
                          batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, hidden, cell):
        embedded = self.dropout(self.embedding(tgt))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.fc_out(outputs)
        return logits, (hidden, cell)

class Seq2SeqModel(nn.Module):
    """Complete Seq2Seq model with BiLSTM encoder and LSTM decoder"""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_length=50):
        batch_size = src.size(0)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Initialize decoder hidden states to match decoder layers
        # Encoder has 2 layers, decoder has 4 layers
        # The BiLSTMEncoder already averages forward/backward states, so hidden/cell have hid_dim
        decoder_hidden = torch.zeros(self.decoder.rnn.num_layers, batch_size, self.decoder.rnn.hidden_size).to(self.device)
        decoder_cell = torch.zeros(self.decoder.rnn.num_layers, batch_size, self.decoder.rnn.hidden_size).to(self.device)
        
        # Use the last layer of encoder hidden states for all decoder layers
        # Copy the averaged encoder hidden state to all decoder layers
        for i in range(self.decoder.rnn.num_layers):
            decoder_hidden[i] = hidden[-1]  # Use last encoder layer
            decoder_cell[i] = cell[-1]      # Use last encoder layer
        
        if tgt is not None:
            # Training mode with teacher forcing
            tgt_len = tgt.size(1)
            outputs = torch.zeros(batch_size, tgt_len-1, self.decoder.fc_out.out_features).to(self.device)
            
            input_tok = tgt[:, 0]
            
            for t in range(1, tgt_len):
                embedded = self.decoder.embedding(input_tok).unsqueeze(1)
                output, (decoder_hidden, decoder_cell) = self.decoder.rnn(embedded, (decoder_hidden, decoder_cell))
                logits = self.decoder.fc_out(output.squeeze(1))
                outputs[:, t-1, :] = logits
                
                # Teacher forcing
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = logits.argmax(1)
                input_tok = tgt[:, t] if teacher_force else top1
                
            return outputs
        else:
            # Inference mode
            outputs = []
            input_tok = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # SOS token
            
            for t in range(max_length):
                embedded = self.decoder.embedding(input_tok).unsqueeze(1)
                output, (decoder_hidden, decoder_cell) = self.decoder.rnn(embedded, (decoder_hidden, decoder_cell))
                logits = self.decoder.fc_out(output.squeeze(1))
                
                input_tok = logits.argmax(1)
                outputs.append(input_tok)
                
                # Stop if all sequences have generated EOS
                if torch.all(input_tok == 1):  # Assuming EOS token has ID 1
                    break
                    
            return torch.stack(outputs, dim=1)

def build_vocab(texts, min_freq=2):
    """Build vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def load_data(file_path):
    """Load Urdu-Roman Urdu pairs"""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                urdu, roman = line.split('\t', 1)
                pairs.append((urdu.strip(), roman.strip()))
    return pairs

def normalize_text(text):
    """Normalize Urdu text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Handle Urdu-specific normalizations
    text = text.replace('۔', '.')
    text = text.replace('،', ',')
    text = text.replace('؟', '?')
    text = text.replace('؛', ';')
    
    return text

def calculate_bleu_score(prediction, reference, n_gram=4):
    """Calculate BLEU score"""
    def get_ngrams(text, n):
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return Counter(ngrams)
    
    def precision(pred_ngrams, ref_ngrams):
        if not pred_ngrams:
            return 0
        matches = 0
        total = sum(pred_ngrams.values())
        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                matches += min(count, ref_ngrams[ngram])
        return matches / total if total > 0 else 0
    
    precisions = []
    for n in range(1, n_gram + 1):
        pred_ngrams = get_ngrams(prediction, n)
        ref_ngrams = get_ngrams(reference, n)
        prec = precision(pred_ngrams, ref_ngrams)
        precisions.append(prec)
    
    # Brevity penalty
    pred_length = len(prediction.split())
    ref_length = len(reference.split())
    
    if pred_length > ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length / pred_length)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        bleu = 0
    else:
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    return bleu

def calculate_perplexity(model, data_loader, device):
    """Calculate perplexity score"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            outputs = model(src, tgt)
            loss = nn.CrossEntropyLoss(ignore_index=0)(outputs.reshape(-1, outputs.size(-1)), 
                                                     tgt[:, 1:].reshape(-1))
            
            total_loss += loss.item() * (tgt[:, 1:] != 0).sum().item()
            total_tokens += (tgt[:, 1:] != 0).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def calculate_cer(prediction, reference):
    """Calculate Character Error Rate"""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(prediction, reference)
    return distance / max(len(prediction), len(reference), 1)

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        outputs = model(src, tgt)
        
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def ids_to_text(ids, vocab):
    """Convert token IDs back to text"""
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(id, '<UNK>') for id in ids]
    tokens = [t for t in tokens if t not in ['<PAD>', '<SOS>', '<EOS>']]
    return ' '.join(tokens)

def run_experiment(exp_name, emb_dim, hid_dim, enc_layers, dec_layers, dropout, lr, batch_size, device):
    """Run a complete experiment with given hyperparameters"""
    print(f"\n�� Starting Experiment: {exp_name}")
    print(f"�� Parameters: emb_dim={emb_dim}, hid_dim={hid_dim}, enc_layers={enc_layers}, dec_layers={dec_layers}")
    print(f"📊 dropout={dropout}, lr={lr}, batch_size={batch_size}")
    print("=" * 80)
    
    # Load and preprocess data
    print("�� Loading data...")
    pairs = load_data('normalized_dataset/urdu_roman_urdu_pairs.txt')
    pairs = [(normalize_text(urdu), normalize_text(roman)) for urdu, roman in pairs]
    
    # Split data: 50% train, 25% val, 25% test
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.5, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)
    
    print(f"📊 Data split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Build vocabularies
    print("🔤 Building vocabularies...")
    urdu_texts = [pair[0] for pair in pairs]
    roman_texts = [pair[1] for pair in pairs]
    
    urdu_vocab = build_vocab(urdu_texts)
    roman_vocab = build_vocab(roman_texts)
    
    print(f"📝 Urdu vocab size: {len(urdu_vocab)}")
    print(f"📝 Roman vocab size: {len(roman_vocab)}")
    
    # Create datasets and data loaders
    train_dataset = UrduRomanDataset(train_pairs, urdu_vocab, roman_vocab)
    val_dataset = UrduRomanDataset(val_pairs, urdu_vocab, roman_vocab)
    test_dataset = UrduRomanDataset(test_pairs, urdu_vocab, roman_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    encoder = BiLSTMEncoder(len(urdu_vocab), emb_dim, hid_dim, enc_layers, dropout)
    decoder = LSTMDecoder(len(roman_vocab), emb_dim, hid_dim, dec_layers, dropout)
    model = Seq2SeqModel(encoder, decoder, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"🚀 Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"📊 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{exp_name.lower().replace(" ", "_")}.pth')
            print("💾 Best model saved!")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f'best_model_{exp_name.lower().replace(" ", "_")}.pth', map_location=device))
    
    # Evaluation on test set
    print("\n📊 Evaluating on test set...")
    
    bleu_scores = []
    cer_scores = []
    sample_predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_loader):
            src = src.to(device)
            
            # Generate predictions
            pred_ids = model(src)
            
            # Convert to text
            for j in range(src.size(0)):
                if i * batch_size + j >= len(test_pairs):
                    break
                    
                urdu_text, roman_true = test_pairs[i * batch_size + j]
                roman_pred = ids_to_text(pred_ids[j].cpu().numpy(), roman_vocab)
                
                # Calculate metrics
                bleu = calculate_bleu_score(roman_pred, roman_true)
                cer = calculate_cer(roman_pred, roman_true)
                
                bleu_scores.append(bleu)
                cer_scores.append(cer)
                
                # Store samples
                if len(sample_predictions) < 10:
                    sample_predictions.append((urdu_text, roman_true, roman_pred, bleu, cer))
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, test_loader, device)
    
    # Results
    avg_bleu = np.mean(bleu_scores)
    avg_cer = np.mean(cer_scores)
    
    print(f"\n�� Experiment {exp_name} Results:")
    print(f"📊 Average BLEU Score: {avg_bleu:.4f}")
    print(f"�� Perplexity: {perplexity:.4f}")
    print(f"📊 Average CER: {avg_cer:.4f}")
    
    print(f"\n📝 Sample Predictions:")
    for i, (urdu, true, pred, bleu, cer) in enumerate(sample_predictions[:5]):
        print(f"{i+1}. BLEU: {bleu:.3f}, CER: {cer:.3f}")
        print(f"   Urdu: {urdu}")
        print(f"   True: {true}")
        print(f"   Pred: {pred}")
        print()
    
    return {
        'experiment': exp_name,
        'bleu': avg_bleu,
        'perplexity': perplexity,
        'cer': avg_cer,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def main():
    """Run all experiments as per assignment requirements"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Experiment configurations
    experiments = [
        {
            'exp_name': 'Experiment 1 - Baseline',
            'emb_dim': 256,
            'hid_dim': 512,
            'enc_layers': 2,
            'dec_layers': 4,
            'dropout': 0.1,
            'lr': 1e-3,
            'batch_size': 32
        },
        {
            'exp_name': 'Experiment 2 - High Capacity',
            'emb_dim': 512,
            'hid_dim': 512,
            'enc_layers': 4,
            'dec_layers': 4,
            'dropout': 0.3,
            'lr': 5e-4,
            'batch_size': 64
        },
        {
            'exp_name': 'Experiment 3 - Efficient',
            'emb_dim': 128,
            'hid_dim': 256,
            'enc_layers': 2,
            'dec_layers': 2,
            'dropout': 0.5,
            'lr': 1e-4,
            'batch_size': 128
        }
    ]
    
    results = []
    
    for exp_config in experiments:
        result = run_experiment(**exp_config, device=device)
        results.append(result)
    
    # Summary of all experiments
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    
    for result in results:
        print(f"\n{result['experiment']}:")
        print(f"  BLEU: {result['bleu']:.4f}")
        print(f"  Perplexity: {result['perplexity']:.4f}")
        print(f"  CER: {result['cer']:.4f}")
    
    # Find best experiment
    best_bleu = max(results, key=lambda x: x['bleu'])
    best_perplexity = min(results, key=lambda x: x['perplexity'])
    
    print(f"\n�� Best BLEU Score: {best_bleu['experiment']} ({best_bleu['bleu']:.4f})")
    print(f"🏆 Best Perplexity: {best_perplexity['experiment']} ({best_perplexity['perplexity']:.4f})")
    
    print("\n✅ All experiments completed!")

if __name__ == "__main__":
    main()