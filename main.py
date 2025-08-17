#    Copyright 2025 Fabian Sauer
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
Transformer Training Script for Machine Translation

This script provides a complete training pipeline for the Transformer model,
including data preprocessing, vocabulary building, training with proper scheduling,
evaluation, and inference capabilities.

Usage:
    # Train a new model
    python train.py -t source.txt target.txt -d model_dir -e 20

    # Load existing model for interactive translation
    python train.py -d model_dir -i
"""

import argparse
import json
import math
import os
import random
import unicodedata
import re
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from tqdm import tqdm

from transformer import Transformer
from preProcessData import DataPreparation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def unicode_to_ascii(s):
    """Turn a Unicode string to plain ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if not unicodedata.combining(c)
    )


def normalize_string(s, prep=None):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # Falls prep gesetzt wurde, entweder Stemming oder Lemmatization zusätzlich hinzufügen
    if prep is not None:
        s = prep.transform(s)
    return s.strip()


def tokenize(s, prep=None):
    """Tokenize a sentence"""
    return normalize_string(s, prep).split()


def read_corpus(src_file, tgt_file, prep, max_len, min_len=0):
    """Read, tokenize and filter a training corpus"""
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f.readlines()]

    assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"

    # Tokenize pairs of sentences
    pairs = list(zip(src_lines, tgt_lines))
    if prep is not None:
        pairs = [(tokenize(src,prep.preparation_french), tokenize(tgt, prep.preparation_english)) for src, tgt in pairs]
    else:
        pairs = [(tokenize(src), tokenize(tgt)) for src, tgt in pairs]

    # Filter pairs by sentence length constraints
    pairs = [(src, tgt) for src, tgt in pairs
             if min_len < len(src) < max_len and min_len < len(tgt) < max_len]

    return pairs


def train_val_split(pairs, val_ratio=0.2, shuffle=True):
    """Perform a train/val split with default ratio 80/20"""
    if shuffle:
        random.shuffle(pairs)
    val_size = int(len(pairs) * val_ratio)
    return pairs[val_size:], pairs[:val_size]


def build_vocab(sentences, min_freq):
    """Build vocabulary from sentences"""
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    # Special tokens
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4

    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def tokens_to_indices(tokens, vocab):
    """Convert tokens to indices"""
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]


def indices_to_tokens(indices, vocab):
    """Convert indices back to tokens"""
    idx_to_token = {v: k for k, v in vocab.items()}
    return [idx_to_token.get(idx, '<UNK>') for idx in indices]


class MTDataset(Dataset):
    def __init__(self, pairs, src_vocab=None, tgt_vocab=None, min_freq=2):
        """
        Dataset for machine translation

        Args:
            pairs: Tuples of source and target token lists
            src_vocab: Existing source vocabulary (for test/val sets)
            tgt_vocab: Existing target vocabulary (for test/val sets)
            min_freq: Minimum frequency for vocabulary
        """

        # Unzip sentence pairs
        src_sentences, tgt_sentences = list(zip(*pairs))

        print(f"Loaded {len(src_sentences)} sentence pairs")

        # Build or reuse vocabularies
        if src_vocab is None:
            self.src_vocab = build_vocab(src_sentences, min_freq)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = build_vocab(tgt_sentences, min_freq)
        else:
            self.tgt_vocab = tgt_vocab

        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.tgt_vocab)}")

        # Convert sentences to index sequences
        self.src_indices = [tokens_to_indices(sent, self.src_vocab) for sent in src_sentences]
        self.tgt_indices = [tokens_to_indices(sent, self.tgt_vocab) for sent in tgt_sentences]

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        src = self.src_indices[idx]
        tgt = self.tgt_indices[idx]

        # Prepare target sequences for teacher forcing
        # Input: <SOS> + target tokens (for decoder input)
        # Output: target tokens + <EOS> (for loss calculation)
        tgt_input = [self.tgt_vocab['<SOS>']] + tgt
        tgt_output = tgt + [self.tgt_vocab['<EOS>']]

        return {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function to handle padding"""
    src_seqs = [item['src'] for item in batch]
    tgt_input_seqs = [item['tgt_input'] for item in batch]
    tgt_output_seqs = [item['tgt_output'] for item in batch]

    # Pad sequences to the same length within the batch
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_input_padded = nn.utils.rnn.pad_sequence(tgt_input_seqs, batch_first=True, padding_value=0)
    tgt_output_padded = nn.utils.rnn.pad_sequence(tgt_output_seqs, batch_first=True, padding_value=0)

    return {
        'src': src_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded
    }


class WarmupCosineScheduler:
    """
    Custom warmup + cosine decay scheduler
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate for the next step."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """
        Calculate current learning rate based on step count.

        Returns:
            float: Current learning rate
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup phase
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to prevent negative values
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr_ratio * self.base_lr + (1 - self.min_lr_ratio) * self.base_lr * cosine_decay


class Translator:
    """
    Complete Transformer training and inference system for machine translation.

    This class handles:
    - Model initialization and configuration
    - Training with proper scheduling and validation
    - Model saving/loading with vocabularies
    - Inference and interactive translation

    Args:
        n_layers (int): Number of transformer layers
        d_model (int): Model dimension
        d_ff (int): Feed-forward network dimension
        n_heads (int): Number of attention heads
        max_seq_length (int): Maximum sequence length
        dropout_p (float): dropout_p probability
        model_dir (str): Directory for saving/loading models
        device (str): Device for computation ('cuda' or 'cpu')
    """

    def __init__(self, n_layers, d_model, d_ff, n_heads, max_seq_length, dropout_p,
                 model_dir='model', device='cuda'):
        # Model architecture parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.dropout_p = dropout_p

        # Training setup
        self.model_dir = model_dir
        self.device = device

        # Data components (initialized during training)
        self.src_vocab = None
        self.tgt_vocab = None
        self.train_loader = None
        self.val_loader = None

        # Model components (initialized when model is created)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    @classmethod
    def load(cls, model_dir='model', device='cuda'):
        """
        Load a trained model from disk.

        Args:
            model_dir (str): Directory containing saved model
            device (str): Device for computation

        Returns:
            Translator: Loaded translator instance
        """
        # Load model parameters
        with open(os.path.join(model_dir, 'params.json'), 'r') as f:
            params = json.load(f)

        # Create translator instance
        translator = cls(params['n_layers'], params['d_model'], params['d_ff'],
                         params['n_heads'], params['max_seq_length'], params['dropout_p'],
                         model_dir=model_dir, device=device)

        # Load vocabularies
        with open(os.path.join(model_dir, 'src_vocab.pkl'), 'rb') as f:
            translator.src_vocab = pickle.load(f)
        with open(os.path.join(model_dir, 'tgt_vocab.pkl'), 'rb') as f:
            translator.tgt_vocab = pickle.load(f)

        # Initialize model (total_steps=0 since we're not training)
        translator.create_model(0)

        # Load model weights
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'),
                                map_location=device, weights_only=True)

        translator.model.load_state_dict(checkpoint['model_state_dict'])
        translator.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return translator

    def save(self):
        """Save model parameters and vocabularies to disk."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Save model architecture parameters
        params = {
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'n_heads': self.n_heads,
            'max_seq_length': self.max_seq_length,
            'dropout_p': self.dropout_p
        }

        with open(os.path.join(self.model_dir, 'params.json'), 'w') as f:
            json.dump(params, f)

        # Save vocabularies
        with open(os.path.join(self.model_dir, 'src_vocab.pkl'), 'wb') as f:
            pickle.dump(self.src_vocab, f)
        with open(os.path.join(self.model_dir, 'tgt_vocab.pkl'), 'wb') as f:
            pickle.dump(self.tgt_vocab, f)

    def create_model(self, total_steps):
        """
        Initialize the transformer model and training components.

        Args:
            total_steps (int): Total number of training steps (for scheduler)
        """
        # Initialize transformer model
        model = Transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            max_seq_length=self.max_seq_length,
            dropout_p=self.dropout_p
        )
        self.model = model.to(self.device)

        print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Adam optimizer with transformer-specific hyperparameters
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

        # Learning rate scheduler with warmup
        warmup_steps = min(4000, total_steps // 10)  # 10% warmup or 4000 steps max
        self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_steps, total_steps)

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(src, tgt_input)

            # Compute loss (reshape for CrossEntropyLoss)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update parameters and learning rate
            self.optimizer.step()
            self.scheduler.step()

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.2f}', 'lr': f'{current_lr:.2e}'})

        return total_loss / len(self.train_loader)

    def evaluate(self):
        """
        Evaluate the model on the validation set.

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)

                # Forward pass without gradient computation
                output = self.model(src, tgt_input)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, source, target, prep,num_epochs):
        """
        Complete training pipeline for the transformer model.

        Args:
            source (str): Path to source language file
            target (str): Path to target language file
            num_epochs (int): Number of training epochs
        """
        # Load and preprocess data
        print("Loading and preprocessing data...")
        train_pairs, val_pairs = train_val_split(read_corpus(source, target, prep, max_len=50))

        # Create datasets
        train_dataset = MTDataset(train_pairs)
        val_dataset = MTDataset(val_pairs, src_vocab=train_dataset.src_vocab,
                                tgt_vocab=train_dataset.tgt_vocab)

        # Save vocabularies for later use
        self.src_vocab = train_dataset.src_vocab
        self.tgt_vocab = train_dataset.tgt_vocab
        self.save()

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

        # Calculate total training steps for scheduler
        total_steps = len(self.train_loader) * num_epochs

        # Initialize model and training components
        self.create_model(total_steps)

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            train_loss = self.train_epoch()

            # Evaluate on validation set
            val_loss = self.evaluate()

            # Log progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")

            # Save checkpoint
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(self.model_dir, f'checkpoint_{epoch + 1}.pt'))

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(self.model_dir, 'best_model.pt'))
                print("Saved best model!")

    def translate(self, sentence, prep, max_len=50):
        """
        Translate a source sequence using greedy decoding.

        Args:
            sentence (str): Source sentence
            max_len (int): Maximum translation length

        Returns:
            str: Translated sentence
        """
        self.model.eval()

        # Preprocess and convert to indices
        if prep is not None:
            tokens = tokenize(sentence, prep.preparation_french)
        else:
            tokens = tokenize(sentence)
        src_indices = tokens_to_indices(tokens, self.src_vocab)

        # Add batch dimension and move to device
        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)

        # Start with SOS token
        tgt_indices = [self.tgt_vocab['<SOS>']]

        with torch.no_grad():
            # Encode source sequence
            encoder_output = self.model.encode(src)

            # Generate translation one token at a time
            for _ in range(max_len):
                tgt = torch.tensor([tgt_indices], dtype=torch.long).to(self.device)

                # Decode current target sequence
                decoder_output = self.model.decode(tgt, encoder_output)

                # Get logits for next token
                next_token_logits = self.model.output_projection(decoder_output[:, -1, :])
                next_token = next_token_logits.argmax(dim=-1).item()

                # Stop if EOS token is generated
                if next_token == self.tgt_vocab['<EOS>']:
                    break

                tgt_indices.append(next_token)

        # Convert indices back to tokens (skip SOS token)
        idx_to_token = {v: k for k, v in self.tgt_vocab.items()}
        translation = [idx_to_token.get(idx, '<UNK>') for idx in tgt_indices[1:]]

        return ' '.join(translation)


def main():
    """
    Main function to handle command line arguments and run training/inference.
    """
    parser = argparse.ArgumentParser(description="Train and use Transformer for machine translation")

    # Model and training parameters
    parser.add_argument('-d', '--dir', default='./model',
                        help='Directory to save/load model (default: ./model)')
    parser.add_argument('-t', '--train', nargs=2, metavar=('SOURCE', 'TARGET'),
                        help='Train the model on aligned corpus files'),
    parser.add_argument('-o', '--optimised', choices=["stemming", "lemmatization"],
                        help='Enable "stemming" or "lemmatization" to optimize the training data'),
    parser.add_argument('-y', '--num-layers', type=int, default=4,
                        help='Number of transformer layers (default: 4)')
    parser.add_argument('-m', '--model-dim', type=int, default=512,
                        help='Model dimension (default: 512)')
    parser.add_argument('-f', '--ff-dim', type=int, default=2048,
                        help='Feed-forward dimension (default: 2048)')
    parser.add_argument('-a', '--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('-l', '--max-len', type=int, default=50,
                        help='Maximum sentence length (default: 50)')
    parser.add_argument('-e', '--n-epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Enter interactive translation mode')
    parser.add_argument('-c', '--compare', nargs=5, metavar=('TRAIN_SOURCE', 'TARGET','STEMMING_MODEL','LEMMATIZATION_MODEL', 'NORMAL_MODEL'),
                        help='Compare the Translations of different Models with BLEU Score')
    # parser.add_argument('input', help='Translate input string to stdout')

    args = parser.parse_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.train is not None:
        # Training mode

        # Data preparation, initially set to None, can be set to "stemming" or "lemmatization"
        prep = None

        if args.optimised is not None:
            if args.optimised == "stemming":
                prep = DataPreparation(option="stemming")
            elif args.optimised == "lemmatization":
                prep = DataPreparation(option="lemmatization")
            else:
                print(f"Error: Unknown optimization method '{args.optimised}'. Use 'stemming' or 'lemmatization'.")
                sys.exit(1)  # Programm wird abgebrochen

            print(f"Preprocess the training data with the following method: {args.optimised}")

            # create new optimized english and french dataset

        train_source, train_target = args.train
        print(f"Training on {train_source} -> {train_target}")

        

        # Initialize and train translator
        translator = Translator(
            n_layers=args.num_layers,
            d_model=args.model_dim,
            d_ff=args.ff_dim,
            n_heads=args.num_heads,
            max_seq_length=args.max_len,
            dropout_p=0.1,
            model_dir=args.dir,
            device=device
        )

        translator.train(train_source, train_target, prep ,num_epochs=args.n_epochs)

        # Show some example translations after training
        print("\n*** Sample Translations ***")
        with open(train_source, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f.readlines() if 0 < len(tokenize(line)) < args.max_len]
            for _ in range(5):
                sentence = random.choice(src_lines)
                print(f"Test sentence: {sentence}")
                print(f"> Translation: {translator.translate(sentence, prep, args.max_len)}\n")

    else:
       # Load existing model
        print(f"Loading model from {args.dir}")
        translator = Translator.load(args.dir, device)

    if args.interactive:
        # Interactive translation mode
        print("\n*** Interactive Translation Mode ***")
        print("Enter sentences to translate (empty line to exit):\n")

        while True:
            sentence = input("Input: ")
            if sentence.strip() == '':
                break

            try:
                translation = translator.translate(sentence, None)
                print(f"Translation: {translation}\n")
            except Exception as e:
                print(f"Error: {e}\n")

    elif args.input is not None:
        print(f"Input sentence: {args.input}")
        print(f">  Translation: {translator.translate(args.input)}")

    if args.compare is not None:
        # load the 2 models given by the user
        train_source, target_source , model_stemming, model_lemmatization, model_noOptimization = args.compare

        # create Translator for each model
        translator_StemmingModel = Translator.load(model_stemming, device)
        translator_LemmatizationModel = Translator.load(model_lemmatization, device)
        translator_noOptimizationModel = Translator.load(model_noOptimization,device)

        avgBLEUStem= 0
        avgBLEULemm = 0
        avgBLEUNoOpt = 0 

        smoothie = SmoothingFunction().method4

        prep_stemming = DataPreparation(option="stemming")
        prep_lemmatization = DataPreparation(option="lemmatization")

        with open(train_source, 'r', encoding='utf-8') as f_src, open(target_source, 'r', encoding='utf-8') as f_tgt:
            src_lines = []
            tgt_lines = []
            
            for src, tgt in zip(f_src, f_tgt):
                src = src.strip()
                tgt = tgt.strip()
                
                if 0 < len(tokenize(src)) < args.max_len and 0 < len(tokenize(tgt)) < args.max_len:
                    src_lines.append(src)
                    tgt_lines.append(tgt)

        number_of_runs = 10

        for _ in range(number_of_runs):

            averageBleuScoreNoOptimization_run = 0 
            averageBlueScoreWithStemming_run = 0
            averageBleuScoreLemmatization_run = 0

            number_of_sentences = 50

            for _ in range(number_of_sentences):

                line_number = random.randrange(len(src_lines))   
                sentence = src_lines[line_number] 
                best_translation = tgt_lines[line_number]
                #print("\n")               
                #print(f"Test sentence: {sentence}")
                #print(f"Given Translation: {best_translation}")

                # No Optimization
                translation_noOptimization = translator_noOptimizationModel.translate(sentence, None, args.max_len).split()
                bleu_noOpt = sentence_bleu([tokenize(best_translation)], translation_noOptimization, smoothing_function=smoothie)
                averageBleuScoreNoOptimization_run = averageBleuScoreNoOptimization_run + bleu_noOpt
                #print(f"\nNo Optimization\n: {' '.join(translation_noOptimization)} (BLEU: {bleu_noOpt:.4f})")

                # Stemming
                translation_stemming = translator_StemmingModel.translate(sentence, prep_stemming, args.max_len).replace(" .", "").split()
                bleu_stem = sentence_bleu([tokenize(best_translation, prep_stemming.preparation_english)], translation_stemming, smoothing_function=smoothie)
                averageBlueScoreWithStemming_run = averageBlueScoreWithStemming_run + bleu_stem
                #print(f"\nWith Stemming\n: {' '.join(translation_stemming)} (BLEU: {bleu_stem:.4f})")

                # Lemmatization
                # Stemming
                translation_lemmatization = translator_LemmatizationModel.translate(sentence, prep_lemmatization, args.max_len).replace(" .", "").split()
                bleu_lemm = sentence_bleu([tokenize(best_translation, prep_lemmatization.preparation_english)], translation_lemmatization, smoothing_function=smoothie)
                averageBleuScoreLemmatization_run= averageBleuScoreLemmatization_run + bleu_lemm
                #print(f"\nWith Lemmatization\n: {' '.join(translation_lemmatization)} (BLEU: {bleu_lemm:.4f})")
            
            #print(f"Average BLEU-Scores of this run = No Optimization: {averageBleuScoreNoOptimization_run/number_of_sentences} | With Stemming: {averageBlueScoreWithStemming_run/number_of_sentences} | With Lemmatization: {(averageBleuScoreLemmatization_run/number_of_sentences)}")
            avgBLEUNoOpt = avgBLEUNoOpt + averageBleuScoreNoOptimization_run/number_of_sentences
            avgBLEUStem = avgBLEUStem + averageBlueScoreWithStemming_run/number_of_sentences
            avgBLEULemm = avgBLEULemm + averageBleuScoreLemmatization_run/number_of_sentences

        print(f"Average BLEU-Score of this run = No Optimization: {avgBLEUNoOpt/number_of_runs} | With Stemming: {avgBLEUStem/10} | With Lemmatization: {avgBLEULemm/number_of_runs}")
# Example usage and entry point
if __name__ == "__main__":
    main()