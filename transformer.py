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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding

    Positional encoding adds information about the position of tokens in the sequence
    since the transformer architecture doesn't have inherent positional awareness.

    Args:
        d_model (int): The model's hidden dimension
        max_seq_length (int): Maximum sequence length to precompute encodings for
    """

    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_seq_length, d_model)

        # Create position indices [0, 1, 2, ..., max_seq_length-1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Create the division term for the sinusoidal pattern
        # This creates [1, 1/10000^(2/d_model), 1/10000^(4/d_model), ...]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's part of the model but not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Input with added positional encoding
        """
        return x + self.pe[:, : x.shape[1]]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism

    Computes attention across multiple representation subspaces simultaneously.
    Each head learns different types of relationships between tokens.

    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout_p (float): Dropout probability
    """

    def __init__(self, d_model, n_heads, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (
                d_model % n_heads == 0
        ), "Model dimension must be divisible by number of heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear projections for queries, keys, values, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_p)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention.

        Attention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V

        Args:
            q, k, v (torch.Tensor): Query, key, value tensors of shape
                                   (batch_size, n_heads, seq_len, d_k)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Attention output of shape (batch_size, n_heads, seq_len, d_k)
                - Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (set masked positions to very negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def split_heads(self, x):
        """
        Split the last dimension into multiple heads.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention.

        Args:
            query, key, value (torch.Tensor): Input tensors of shape
                                            (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            tuple: (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()

        # Apply linear transformations to get Q, K, V
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Split into multiple heads
        q = self.split_heads(q)  # (batch_size, n_heads, seq_len, d_k)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Apply scaled dot-product attention
        (attention_output,
         attention_weights) = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads back together (inverse operation to split_heads)
        batch_size, _, seq_len, _ = attention_output.shape
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)

        # Apply final linear transformation
        output = self.w_o(attention_output)

        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applies two linear transformations with ReLU activation in between.

    Args:
        d_model (int): Model dimension
        d_ff (int): Hidden dimension of feed-forward network (usually 4 * d_model)
        dropout_p (float): Dropout probability
    """

    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single layer of the Transformer encoder.

    Consists of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Each sub-layer has residual connection and layer normalization.

    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        dropout_p (float): Dropout probability
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        """
        Forward pass through encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask for padding

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder.

    Consists of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention (attend to encoder output)
    3. Position-wise feed-forward network
    Each sub-layer has residual connection and layer normalization.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_p)  # masked self-attn
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout_p) # encoder-decoder attn
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention
        attn1, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # 2. Cross-attention: queries from decoder, keys/values from encoder
        attn2, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.

    Consists of an encoder stack and decoder stack, each with multiple layers.

    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary
        n_layers (int): Number of encoder/decoder layers
        d_model (int): Model dimension
        d_ff (int): Feed-forward hidden dimension
        n_heads (int): Number of attention heads
        max_seq_length (int): Maximum sequence length for positional encoding
        dropout_p (float): Dropout probability
    """

    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            n_layers=6,
            d_model=512,
            d_ff=2048,
            n_heads=8,
            max_seq_length=5000,
            dropout_p=0.1,
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model

        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout_p) for _ in range(n_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout_p) for _ in range(n_layers)]
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def create_padding_mask(self, seq, pad_idx=0):
        """
        Create a mask to hide padding tokens from attention.

        Args:
            seq (torch.Tensor): Input sequence with padding
            pad_idx (int): Index used for padding tokens

        Returns:
            torch.Tensor: Boolean mask where True indicates real tokens
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        """
        Create a look-ahead mask to prevent attention to future positions.

        Used in decoder self-attention to maintain autoregressive property.

        Args:
            size (int): Sequence length

        Returns:
            torch.Tensor: Lower triangular boolean matrix
        """
        return torch.ones(size, size).tril().bool()

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source token indices of shape (batch_size, src_seq_len)
            src_mask (torch.Tensor, optional): Source padding mask

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, src_seq_len, d_model)
        """
        # Embedding and positional encoding
        # Scale embeddings by sqrt(d_model) as per original paper
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        src_embedded = self.dropout(src_embedded)

        # Pass through encoder layers
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        return encoder_output

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode the target sequence given encoder output.

        Args:
            tgt (torch.Tensor): Target token indices of shape (batch_size, tgt_seq_len)
            encoder_output (torch.Tensor): Encoder output
            src_mask (torch.Tensor, optional): Source padding mask
            tgt_mask (torch.Tensor, optional): Target mask (padding + look-ahead)

        Returns:
            torch.Tensor: Decoded representation of shape (batch_size, tgt_seq_len, d_model)
        """
        # Embedding and positional encoding
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        tgt_embedded = self.dropout(tgt_embedded)

        # Pass through decoder layers
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        return decoder_output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the complete transformer.

        Args:
            src (torch.Tensor): Source token indices of shape (batch_size, src_seq_len)
            tgt (torch.Tensor): Target token indices of shape (batch_size, tgt_seq_len)
            src_mask (torch.Tensor, optional): Source padding mask
            tgt_mask (torch.Tensor, optional): Target mask

        Returns:
            torch.Tensor: Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(
                tgt.device
            )
            # Expand look_ahead_mask to match padding_mask dimensions
            tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)
            # Combine padding and look-ahead masks
            tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

        # Encode source sequence
        encoder_output = self.encode(src, src_mask)

        # Decode target sequence
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        return output


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating how to use the Transformer model.

    This creates a small transformer and runs a forward pass with random data
    to verify the implementation works correctly.
    """
    # Model hyperparameters
    src_vocab_size = 10000  # Source vocabulary size
    tgt_vocab_size = 10000  # Target vocabulary size
    d_model = 512  # Model dimension
    n_heads = 8  # Number of attention heads
    n_layers = 6  # Number of encoder/decoder layers
    d_ff = 2048  # Feed-forward hidden dimension
    max_seq_length = 100  # Maximum sequence length

    # Create model instance
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
    )

    # Example input data
    batch_size = 2
    src_seq_len = 20  # Source sequence length
    tgt_seq_len = 15  # Target sequence length

    # Random token indices (excluding 0 which is typically used for padding)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    # Forward pass
    output = model(src, tgt)

    # Print model information
    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1e6:.1f}MB")
