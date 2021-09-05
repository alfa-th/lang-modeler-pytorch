import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
  pe: Tensor

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    """Given an embedding depth, this will initiates PositionalEncoding
    and prepares positional tensor with the length of 'max_len' with forward
    method that accepts tensor with the size of [seq_len, batch_size, d_model]
    that will add positional tensor with the same shape.
    Refer to: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    Args:
        d_model (int): Embedding depth 
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        max_len (int, optional): The default length of the positional tensor. Defaults to 5000.
    """
    super().__init__()

    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)  # shape: [max_len, 1]
    div_term = torch.exp(  # shape: [d_model // 2]
        torch.arange(0, d_model, 2) * # shape: [d_model // 2]
        (-math.log(10000.0) / d_model) 
    ) 
    pe = torch.zeros(max_len, 1, d_model) # shape: [max_len, 1, d_model]
    pe[:, 0, 0::2] = torch.sin(position * div_term) 
    pe[:, 0, 1::2] = torch.cos(position * div_term)

    self.register_buffer("pe", pe)  # type: Tensor


  def forward(self, x: Tensor) -> Tensor:
    """Given a tensor with the shape of [seq_len, batch_size, embedding_dim],
    this method will add positional encoding, despite of 'batch_size' and
    applies nn.Dropout.

    Args:
       x: Tensor, shape [seq_len, batch_size, embedding_dim]

    Returns:
        Tensor: []
    """
    x = x + self.pe[:x.size(0)]

    return self.dropout(x)
