import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
    """Initiates Transformer Model

    Args:
        ntoken (int): Number of tokens in the vocabulary
        d_model (int): Model embedding dim
        nhead (int): Number of heads in multihead attention model
        d_hid (int): Embedding dimension in encoder
        nlayers (int): Number of encoder layers
        dropout (float, optional): Probability to dropout in encoder. Defaults to 0.5.
    """
    super().__init__()

    self.model_type = "Transformers"

    self.pos_encoder = PositionalEncoding(d_model, dropout)
    encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.encoder = nn.Embedding(ntoken, d_model)
    self.decoder = nn.Linear(d_model, ntoken)

    self.d_model = d_model
    self.init_weights()

  def init_weights(self) -> None:
    """This method initiates weights and biases
    """
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
    """
    Args:
      src: Tensor, shape [seq_len, batch_size]
      src_mask: Tensor, shape [seq_len, seq_len]

    Returns:
      output Tensor of shape [seq_len, batch_size, ntoken]
    """
    src = self.encoder(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, src_mask)
    output = self.decoder(output)

    return output
