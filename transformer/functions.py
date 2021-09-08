import time
import math

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

from dataset import get_data_pair
from torch import Tensor


def train(model,
          criterion,
          optimizer,
          scheduler,
          device,
          train_data: Tensor,
          epoch: int,
          lr: float,
          bptt: int,
          ntokens: int) -> None:
  """Train a model

  Args:
      model: PyTorch model
      criterion: PyTorch criterion
      optimizer: PyTorch optimizer
      scheduler: PyTorch scheduler
      device: PyTorch device
      epoch (int): How many times training happens
      lr (float): Learning rate
      bptt (int): Subbatch per batch
      train_data (Tensor): Training data in tensor form
      ntokens (int): How many tokens in vocab
  """
  model.train()

  total_loss = 0.0
  log_interval = 200
  start_time = time.time()
  src_mask = generate_square_subsequent_mask(bptt).to(device)

  num_batches = len(train_data) // bptt

  for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    data, targets = get_data_pair(train_data, i, bptt)
    batch_size = data.size(0)
    if batch_size != bptt:
      src_mask = src_mask[:batch_size, :batch_size]
    output = model(data, src_mask)
    loss = criterion(output.view(-1, ntokens), targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()

    if batch % log_interval == 0 and batch > 0:
      lr = scheduler.get_last_lr()[0]
      ms_per_batch = (time.time() - start_time) * 1000 / log_interval
      cur_loss = total_loss / log_interval
      ppl = math.exp(cur_loss)
      print(
          f"| Epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches "
          f"| lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} "
          f"| loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
      )
      total_loss = 0
      start_time = time.time()

  scheduler.step()


def evaluate(model,
             criterion,
             eval_data: Tensor,
             bptt: int,
             device: torch.device,
             ntokens: int) -> float:
  """Evaluate model given a criterion and evaluation data

  Args:
      model (): PyTorch model
      criterion (): PyTorch criterion
      eval_data (Tensor): Evaluation data in tensor form
      bptt (int): Subbatch per batch
      device (torch.device): PyTorch device
      ntokens (int): Number of tokens in vocabulary

  Returns:
      float: Evaluation score
  """
  model.eval()
  total_loss = 0
  src_mask = generate_square_subsequent_mask(bptt).to(device)

  with torch.no_grad():
    for i in range(0, eval_data.size(0) - 1, bptt):
      data, targets = get_data_pair(eval_data, i, bptt)
      batch_size = data.size(0)
      if batch_size != bptt:
        src_mask = src_mask[:batch_size, :batch_size]
      output = model(data, src_mask)
      output_flat = output.view(-1, ntokens)
      total_loss += batch_size * criterion(output_flat, targets).item()

  return total_loss / (len(eval_data) - 1)


def generate_square_subsequent_mask(size: int) -> Tensor:
  """Generates an upper triangular matrix with of -inf, with zeros on diag"""
  return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)


if __name__ == "__main__":
  from classes.transformer_model import TransformerModel
  from dataset import get_vocab, get_processed_data
  from torchtext.datasets import WikiText2

  tokenizer = get_tokenizer("basic_english")
  vocab = get_vocab(WikiText2(split="train"), tokenizer)

  ntoken = len(vocab)
  lr = 0.5
  epoch = 3
  bptt = 2
  batch_size = 10
  device = torch.device("cpu")
  train_data, _, _ = get_processed_data(
      WikiText2(), batch_size, device, vocab, tokenizer)

  model = TransformerModel(
      ntoken=ntoken,  # Size of vocab
      d_model=200,  # Embedding dimension
      d_hid=200,  # Dimension of hidden feedforward network in nn.TransformerEncoder
      nhead=2,  # Number of heads in nn.MultiHeadAttention
      nlayers=2,  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
      dropout=0.2,  # Dropout probability
  )

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

  train(model, criterion, optimizer, scheduler,
        device, train_data, epoch, lr, bptt,  ntoken)
