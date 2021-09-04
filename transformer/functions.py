from dataset import get_batch
from torch import Tensor

import time
import math

import torch
import torch.nn as nn


def train(model,
          criterion,
          optimizer,
          scheduler,
          epoch: int,
          lr: float,
          bptt: int,
          device: torch.device,
          train_data: Tensor,
          ntokens: int) -> None:

  model.train()

  total_loss = 0.0
  log_interval = 200
  start_time = time.time()
  src_mask = generate_square_subsequent_mask(bptt).to(device)

  num_batches = len(train_data) // bptt

  for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    data, targets = get_batch(train_data, i)
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


def evaluate(model, criterion, eval_data: Tensor, bptt: int, device: torch.device, ntokens: int) -> float:
  model.eval()
  total_loss = 0
  src_mask = generate_square_subsequent_mask(bptt).to(device)

  with torch.no_grad():
    for i in range(0, eval_data.size(0) - 1, bptt):
      data, targets = get_batch(eval_data, i)
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