import torch

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch import Tensor

from typing import Tuple

train_iter = WikiText2(split="train")
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(
  map(tokenizer, train_iter),
  specials=["<unk>"]
)
vocab.set_default_index(vocab["<unk>"])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
  """Converts raw text into a flat tensor"""
  data = [
    torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
    for item in raw_text_iter
  ]

  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data: Tensor, batch_size: int) -> Tensor:
  """Divides the data into batch_size seperate sequences, removing extra elements
  that wouldnt cleanly fit
  
  Args: 
    data: Tensor, shape [N]
    batch_size: int, batch_size
  
  Returns:
    Tensor of shape [N // batch_size, batch_size]
  """
  seq_len = data.size(0) // batch_size
  data = data[:seq_len * batch_size]
  data = data.view(batch_size, seq_len).t().contiguous()

  return data.to(device)

batch_size = 20
eval_batch_size = 20
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, batch_size)
test_data = batchify(test_data, batch_size)

bptt = 35

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
  """
  Args:
    source: Tensor, shape [full_seq_len, batch_size]
    i: int
  
  Returns:
    tuple (data, target), where data has shape [seq_len, batch_size] and target
    has shape [seq_len * batch_size]
  """
  seq_len = min(bptt, len(source) - 1 - i)
  data = source[i:i + seq_len]
  target = source[i + 1: i + 1 + seq_len].reshape(-1)

  return data, target