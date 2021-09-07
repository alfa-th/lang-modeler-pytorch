import torch

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch import Tensor

from typing import Tuple, Iterable, Callable, Optional


def get_vocab(train_iter: Iterable,
              tokenizer: Callable[[str], int]):
  """Get vocabulary

  Args:
      tokenizer (Callable[[str], int]): Tokenizer object
      train_iter (Iterable): Train dataset

  Returns:
      vocab: Vocabulary object
  """
  vocab = build_vocab_from_iterator(
      map(tokenizer, train_iter),
      specials=["<unk>"]
  )
  vocab.set_default_index(vocab["<unk>"])

  return vocab


def process_data(raw_text_iter: Iterable, vocab, tokenizer) -> Tensor:
  """Converts raw text into a flat tensor

  Args:
      raw_text_iter (dataset.IterableDataset): Iterable tuple

  Returns:
      Tensor: Flat tensor representing whole dataset
  """
  raw_text_iter = list(raw_text_iter)
  data = [  # shape:
      torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
      for item in raw_text_iter
  ]

  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, batch_size: int, device: torch.device) -> Tensor:
  """Divides the data into batch_size seperate sequences, removing extra elements
  that wouldnt cleanly fit

  Args: 
    data: Tensor, shape [N]
    batch_size: int, batch_size
    device: torch.device

  Returns:
    Tensor of shape [N // batch_size, batch_size]
  """
  seq_len = data.size(0) // batch_size
  data = data[:seq_len * batch_size]
  data = data.view(batch_size, seq_len).t().contiguous()

  return data.to(device)


def get_processed_data(iterable_dataset: Tuple[Iterable, Iterable, Iterable],
                       batch_size: int,
                       device,
                       vocab,
                       tokenizer
                       ) -> Tuple[Tensor, Tensor, Tensor]:
  """Get tuples of processed dataset

  Args:
      device (torch.device)
      batch_size (int): Batch Size

  Returns:
      Tuple[Iterable, Iterable, Iterable]: Tuples of processed dataset
  """
  train_iter, val_iter, test_iter = iterable_dataset
  train_data = batchify(process_data(
      train_iter, vocab, tokenizer), batch_size, device)
  val_data = batchify(process_data(
      val_iter, vocab, tokenizer), batch_size, device)
  test_data = batchify(process_data(
      test_iter, vocab, tokenizer), batch_size, device)

  return train_data, val_data, test_data


def get_data_pair(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
  """
  Args:
    source: Tensor, shape [full_seq_len, batch_size]
    i: int
    bptt: int

  Returns:
    tuple (data, target), where data has shape [seq_len, batch_size] and target
    has shape [seq_len * batch_size]
  """
  seq_len = min(bptt, len(source) - 1 - i)
  data = source[i:i + seq_len]
  target = source[i + 1: i + 1 + seq_len].reshape(-1)

  return data, target


if __name__ == "__main__":
  device = torch.device("cpu")
  tokenizer = get_tokenizer("basic_english")
  vocab = get_vocab(WikiText2(split="train"), tokenizer)
  train_data, _, _ = get_processed_data(
      WikiText2(), 20, device, vocab, tokenizer)
