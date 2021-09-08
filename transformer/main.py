import time
import math
import copy

from torchtext.datasets import WikiText2
import torch.nn as nn
import torch

from functions import evaluate, train
from dataset import get_vocab, get_processed_data, get_tokenizer
from classes.transformer_model import TransformerModel

tokenizer = get_tokenizer("basic_english")
vocab = get_vocab(WikiText2(split="train"), tokenizer)
device = torch.device("cpu")

ntokens = len(vocab)
lr = 0.5
epoch = 3
bptt = 2
batch_size = 10
device = torch.device("cpu")
train_data, val_data, test_data = get_processed_data(
    WikiText2(), batch_size, device, vocab, tokenizer)

model = TransformerModel(
    ntoken=ntokens,  # Size of vocab
    d_model=200,  # Embedding dimension
    d_hid=200,  # Dimension of hidden feedforward network in nn.TransformerEncoder
    nhead=2,  # Number of heads in nn.MultiHeadAttention
    nlayers=2,  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout=0.2,  # Dropout probability
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

epoch = 3
best_val_loss = float("inf")
best_model = None

print("Training")

for epoch in range(1, epoch + 1):
  epoch_start_time = time.time()
  train(model, criterion, optimizer, scheduler,
        device, train_data, epoch, lr, bptt,  ntokens)
  val_loss = evaluate(model, criterion, val_data, bptt, device, ntokens)
  val_ppl = math.exp(val_loss)
  elapsed = time.time() - epoch_start_time
  print(
      f"{'=' * 89} \n"
      f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s"
      f"| valid loss {val_loss:5.2f} | valid_ppl: {val_ppl:8.2f}s"
      f"{'=' * 89} \n"
  )

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = copy.deepcopy(model)

test_loss = evaluate(best_model, criterion, test_data, bptt, device, ntokens)
test_ppl = math.exp(test_loss)
print(
    f"{'=' * 89}"
    f"| end of training | test loss {test_loss:5.2f} "
    f"| test ppl {test_ppl:8.2f} "
    f"{'=' * 89}"
)
