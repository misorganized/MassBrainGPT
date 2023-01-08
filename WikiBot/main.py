import torch
import torch.nn as nn
import torch.optim as optim
import random
import wikipedia
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
import os
import pickle


if torch.cuda.is_available():
    device = 'cuda'
    num_devices = torch.cuda.device_count()
    if num_devices == 1:
        print(f'Using {num_devices} GPU')
    else:
        print(f'Using {num_devices} GPUs')
else:
    device = 'cpu'
    print('Using the CPU')

# Set up the Wikipedia API
wikipedia.set_lang("en")

# Create an empty vocabulary list
vocab = []

# Define the maximum length for the input and target sequences
max_input_length = 50
max_target_length = 50

# Define the start-of-sentence (SOS) and end-of-sentence (EOS) tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# Add the SOS and EOS tokens to the vocabulary list
vocab.append(SOS_TOKEN)
vocab.append(EOS_TOKEN)

# Function to get a random Wikipedia page
def get_random_page():
  while True:
    try:
      page = wikipedia.random(1)
      summary = wikipedia.summary(page, sentences=5)
      return summary
    except Exception:
      pass

# Function to process a page and extract the input and target sequences
def process_page(page):
  # Tokenize the page
  tokens = page.split(" ")
  
  # Trim the input and target sequences to the maximum length
  input_tokens = tokens[:max_input_length]
  target_tokens = tokens[1:max_input_length+1]
  
  # Add the SOS and EOS tokens to the input and target sequences
  input_tokens.insert(0, SOS_TOKEN)
  target_tokens.append(EOS_TOKEN)
  
  # Add the tokens to the vocabulary
  for token in input_tokens:
    if token not in vocab:
      vocab.append(token)
  for token in target_tokens:
    if token not in vocab:
      vocab.append(token)
  
  # Convert the input and target sequences to tensors
  input_tensor = torch.tensor([vocab.index(token) for token in input_tokens])
  target_tensor = torch.tensor([vocab.index(token) for token in target_tokens])
  
  return input_tensor, target_tensor

# Function to create a dataset of input/target tensors from Wikipedia pages
def create_dataset():
  dataset = []
  for i in tqdm(range(30)):
    page = get_random_page()
    input_tensor, target_tensor = process_page(page)
    dataset.append((input_tensor, target_tensor))
  return dataset

print("Starting Dataset")
# Create the dataset
dataset = create_dataset()

batch_size = 25

dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

# Shuffle the dataset
random.shuffle(dataset)

def save_model(folder, name):
    MODEL_PATH = Path(folder)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = name + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. Save the model state dict
    torch.save(obj=bot.state_dict(),  # only saving the state_dict() only saves the models learned parameters
                f=MODEL_SAVE_PATH)

def print_training_progress(epoch, batch_index, batch_size, loss, start_time):
  # Calculate the batch loss
  batch_loss = loss.item() / batch_size
  
  # Calculate the elapsed time
  elapsed_time = time.time() - start_time
  
  # Calculate the estimated time to finish
  batches_per_second = batch_index / elapsed_time
  remaining_time = (num_epochs - epoch) * len(dataloader) / batches_per_second
  
  # Calculate the average loss over the past 10 batches
  if len(losses) < 10:
    avg_loss = sum(losses) / len(losses)
  else:
    avg_loss = sum(losses[-10:]) / 10
  
  # Print the training progress
  print(f"Epoch {epoch}, Batch {batch_index}/{len(dataset)//batch_size}: "
        f"Loss = {batch_loss:.4f}, Avg Loss = {avg_loss:.4f}, "
        f"Time Remaining = {remaining_time:.1f} sec")

# Print the size of the vocabulary
print(len(vocab))
print(dataset)

# Define the model
class TransformerBot(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TransformerBot, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.linear = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, src, tgt):
        x = self.embedding(src)
        y = self.embedding(tgt)

        x = self.transformer(x, y)

        x = self.linear(x)
        return x

bot = TransformerBot(vocab_size=len(vocab), embedding_dim=512, hidden_dim=512, num_layers=4).to(device)


if os.path.exists('./models/wikibot.pth'):
        print("Loaded Model")
        bot = torch.load('./models/wikibot.pth')

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bot.parameters(), lr=0.01)

num_epochs = 10
print("Starting Training")

smoothing_factor = 0.1
losses = []

start_time = time.time()
# Train the model
for epoch in range(num_epochs + 1):
    bot.train()

    for batch_index, (input_tensor, target_tensor) in enumerate(dataset):
        # Move input and target tensors to the device
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        optimizer.zero_grad()

        # Get the output from the model
        output = bot(input_tensor, target_tensor)

        # Calculate the loss and backprpagate the gradients
        loss = loss_fn(output, target_tensor)

        loss.backward()

        optimizer.step()

        if len(losses) == 0:
            losses.append(loss.item())
        else:
            losses.append((1 - smoothing_factor) * losses[-1] + smoothing_factor * loss.item())

    if epoch % 10:
        print_training_progress(epoch, batch_index, batch_size, loss, start_time)
    save_model("models", "wikibot")

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "wikibot-0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=bot.state_dict(),  # only saving the state_dict() only saves the models learned parameters
            f=MODEL_SAVE_PATH)

# Test the model
with torch.no_grad():
  input_tensor = torch.tensor([[vocab.index(SOS_TOKEN)]]).to(device)
  target_tensor = torch.tensor([[vocab.index(EOS_TOKEN)]]).to(device)
  prediction = bot(input_tensor, target_tensor)
  predicted_word = vocab[prediction.argmax().item()]
  output_sentence = predicted_word
  while predicted_word != EOS_TOKEN:
    input_tensor = torch.tensor([[prediction.argmax().item()]]).to(device)
    target_tensor = torch.tensor([[vocab.index(EOS_TOKEN)]]).to(device)
    prediction = bot(input_tensor, target_tensor)
    predicted_word = vocab[prediction.argmax().item()]
    output_sentence += " " + predicted_word
  print(output_sentence)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()