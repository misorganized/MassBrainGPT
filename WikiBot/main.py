import torch
import torch.nn as nn
import torch.optim as optim
import random
import wikipedia
from tqdm import tqdm
import json
from bs4 import BeautifulSoup

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
      with open('pages.json', 'a') as file:
        json.dump(summary, file, indent=2)
        file.write('\n')
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
  for i in tqdm(range(10)):
    page = get_random_page()
    input_tensor, target_tensor = process_page(page)
    dataset.append((input_tensor, target_tensor))
  return dataset
print("Starting Dataset")
# Create the dataset
dataset = create_dataset()

# Shuffle the dataset
random.shuffle(dataset)

# Print the size of the vocabulary
print(len(vocab))

# Define the model
class TransformerBot(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
    super(TransformerBot, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.transformer = nn.Transformer(embedding_dim, hidden_dim, num_layers)
    self.fc = nn.Linear(hidden_dim, vocab_size)
  
  def forward(self, x):
    x = self.embedding(x) 
    x = self.transformer(x)
    x = self.fc(x)
    return x

bot = TransformerBot(vocab_size=len(vocab), embedding_dim=256, hidden_dim=512, num_layers=2).to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bot.parameters())

num_epochs = 1
print("Starting Training")
# Train the model
for epoch in range(num_epochs):
    for input_tensor, target_tensor in dataset:
        # Move input and target tensors to the device
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # Get the output from the model
        output = bot(input_tensor)

        # Calculate the loss and backprpagate the gradients
        loss = loss_fn(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test the model
with torch.no_grad():
  input_tensor = torch.tensor([[vocab.index(SOS_TOKEN)]]).to(device)
  prediction = bot(input_tensor)
  predicted_word = vocab[prediction.argmax().item()]
  output_sentence = predicted_word
  while predicted_word != EOS_TOKEN:
    input_tensor = torch.tensor([[prediction.argmax().item()]]).to(device)
    prediction = bot(input_tensor)
    predicted_word = vocab[prediction.argmax().item()]
    output_sentence += " " + predicted_word
  print(output_sentence)
