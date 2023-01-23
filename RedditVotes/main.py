import torch
import torch.nn as nn
import torch.optim as optim
import torch
import csv
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

with open('RedditVotes/comments.csv', 'r') as f:
    reader = csv.reader(f)
    preprocessed_comments = [row for row in reader]

comments = 30

preprocessed_comments = preprocessed_comments[0:comments]


# better model
class TextClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Convolutional layer
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # Gated Recurrent Unit (GRU) layer
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True,
                          bidirectional=True)

        # Fully-connected layer
        self.fc = nn.Linear(in_features=hidden_dim * 2, out_features=output_dim)

        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Embed the input tensor
        print(x)
        x = self.embedding(x.long())
        print(x)
        # Apply the convolutional layer
        x = self.conv(x.permute(0, 2, 1))
        print(x)
        x = torch.relu(x)
        print(x)

        # Apply the GRU layer
        x, _ = self.gru(x)
        print(x)

        # Apply the fully-connected layer
        x = self.fc(x[:, -1, :])
        print(x)

        # Apply the activation function
        x = self.act(x)
        print(x)

        return x


# Convert the comments to vectors
vectors = []


def top_words(texts, n):
    # Flatten the list of texts into a single list of words
    words = [word for text in texts for word in text[0].split()]

    # Count the frequency of each word
    word_counts = Counter(words)

    # Return the top N words
    return [word for word, count in word_counts.most_common(n)]


def one_hot_encode(text, vocabulary):
    # Initialize a vector of zeros
    vector = np.zeros(len(vocabulary))

    # Split the text into words
    words = text.split()

    # Set the corresponding element of the vector to 1 for each word in the text
    for word in words:
        if word in vocabulary:
            vector[vocabulary[word]] = 1

    return vector


vocab_amount = 256
top_words = top_words(preprocessed_comments, vocab_amount)
print(len(top_words))

vocabulary = {word: i for i, word in enumerate(top_words)}

for text, score in tqdm(preprocessed_comments):
    # Convert the text to a vector using one-hot encoding
    vector = one_hot_encode(text, vocabulary)
    vectors.append(vector)

# Convert the vectors to tensors

vectors = np.array(vectors)

tensors = torch.tensor(vectors).to(device)
# Get the scores
scores = [score for _, score in tqdm(preprocessed_comments)]
scores = [int(score) for score in scores]

# Convert the scores to a tensor
targets = torch.tensor(scores).to(device)

# Create the model, criterion, and optimizer
# model = UpvotePredictor(input_size=tensors.size(1), hidden_size=64, output_size=1).to(device)

model = TextClassifier(len(top_words), 128, vocab_amount, 1).to(device)

if os.path.exists('./models/redditvotes-0.pth'):
    print("Loaded Model")
    model.load_state_dict(torch.load(f='./models/redditvotes-0.pth'))

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1.0)

losses = []
metric = []

max_epochs = 1

max_loss = 1

print("Starting Training")

# Training loop
for epoch in range(max_epochs):
    # Forward pass and compute the loss

    tensors = tensors.to(torch.float32)
    targets = targets.to(torch.float32)
    print(tensors, tensors.shape)
    output = model(tensors)
    print(output)
    targets = targets.to(torch.float32)
    output = output.view(-1)
    loss = criterion(output, targets)

    losses.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_id = random.randint(0, comments)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {loss:.4f}, Input: {targets[test_id]}, Output: {output[test_id]}')
    if epoch % 50 == 0:
        # 1. Create models directory
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        # 2. Create model save path
        MODEL_NAME = "redditvotes-0.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        # 3. Save the model state dict
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model.state_dict(),  # only saving the state_dict() only saves the models learned parameters
                   f=MODEL_SAVE_PATH)

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "redditvotes-0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),  # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
