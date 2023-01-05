import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import csv
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))


with open('RedditVotes/comments.csv', 'r') as f:
    reader = csv.reader(f)
    preprocessed_comments = [row for row in reader]
    
preprocessed_comments = preprocessed_comments[0:100000]

# Define the model
class UpvotePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(UpvotePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc6(x)
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


top_words = top_words(preprocessed_comments, 2000)

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
model = UpvotePredictor(input_size=tensors.size(1), hidden_size=64, output_size=1).to(device)
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

    output = model(tensors)
    targets = targets.to(torch.float32)
    output = output.view(-1)
    print(output[0:10], targets[0:10])
    loss = criterion(output, targets)

    losses.append(loss.item())



    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: loss = {loss:.4f}')
