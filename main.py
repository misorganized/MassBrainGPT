import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the IMDB Movie Review Dataset
train_data = torch.load('imdb.pt')

# Create a TextCNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (K, embedding_dim)) for K in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch_size, num_filters, seq_len]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [batch_size, num_filters]
        x = torch.cat(x, 1)  # [batch_size, num_filters * len(kernel_sizes)]
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, 1]
        return x

# Instantiate the model
model = TextCNN(vocab_size=len(train_data.vocab),
                embedding_dim=100,
                kernel_sizes=[3, 4, 5],
                num_filters=100)

# Define a loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(10):
    for i, (text, label) in enumerate(train_data):
        # Forward pass
        logits = model(text)
        loss = loss_fn(logits.squeeze(), label)

        # Backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss and accuracy
        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch+1} | Iteration: {i+1} | Loss: {loss.item():.4f}')
