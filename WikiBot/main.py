import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Define the database loader function
def load_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT i FROM list")
    text = [row[0] for row in cursor.fetchall()]
    conn.close()
    return text


# Define the vectorizer function
def vectorize(text, vocab):
    word2idx = {word: i for i, word in enumerate(vocab)}
    text_vector = [[word2idx[word] for word in sentence.split() if word in vocab] for sentence in text]
    text_vector = list(filter(None, text_vector))
    print(f"Text Vectors: {len(text_vector)}")
    return text_vector


# Define the embedder function
def embed(text_vector, embedding_matrix):
    embedded_text = [[embedding_matrix[idx] for idx in sentence] for sentence in text_vector]
    return embedded_text


# Define the tensorizer function
def tensorize(embedded_text):
    embedded_text = np.array([elem for sublist in embedded_text for elem in sublist], dtype=np.float32)
    shape = embedded_text.shape
    embedded_text = embedded_text.reshape(-1, shape[-1])
    return torch.tensor(embedded_text)


# Define the model class
class Text2TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Text2TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = x.long()
        print(x, x.shape)
        x = self.embedding(x)
        print(x.shape)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# Define the training loop
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    for i, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, vocab_size), data.view(-1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            torch.save(model.state_dict(), 'model.pt')


# Define the test function
def test(model, test_text, word2idx, device):
    test_vector = [word2idx[word] for word in test_text.split()]
    test_tensor = torch.tensor(test_vector).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(test_tensor)
    output_text = ' '.join([vocab[idx] for idx in output.argmax(2)[0].tolist()])
    return output_text


# Define the path to the database
db_path = 'Wikipedia.db'

# Load the text from the database
text = load_db(db_path)

# Create a vocabulary from the text
vocab = set(' '.join(text))
vocab_size = len(vocab)
print("Vocab Size: {}".format(vocab_size))

# Create a mapping from word to index
word2idx = {word: i for i, word in enumerate(vocab)}

# Vectorize the text
text_vector = vectorize(text, vocab)


# Define the embedding size
embedding_size = 50

# Create an embedding matrix
embedding_matrix = np.random.randn(vocab_size, embedding_size)

# Embed the vectorized text
embedded_text = embed(text_vector, embedding_matrix)

# Convert the embedded text to a Pytorch tensor
data = tensorize(embedded_text)

# Create a DataLoader to load the data in batches
batch_size = 8
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Define the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
hidden_size = 256
model = Text2TextModel(vocab_size, embedding_size, hidden_size)
model.to(device)

# Define the criterion, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
train(model, data_loader, criterion, optimizer, device)

# Test the model
test_text = "This is a positive sentence."
print(test(model, test_text, word2idx, device))
