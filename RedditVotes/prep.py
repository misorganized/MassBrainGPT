import json
import string
import re
import torch
from tqdm import tqdm

# Load the Reddit Comment dataset
comments = []
with open('dataset/RCD/reddit_comments.json', 'r') as f:
    for line in f:
        comment = json.loads(line)
        comments.append(comment)

# Filter out unwanted data
comments = [comment for comment in tqdm(comments) if comment['score'] > 0 and 'spam' not in comment['body'].lower()]

# Tokenize the text
def tokenize(text):
    # Remove punctuation and make the text lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Split the text into tokens
    tokens = re.split(r'\s+', text)
    # Remove tokens that are too short or too long
    tokens = [token for token in tokens if 3 <= len(token) <= 15]
    return tokens

# Convert the text to a numerical representation using count vectors
def vectorize(tokens, vocabulary):
    # Create a count vector for the document
    vector = torch.zeros(len(vocabulary))
    for token in tokens:
        if token in vocabulary:
            vector[vocabulary[token]] += 1
    return vector

# Preprocess the comments
preprocessed_comments = []
vocabulary = {}

for comment in tqdm(comments):
    tokens = tokenize(comment['body'])
    vector = vectorize(tokens, vocabulary)
    preprocessed_comments.append((tokens, vector))

# Update the vocabulary with the new tokens
for token in set(tokens):
    if token not in vocabulary:
        vocabulary[token] = len(vocabulary)

# Save the vocabulary to a file
with open('dataset/RCD/vocabulary.json', 'w') as f:
    json.dump(vocabulary, f)

print(type(preprocessed_comments))
print("done")