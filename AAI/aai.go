package aai

import "github.com/gin-gonic/gin"

var code1 string = `
import cv2
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from PIL import Image

def display_image(title, image):
    """Helper function to display images."""
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def compute_dft(image):
    """Compute and display the DFT magnitude spectrum."""
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    display_image("DFT Magnitude Spectrum", magnitude_spectrum)


def apply_dct(image_array):
    """Apply block-wise DCT to the image."""
    block_size = 8  # Typically 8x8 blocks are used for DCT
    h, w = image_array.shape  # Get height and width

    dct_image = np.zeros_like(image_array)

    # Apply DCT block by block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image_array[i:i + block_size, j:j + block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue  # Skip blocks that are incomplete
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_image[i:i + block_size, j:j + block_size] = dct_block

    return dct_image


def compute_dct(image):
    """Compute and display the DCT magnitude spectrum."""
    # Ensure grayscale image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image to float range [0, 1]
    image = np.float32(image) / 255.0

    # Apply DCT
    dct_image = apply_dct(image)

    # Visualize the DCT magnitude spectrum
    magnitude_spectrum = np.log(np.abs(dct_image) + 1)
    display_image("DCT Magnitude Spectrum", magnitude_spectrum)
def apply_laplacian(image):
    """Apply the Laplacian filter to the image."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    display_image("Laplacian Filtered Image", laplacian)


def apply_weighted_average(image):
    """Apply a weighted average (smoothing) filter to the image."""
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv2.filter2D(image, -1, kernel)
    display_image("Weighted Average Filtered Image", smoothed)
image_path = "./data/veg.jpg"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
display_image("Original Image", image)

# Perform tasks
compute_dft(image)
compute_dct(image)
apply_laplacian(image)
apply_weighted_average(image)
  `

var code2 string = `
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from skimage.filters import sobel

# Helper function to display images


def display_image(title, image):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Function to perform the Walsh-Hadamard transform


def compute_walsh_hadamard(image):
    """Compute and display the Walsh-Hadamard transform."""
    h_size = 1
    while h_size < max(image.shape):
        h_size *= 2
    h_matrix = hadamard(h_size)
    padded_image = np.zeros((h_size, h_size))
    padded_image[:image.shape[0], :image.shape[1]] = image
    transformed = np.dot(np.dot(h_matrix, padded_image), h_matrix)
    magnitude_spectrum = np.log(np.abs(transformed) + 1)
    display_image("Walsh-Hadamard Transform", magnitude_spectrum)

# Function to perform the Slant transform


def compute_slant_transform(image):
    """Compute and display the Slant transform."""
    # Define a Slant transform matrix
    def slant_matrix(size):
        S = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == 0:
                    S[i, j] = 1 / np.sqrt(size)
                else:
                    S[i, j] = np.sqrt(
                        2 / size) * np.cos((np.pi * i * (2 * j + 1)) / (2 * size))
        return S

    h, w = image.shape
    size = max(h, w)
    S = slant_matrix(size)

    # Pad the image to match the Slant matrix size
    padded_image = np.zeros((size, size))
    padded_image[:h, :w] = image

    # Apply the Slant transform
    transformed = np.dot(np.dot(S, padded_image), S.T)
    magnitude_spectrum = np.log(np.abs(transformed) + 1)
    display_image("Slant Transform", magnitude_spectrum)

# Function to apply the Sobel filter


def apply_sobel_filter(image):
    """Apply and display the Sobel filter."""
    sobel_filtered = sobel(image)
    display_image("Sobel Filtered Image", sobel_filtered)

# Function to apply a composite masking filter


def apply_composite_mask_filter(image):
    """Apply and display a composite masking filter."""
    kernel = np.array([[1, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 1]])
    composite_filtered = cv2.filter2D(image, -1, kernel)
    display_image("Composite Mask Filtered Image", composite_filtered)
image_path = "./data/veg.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform tasks
compute_walsh_hadamard(image)
compute_slant_transform(image)
apply_sobel_filter(image)
apply_composite_mask_filter(image)
`

var code3 string = `
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio = './data/datasets/Lab-Exam/audio/sample.mp3'
audio_np_array, sample_rate = librosa.load(audio, sr=None)

amplitude_envelope = np.abs(librosa.effects.preemphasis(
    audio_np_array))  # preemphasis is a filter
librosa.display.waveshow(amplitude_envelope, sr=sample_rate)

loudness_db = librosa.amplitude_to_db(amplitude_envelope, ref=np.max)
librosa.display.waveshow(loudness_db, sr=sample_rate)

mfccs = librosa.feature.mfcc(y=audio_np_array, sr=sample_rate, n_mfcc=13)
librosa.display.specshow(mfccs, sr=sample_rate)
plt.colorbar()

S = librosa.feature.melspectrogram(y=audio_np_array, sr=sample_rate)
spectral_centroid = librosa.feature.spectral_centroid(S=S)
librosa.display.specshow(librosa.power_to_db(
    S, ref=np.max), x_axis='time', y_axis='mel')
plt.semilogy(spectral_centroid, label='Spectral Centroid', color='b')
plt.colorbar()
`

var code4 string = `
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio = './data/datasets/Lab-Exam/audio/sample.mp3'
audio_np_array, sample_rate = librosa.load(audio, sr=None)

chroma = librosa.feature.chroma_stft(y=audio_np_array, sr=sample_rate)
librosa.display.specshow(chroma, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title('Chroma STFT')
plt.tight_layout()
plt.show()

spectral_contrast = librosa.feature.spectral_contrast(
    y=audio_np_array, sr=sample_rate)
librosa.display.specshow(spectral_contrast, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title('Spectral Contrast')
plt.tight_layout()
plt.show()

pitch_sal = librosa.piptrack(y=audio_np_array, sr=sample_rate)
pitch_values = np.argmax(pitch_sal, axis=1)
librosa.display.specshow(pitch_values, sr=sample_rate)
plt.colorbar()
plt.title('Pitch Salience')
plt.tight_layout()
plt.show()
`

var code5 string = `
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def dfs_search(graph, start, target, visited=None):
    if visited is None:
        visited = np.zeros(graph.shape[0], dtype=bool)

    # Mark the current node as visited
    visited[start] = True
    print(f"Visited node {start}")

    # If the target node is found, stop the search
    if start == target:
        print(f"Target node {target} found!")
        return True  # Found the target

    # Recursively visit all adjacent, unvisited nodes
    for neighbor, is_connected in enumerate(graph[start]):
        if is_connected and not visited[neighbor]:
            if dfs_search(graph, neighbor, target, visited):
                return True  # Propagate success up the call stack

    return False  # Target not found in this branch


# Example graph as an adjacency matrix (6 nodes)
graph = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [1, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

# DFS search with path recording


def dfs_find_path(graph, start, target, visited=None, path=None):
    if visited is None:
        visited = np.zeros(graph.shape[0], dtype=bool)
    if path is None:
        path = []

    # Mark the current node as visited and add it to the path
    visited[start] = True
    path.append(start)

    # If the target node is found, return the path
    if start == target:
        return path

    # Recursively visit all adjacent, unvisited nodes
    for neighbor, is_connected in enumerate(graph[start]):
        if is_connected and not visited[neighbor]:
            result = dfs_find_path(graph, neighbor, target, visited, path)
            if result is not None:
                return result  # Return the path if target is found

    # Backtrack if no path to target found in this branch
    path.pop()
    return None


# Example usage
path_to_target = dfs_find_path(graph, start=0, target=3)
if path_to_target:
    print(f"Path to target: {path_to_target}")
else:
    print("Target node not found.")


def visualize_graph(graph, start, path=None):
    # Create a graph
    G = nx.Graph()

    # Add edges to the graph from the adjacency matrix
    num_nodes = graph.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if graph[i, j] == 1:  # There's an edge between node i and j
                G.add_edge(i, j)

    # Define node colors, making the starting node red and others blue
    node_colors = ['red' if node ==
                   start else 'lightblue' for node in range(num_nodes)]

    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=500, font_size=10, font_color='black', edge_color='gray')

    # Highlight the DFS path if it exists
    if path:
        # Create a list of edges in the path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               width=2, edge_color='blue')

    # Display the plot
    plt.title("Graph Visualization with DFS Path")
    plt.show()


# Call the function to visualize the graph with the DFS path highlighted
visualize_graph(graph, start=0, path=path_to_target)
`
var code6 string = `
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from heapq import heappop, heappush


# Heuristic function (for simplicity, we use straight-line distance)
def heuristic(node, target, positions):
    pos1 = positions[node]
    pos2 = positions[target]
    # Euclidean distance
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

# A* search algorithm
def a_star_search(graph, start, target, positions):
    num_nodes = graph.shape[0]
    open_set = []
    heappush(open_set, (0, start))  # (total_cost, node)

    came_from = {start: None}
    g_score = {node: float('inf') for node in range(num_nodes)}
    g_score[start] = 0

    f_score = {node: float('inf') for node in range(num_nodes)}
    f_score[start] = heuristic(start, target, positions)

    while open_set:
        _, current = heappop(open_set)

        if current == target:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        for neighbor, weight in enumerate(graph[current]):
            if weight > 0:  # If there is an edge
                tentative_g_score = g_score[current] + weight

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target, positions)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


# Visualization function with highlighted path, start, and target nodes
def visualize_a_star(graph, positions, path, start, target):
    G = nx.Graph()

    num_nodes = graph.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if graph[i, j] > 0:
                G.add_edge(i, j, weight=graph[i, j])

    pos = positions

    # Draw nodes
    node_colors = ['green' if node == target else 'red' if node ==
                   start else 'lightblue' for node in range(num_nodes)]
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=500, font_color='black', edge_color='gray')

    # Highlight the path found
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               width=2, edge_color='blue')

    # Show plot
    plt.title("A* Search Visualization")
    plt.show()


# Example graph as an adjacency matrix (weighted graph, 6 nodes)
graph = np.array([[0, 1, 4, 0, 0, 0],
                  [1, 0, 4, 2, 7, 0],
                  [4, 4, 0, 3, 5, 0],
                  [0, 2, 3, 0, 4, 6],
                  [0, 7, 5, 4, 0, 7],
                  [0, 0, 0, 6, 7, 0]])

# Node positions (for visualization)
positions = {0: (0, 0), 1: (1, 1), 2: (1, -1),
             3: (2, 0), 4: (3, 1), 5: (3, -1)}

# Run A* search from node 0 to node 5
start_node = 0
target_node = 5
path = a_star_search(graph, start=start_node,
                     target=target_node, positions=positions)

if path:
    print(f"Path found: {path}")
else:
    print("No path found.")

# Visualize the graph and the path found by A*
visualize_a_star(graph, positions, path, start=start_node, target=target_node)
`
var code7 string = `
!pip install PyPDF2 nltk textblob matplotlib pandas
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Preprocessing Function
def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize the input text."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Stopword Removal and Lemmatization
    filtered_tokens = [lemmatizer.lemmatize(
        word) for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(filtered_tokens), filtered_tokens

# Sentiment Analysis Function


def analyze_sentiment(text):
    """Analyze the sentiment of the text and return sentiment score and label."""
    blob = TextBlob(text)
    # Sentiment polarity: -1 (negative) to +1 (positive)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0.1:
        sentiment_label = "Positive"
    elif sentiment_score < -0.1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return sentiment_score, sentiment_label

# Specify the path to your PDF
pdf_path = "./data/datasets/Lab-Exam/NLP/Nlp4.pdf"  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(pdf_path)

# Split extracted text into sentences
sentences = nltk.sent_tokenize(extracted_text)
results = []

print("\nProcessing Text from PDF...\n")

for sentence in sentences:
    preprocessed, tokens = preprocess_text(sentence)
    sentiment_score, sentiment_label = analyze_sentiment(preprocessed)
    results.append({
        "Original": sentence,
        "Preprocessed": preprocessed,
        "Tokens": tokens,
        "Sentiment Score": sentiment_score,
        "Sentiment Label": sentiment_label
    })

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Display results
df.head()

# Display sentiment distribution
sentiment_distribution = Counter(df['Sentiment Label'])

print("\nSentiment Distribution:")
for label, count in sentiment_distribution.items():
    print(f"{label}: {count}")

# Plot the sentiment distribution
plt.bar(sentiment_distribution.keys(), sentiment_distribution.values())
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
`
var code8 string = `
!pip install contractions
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import re
import contractions

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
# Preprocessing Functions


def expand_contractions(text):
    """Expand contractions using the contractions library."""
    return contractions.fix(text)


def remove_special_characters(text):
    """Remove special characters from the text."""
    return re.sub(r'[^a-zA-Z\s]', '', text)


def tokenize_text(text, level="word"):
    """Tokenize text into words or sentences."""
    if level == "word":
        return word_tokenize(text)
    elif level == "sentence":
        return sent_tokenize(text)
    else:
        raise ValueError("Level must be 'word' or 'sentence'.")


def remove_stopwords(tokens):
    """Remove stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]


def pos_tagging(tokens):
    """Perform Part-of-Speech tagging on tokens."""
    return pos_tag(tokens)


# Example Text
text_q1 = "I'm loving the new updates! It's amazing to see progress. Can't wait for more."

# Pipeline
print("Original Text:", text_q1)

# Step 1: Expand Contractions
expanded_text = expand_contractions(text_q1)
print("After Expanding Contractions:", expanded_text)

# Step 2: Remove Special Characters
cleaned_text = remove_special_characters(expanded_text)
print("After Removing Special Characters:", cleaned_text)

# Step 3: Tokenization
tokens = tokenize_text(cleaned_text)
print("Tokens:", tokens)

# Step 4: Remove Stopwords
filtered_tokens = remove_stopwords(tokens)
print("Filtered Tokens:", filtered_tokens)

# Step 5: POS Tagging
pos_tags = pos_tagging(filtered_tokens)
print("POS Tags:", pos_tags)
`

var code9 string = `
# install via python -m spacy download en_core_web_sm


import nltk
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load SpaCy for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Preprocessing Functions


def expand_contractions(text):
    """Expand contractions using the contractions library."""
    return contractions.fix(text)


def correct_spelling(text):
    """Correct spelling errors in the text."""
    return str(TextBlob(text).correct())


def apply_stemming(tokens):
    """Apply stemming to tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def apply_lemmatization(tokens):
    """Apply lemmatization to tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def extract_text_features(text):
    """Extract features such as word count, sentence count, and average word length."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / \
        word_count if word_count > 0 else 0
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Average Word Length": avg_word_length
    }


def named_entity_recognition(text):
    """Perform Named Entity Recognition to identify entities like names, dates, locations."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Example Text
text_q2 = "I'm traveling to New York in September. Can't wait to visit the Empire State Building!"

# Pipeline
print("Original Text:", text_q2)

# Step 1: Expand Contractions
expanded_text = expand_contractions(text_q2)
print("After Expanding Contractions:", expanded_text)

# Step 2: Spelling Correction
corrected_text = correct_spelling(expanded_text)
print("After Spelling Correction:", corrected_text)

# Step 3: Tokenization
tokens = word_tokenize(corrected_text)
print("Tokens:", tokens)

# Step 4: Stemming and Lemmatization
stemmed_tokens = apply_stemming(tokens)
lemmatized_tokens = apply_lemmatization(tokens)
print("Stemmed Tokens:", stemmed_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)

# Step 5: Extract Text Features
features = extract_text_features(corrected_text)
print("Text Features:", features)

# Step 6: Named Entity Recognition
entities = named_entity_recognition(corrected_text)
print("Named Entities:", entities)
`
var code10 string = `
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess text data


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def preprocess_text(text):
    # Create a mapping from characters to integers and vice versa
    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return chars, char2idx, idx2char

# Step 2: Define the dataset


class TextDataset(Dataset):
    def __init__(self, text, char2idx, seq_length):
        self.text = text
        self.char2idx = char2idx
        self.seq_length = seq_length
        self.data = [char2idx[c] for c in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# Step 3: Define the RNN-based generative model


class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMGenerator, self).__init__()

        # vocab size - number of unique meanings / words. 
        # Embed Size - dimension of embedding per word
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))

# Step 4: Training function
def train_model(model, dataset, batch_size, epochs, lr, char2idx):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)  # Dynamically determine the batch size
            # Reinitialize hidden state for each batch
            hidden = model.init_hidden(batch_size)

            optimizer.zero_grad()
            # Detach hidden states to prevent backprop through time
            hidden = tuple([h.detach() for h in hidden])
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, len(char2idx)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training complete!")
    return model


# Step 5: Generate sequences using the trained model

def generate_sequence(model, start_text, char2idx, idx2char, seq_length, gen_length):
    model.eval()

    # Ensure all characters in the start_text are in the char2idx vocabulary
    start_text = ''.join([c if c in char2idx else '' for c in start_text])
    if not start_text:
        raise ValueError(
            "The start_text does not contain any valid characters from the training vocabulary.")

    input_seq = torch.tensor(
        [char2idx[c] for c in start_text], dtype=torch.long
    ).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    generated_text = start_text

    for _ in range(gen_length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output[0, -1], dim=0).cpu().numpy()
            char_idx = np.random.choice(len(idx2char), p=probs)
            generated_text += idx2char[char_idx]
            input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return generated_text
# Load the text file
text_file = "./data/datasets/Lab-Exam/RNN/RNN sample.txt"  # Replace with your text file path
text = load_text(text_file)

# Preprocess the text
seq_length = 100  # Sequence length for training
chars, char2idx, idx2char = preprocess_text(text)

# Create dataset
dataset = TextDataset(text, char2idx, seq_length)

# Model parameters
vocab_size = len(chars)
embed_size = 128
hidden_size = 256
num_layers = 2
batch_size = 64
epochs = 20
lr = 0.001

# Initialize and train the model
model = LSTMGenerator(vocab_size, embed_size,
                        hidden_size, num_layers).to(device)
model = train_model(model, dataset, batch_size,epochs, lr, char2idx)

# Generate new sequences
start_text = "Once upon a time"  # Provide a starting text for generation
gen_length = 200  # Number of characters to generate
generated_text = generate_sequence(
    model, start_text, char2idx, idx2char, seq_length, gen_length)
print("\nGenerated Text:")
print(generated_text)
`
var code11 string = `
# Python3 implementation of the above approach
from random import randint

INT_MAX = 2147483647
# Number of cities in TSP
V = 5

# Names of the cities
GENES = "ABCDE"

# Starting Node Value
START = 0

# Initial population size for the algorithm
POP_SIZE = 10

# Structure of a GNOME
# defines the path traversed
# by the salesman while the fitness value
# of the path is stored in an integer


class individual:
    def __init__(self) -> None:
        self.gnome = ""
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


# Function to return a random number
# from start and end
def rand_num(start, end):
    return randint(start, end-1)


# Function to check if the character
# has already occurred in the string
def repeat(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            return True

    return False


# Function to return a mutated GNOME
# Mutated GNOME is a string
# with a random interchange
# of two genes to create variation in species
def mutatedGene(gnome):
    gnome = list(gnome)
    while True:
        r = rand_num(1, V)
        r1 = rand_num(1, V)
        if r1 != r:
            temp = gnome[r]
            gnome[r] = gnome[r1]
            gnome[r1] = temp
            break
    return ''.join(gnome)


# Function to return a valid GNOME string
# required to create the population
def create_gnome():
    gnome = "0"
    while True:
        if len(gnome) == V:
            gnome += gnome[0]
            break

        temp = rand_num(1, V)
        if not repeat(gnome, chr(temp + 48)):
            gnome += chr(temp + 48)

    return gnome


# Function to return the fitness value of a gnome.
# The fitness value is the path length
# of the path represented by the GNOME.
def cal_fitness(gnome):
    mp = [
        [0, 2, INT_MAX, 12, 5],
        [2, 0, 4, 8, INT_MAX],
        [INT_MAX, 4, 0, 3, 3],
        [12, 8, 3, 0, 10],
        [5, INT_MAX, 3, 10, 0],
    ]
    f = 0
    for i in range(len(gnome) - 1):
        if mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48] == INT_MAX:
            return INT_MAX
        f += mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48]

    return f


# Function to return the updated value
# of the cooling element.
def cooldown(temp):
    return (90 * temp) / 100


# Comparator for GNOME struct.
# def lessthan(individual t1,
#               individual t2)
# :
#     return t1.fitness < t2.fitness


# Utility function for TSP problem.
def TSPUtil(mp):
    # Generation Number
    gen = 1
    # Number of Gene Iterations
    gen_thres = 5

    population = []
    temp = individual()

    # Populating the GNOME pool.
    for i in range(POP_SIZE):
        temp.gnome = create_gnome()
        temp.fitness = cal_fitness(temp.gnome)
        population.append(temp)

    print("\nInitial population: \nGNOME     FITNESS VALUE\n")
    for i in range(POP_SIZE):
        print(population[i].gnome, population[i].fitness)
    print()

    found = False
    temperature = 10000

    # Iteration to perform
    # population crossing and gene mutation.
    while temperature > 1000 and gen <= gen_thres:
        population.sort()
        print("\nCurrent temp: ", temperature)
        new_population = []

        for i in range(POP_SIZE):
            p1 = population[i]

            while True:
                new_g = mutatedGene(p1.gnome)
                new_gnome = individual()
                new_gnome.gnome = new_g
                new_gnome.fitness = cal_fitness(new_gnome.gnome)

                if new_gnome.fitness <= population[i].fitness:
                    new_population.append(new_gnome)
                    break

                else:

                    # Accepting the rejected children at
                    # a possible probability above threshold.
                    prob = pow(
                        2.7,
                        -1
                        * (
                            (float)(new_gnome.fitness - population[i].fitness)
                            / temperature
                        ),
                    )
                    if prob > 0.5:
                        new_population.append(new_gnome)
                        break

        temperature = cooldown(temperature)
        population = new_population
        print("Generation", gen)
        print("GNOME     FITNESS VALUE")

        for i in range(POP_SIZE):
            print(population[i].gnome, population[i].fitness)
        gen += 1


mp = [
    [0, 2, INT_MAX, 12, 5],
    [2, 0, 4, 8, INT_MAX],
    [INT_MAX, 4, 0, 3, 3],
    [12, 8, 3, 0, 10],
    [5, INT_MAX, 3, 10, 0],
]
TSPUtil(mp)
`
var code12 string = `
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset


def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True,
                             transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Function to plot generated images
def plot_generated_images(generator, epoch, noise_dim, examples=16, dim=(4, 4)):
    generator.eval()
    noise = torch.randn(examples, noise_dim).to(device)
    fake_images = generator(noise).detach().cpu()
    fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]

    fig, axes = plt.subplots(dim[0], dim[1], figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i][0], cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Generated Images - Epoch {epoch + 1}")
    plt.show()

# Generator model
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # Output scaled to [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Discriminator model


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))
# Training the GAN
def train_gan(generator, discriminator, data_loader, epochs=10, noise_dim=100, lr=0.0002):
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)

            # Move real images to GPU
            real_images = real_images.to(device)

            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise)

            d_loss_real = criterion(discriminator(real_images), real_labels)
            d_loss_fake = criterion(discriminator(
                fake_images.detach()), fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            noise = torch.randn(batch_size, noise_dim).to(device)
            g_loss = criterion(discriminator(fake_images), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        # Plot generated images at the end of each epoch
        print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        plot_generated_images(generator, epoch, noise_dim)


# Initialize models and data loader
noise_dim = 100
batch_size = 128
epochs = 10

data_loader = load_data(batch_size)
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Train the GAN
train_gan(generator, discriminator,
          data_loader, epochs, noise_dim)
`

func Lab1Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code1))
}
func Lab2Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code2))
}
func Lab3Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code3))
}
func Lab4Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code4))
}
func Lab5Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code5))
}
func Lab6Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code6))
}
func Lab7Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code7))
}
func Lab8Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code8))
}
func Lab9Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code9))
}
func Lab10Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code10))
}
func Lab11Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code11))
}
func Lab12Code(c *gin.Context) {
	c.Data(200, "text/plain", []byte(code12))
}
