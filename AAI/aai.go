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