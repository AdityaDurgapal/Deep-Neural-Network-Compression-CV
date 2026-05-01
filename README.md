# Deep Compression Pipeline

A practical implementation of neural network compression using **Pruning + Quantization + Huffman Coding**, inspired by the classic "Deep Compression" paper by Song Han et al.

This project compresses an MLP trained on MNIST (flattened images) while trying to keep accuracy as high as possible.

**Note:** No CNNs are used in this version — only fully connected layers.

---

## Repository Structure

- **`data/`**  
  - `dataset.py` — Loads MNIST, normalizes it, and creates DataLoaders.

- **`models/`**  
  - `mlp.py` — Defines the baseline MLP model (784 → 256 → 128 → 10).

- **`compression/`**  
  - `pruning.py` — Magnitude-based unstructured pruning.  
  - `quantization.py` — K-Means clustering for weight sharing.  
  - `huffman.py` — Huffman coding for final entropy compression.

- **`utils/`**  
  - `training.py` — Training and evaluation functions.

- **Root Files**
  - `config.py` — All hyperparameters (epochs, learning rate, pruning threshold, quantization bits, etc.)
  - `main.py` — Runs the complete pipeline and prints results.
  - `final_training.ipynb` — Main Jupyter notebook for running and visualizing everything.

---

## Pipeline Stages

1. **Baseline Training**  
   Train a normal 32-bit floating point MLP on flattened MNIST images.

2. **Pruning**  
   Remove 50% of the smallest weights using magnitude pruning.  
   Then fine-tune the sparse model for a few epochs to recover accuracy.

3. **Quantization**  
   Apply K-Means clustering on the remaining non-zero weights (currently 8-bit / 256 clusters).

4. **Huffman Encoding**  
   Use Huffman coding to further compress the quantized weights based on their frequency.

---

## Tasks

- **Task 1:** Load MNIST data and flatten 28×28 images into 784-dimensional vectors.
- **Task 2:** Build and train the baseline MLP model.
- **Task 3:** Implement magnitude-based pruning + fine-tuning.
- **Task 4:** Perform weight quantization using K-Means.
- **Task 5:** Run the full pipeline end-to-end.
- **Task 6:** Serialize the compressed model (sparsity + centroids) to `.npz` files.
- **Task 7:** Apply Huffman encoding on the serialized files and calculate final compression ratio.

---

## Current Status

- Baseline training and pruning are working.
- Quantization is implemented.
- Huffman coding is done on model weights.
- Memory footprint is calculated theoretically.
- **Next:** Proper `.npz` serialization + inference directly from compressed files.

---

## Goal

Create a lightweight, compressed classifier that uses significantly less storage and memory while maintaining decent accuracy.

The code is kept modular so we can easily add CNN support later.

---

## How to Run

```bash
# Run the full pipeline
python main.py

# Or use the notebook
jupyter notebook final_training.ipynb
