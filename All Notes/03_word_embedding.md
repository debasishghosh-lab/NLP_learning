# Day 4: Word Embeddings, Word2Vec, CBOW, and Skip-Gram

## Overview

Traditional NLP techniques such as:

* One-Hot Encoding
* Bag of Words (BoW)
* TF-IDF

have several limitations:

* Sparse Matrices
* High Dimensionality
* No Semantic Understanding
* Out-of-Vocabulary (OOV) Problems

To overcome these issues, Word Embeddings were introduced.

---

# What is Word Embedding?

Word Embedding is a technique that converts words into dense numerical vectors while preserving semantic relationships between words.

---

# Types of Word Embedding Techniques

## 1. Count/Frequency-Based Methods

* One-Hot Encoding
* Bag of Words (BoW)
* TF-IDF

---

## 2. Deep Learning-Based Methods

* Word2Vec
* GloVe
* FastText

Word2Vec is one of the most popular embedding techniques.

---

# Why Do We Need Word Embeddings?

## Problems with Traditional Methods

### Sparse Matrices

Most values become zero.

```text
[0,0,0,0,1,0,0,0]
```

---

### High Dimensionality

Vocabulary size increases rapidly.

More words → More dimensions.

---

### No Semantic Meaning

BoW and TF-IDF cannot understand:

```text
King ≈ Queen
Good ≈ Excellent
Car ≈ Automobile
```

---

# Advantages of Word Embeddings

## Dense Representation

Instead of sparse vectors:

```text
[0,0,0,0,1,0,0]
```

we get:

```text
[0.72, 0.14, -0.31, 0.88]
```

---

## Reduced Dimensionality

Typically:

* 100 Dimensions
* 200 Dimensions
* 300 Dimensions

represent the entire vocabulary.

---

## Semantic Relationships Preserved

Words with similar meanings have similar vectors.

Example:

```text
Good ≈ Excellent
King ≈ Queen
Boy ≈ Girl
```

---

# Word2Vec

Word2Vec is a neural-network-based technique that learns vector representations of words.

---

# Key Idea Behind Word2Vec

Each word is represented as a dense vector.

Example:

```text
King  → [0.91, 0.85, 0.72]
Queen → [0.90, 0.84, 0.74]
```

Because the vectors are similar:

```text
King ≈ Queen
```

semantic meaning is preserved.

---

# Famous Word2Vec Relationship

```text
King - Man + Woman ≈ Queen
```

This demonstrates how semantic information is captured by embeddings.

---

# Cosine Similarity

Used to measure similarity between two word vectors.

## Formula

```text
Similarity = cos(θ)
```

---

## Interpretation

| Cosine Value | Meaning         |
| ------------ | --------------- |
| 1            | Exactly Similar |
| 0            | Unrelated       |
| -1           | Opposite        |

---

## Examples

```text
King ↔ Queen
```

High similarity.

```text
Cricket ↔ Sports
```

Moderate similarity.

```text
King ↔ Mango
```

Low similarity.

---

# Types of Word2Vec

## 1. CBOW

Continuous Bag of Words

## 2. Skip-Gram

These are the two architectures used to train Word2Vec models.

---

# Continuous Bag of Words (CBOW)

## Goal

Predict the center word using surrounding context words.

---

## Example Sentence

```text
Krish Channel Is Related To Data Science
```

---

## Window Size = 5

```text
Krish Channel [Is] Related To
```

Target Word:

```text
Is
```

Context Words:

```text
Krish
Channel
Related
To
```

---

## Training Sample

### Input

```text
Krish
Channel
Related
To
```

### Output

```text
Is
```

---

## Next Training Example

```text
Channel Is [Related] To Data
```

### Input

```text
Channel
Is
To
Data
```

### Output

```text
Related
```

---

# CBOW Summary

```text
Context Words
      ↓
Predict
      ↓
Target Word
```

---

# Skip-Gram

CBOW works:

```text
Context → Target
```

Skip-Gram works:

```text
Target → Context
```

---

## Example

Sentence:

```text
Krish Channel Is Related To Data Science
```

Target Word:

```text
Is
```

---

### Input

```text
Is
```

### Outputs

```text
Krish
Channel
Related
To
```

---

# Skip-Gram Summary

```text
Target Word
      ↓
Predict
      ↓
Context Words
```

---

# CBOW vs Skip-Gram

| CBOW                    | Skip-Gram           |
| ----------------------- | ------------------- |
| Context → Target        | Target → Context    |
| Faster                  | Slower              |
| Good for Frequent Words | Good for Rare Words |
| Smaller Datasets        | Larger Datasets     |

---

# Word2Vec Architecture

Basic Architecture:

```text
Input Layer
      ↓
Embedding Layer
      ↓
Hidden Layer
      ↓
Output Layer
```

---

# Embedding Layer

The most important layer.

Purpose:

```text
Text → Dense Vectors
```

The weights learned in this layer become the final word embeddings.

---

# Pretrained Word2Vec Models

Google trained Word2Vec on approximately:

```text
3 Billion Words
```

and released:

```text
Google News Word2Vec
```

which generates:

```text
300-Dimensional Word Vectors
```

---

# Gensim Library

Most commonly used library for Word2Vec.

---

## Import

```python
import gensim

from gensim.models import Word2Vec
```

---

# Loading Pretrained Model

```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors.bin",
    binary=True
)
```

---

# Find Similar Words

```python
model.most_similar("king")
```

Output:

```text
queen
prince
monarch
princess
```

---

# Similarity Between Words

```python
model.similarity(
    "cricket",
    "sports"
)
```

Returns cosine similarity score.

---

# Benefits of Word2Vec

✅ Captures semantic meaning

✅ Dense vectors

✅ Reduced dimensionality

✅ Better than BoW and TF-IDF

✅ Useful for Deep Learning NLP

---

# Limitations of Word2Vec

❌ OOV Problem

❌ Same word gets same embedding in every context

Example:

```text
Bank (River)
Bank (Money)
```

Both receive the same vector.

---

# Interview Questions

## What is Word Embedding?

A technique that converts words into dense vectors while preserving semantic relationships.

---

## Why is Word2Vec better than TF-IDF?

Word2Vec captures semantic meaning and produces dense vectors.

---

## Difference Between CBOW and Skip-Gram?

CBOW:

```text
Context → Target
```

Skip-Gram:

```text
Target → Context
```

---

## What is Cosine Similarity?

Measures similarity between two vectors.

---

## What is the Role of the Embedding Layer?

Converts words into dense numerical vectors.

---

# Quick Revision

```text
One-Hot Encoding
        ↓
Bag of Words
        ↓
TF-IDF
        ↓
Word Embeddings
        ↓
Word2Vec
      ↙     ↘
   CBOW   Skip-Gram
        ↓
Deep Learning NLP
```

---

# Next Topics

* Training Word2Vec from Scratch
* Average Word2Vec
* GloVe
* FastText
* RNN
* LSTM
* GRU

```
```
