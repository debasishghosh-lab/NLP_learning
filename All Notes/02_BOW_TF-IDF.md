# Day 3: TF-IDF and Advanced Text Vectorization

## Overview

In this section, we explored the limitations of Bag of Words (BoW) and learned how TF-IDF improves text representation by assigning importance scores to words.

---

# Recap: Bag of Words (BoW)

Bag of Words converts text into numerical vectors using word frequency.

### Example

Sentence 1:

```text
good boy
```

Sentence 2:

```text
good girl
```

Sentence 3:

```text
boy girl good
```

### Vocabulary

```text
good
boy
girl
```

### BoW Representation

| Sentence | Good | Boy | Girl |
| -------- | ---- | --- | ---- |
| S1       | 1    | 1   | 0    |
| S2       | 1    | 0   | 1    |
| S3       | 1    | 1   | 1    |

---

# Problems with Bag of Words

## 1. Sparse Matrix

Most values become zero.

### Example

```text
[0,0,0,0,1,0,0,0]
```

Large vocabularies create huge sparse matrices.

---

## 2. Out Of Vocabulary (OOV)

New words cannot be represented.

### Example

Training Vocabulary:

```text
cat
dog
```

Test Word:

```text
elephant
```

The model cannot represent "elephant".

---

## 3. Semantic Meaning Lost

### Example

```text
The food is good
The food is not good
```

These sentences have opposite meanings.

However, BoW creates nearly identical vectors.

Therefore:

```text
Semantic Information Lost
```

---

## 4. Equal Importance to All Words

BoW treats all words equally.

### Example

```text
good boy
```

Both words receive the same weight.

In reality, some words may be more important than others.

---

# TF-IDF

TF-IDF stands for:

```text
Term Frequency – Inverse Document Frequency
```

TF-IDF helps identify important words within documents.

Words appearing frequently in one document but rarely across all documents receive higher importance.

---

# Why TF-IDF?

Consider:

```text
Cat eats food
Bat eats food
Krish eats food
```

Words:

```text
eats
food
```

appear everywhere.

Words:

```text
cat
bat
krish
```

carry more unique information.

TF-IDF gives higher importance to rare words.

---

# Term Frequency (TF)

Measures how frequently a word appears in a document.

### Formula

```text
TF =
(Number of times word appears in document)
/
(Total number of words in document)
```

---

## Example

Sentence:

```text
good boy
```

For word:

```text
good
```

TF:

```text
1 / 2 = 0.5
```

---

# Inverse Document Frequency (IDF)

Measures how unique a word is across all documents.

### Formula

```text
IDF =
log(
Total Number of Documents
/
Number of Documents Containing Word
)
```

---

# Example

Documents:

```text
good boy
good girl
boy girl good
```

Total Documents:

```text
3
```

---

## IDF for "good"

Appears in:

```text
Document 1
Document 2
Document 3
```

IDF:

```text
log(3/3)
=
log(1)
=
0
```

Therefore:

```text
good → low importance
```

---

## IDF for "boy"

Appears in:

```text
Document 1
Document 3
```

IDF:

```text
log(3/2)
```

Positive value.

Therefore:

```text
boy → higher importance
```

---

## IDF for "girl"

Appears in:

```text
Document 2
Document 3
```

IDF:

```text
log(3/2)
```

Positive value.

---

# TF-IDF Formula

```text
TF-IDF = TF × IDF
```

---

# Interpretation

Words appearing in every document:

```text
good
the
is
are
```

receive low importance.

Words appearing in fewer documents:

```text
boy
girl
cat
krish
```

receive higher importance.

---

# Advantages of TF-IDF

### Captures Word Importance

Important words receive larger weights.

---

### Better than Bag of Words

Reduces the impact of extremely common words.

---

### Simple and Effective

Widely used in:

* Text Classification
* Information Retrieval
* Search Engines

---

# Limitations of TF-IDF

## Sparse Matrix Problem

Still produces sparse vectors.

---

## OOV Problem

Unknown words cannot be represented.

---

## No Deep Semantic Understanding

TF-IDF cannot understand relationships such as:

```text
King ↔ Queen
Car ↔ Automobile
```

---

# N-Grams

Used to preserve word order information.

---

## Unigram

```text
I love NLP
```

Output:

```text
I
love
NLP
```

---

## Bigram

```text
I love NLP
```

Output:

```text
I love
love NLP
```

---

## Trigram

```text
I love NLP Course
```

Output:

```text
I love NLP
love NLP Course
```

---

# CountVectorizer

Used for Bag of Words implementation.

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus)
```

---

# TF-IDF Vectorizer

Used for TF-IDF implementation.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(corpus)
```

---

# N-Gram Implementation

### Trigrams Only

```python
TfidfVectorizer(
    ngram_range=(3,3)
)
```

---

### Bigrams + Trigrams

```python
TfidfVectorizer(
    ngram_range=(2,3)
)
```

---

# Feature Selection using max_features

Keeps only the most frequent words.

### Example

```python
TfidfVectorizer(
    max_features=10
)
```

Only top 10 highest-frequency features are selected.

---

# Assignment Completed

Apply:

* Bag of Words
* TF-IDF
* N-Grams

on any text classification dataset from:

* Kaggle
* UCI Repository

---

# Key Interview Questions

## Why is TF-IDF better than Bag of Words?

Because TF-IDF assigns importance to words instead of treating all words equally.

---

## What does IDF measure?

Measures how rare or informative a word is across documents.

---

## What is OOV?

Out Of Vocabulary.

Words unseen during training cannot be represented.

---

## What does max_features do?

Selects only the top N most frequent features.

---

## Difference Between BoW and TF-IDF

| Bag of Words    | TF-IDF              |
| --------------- | ------------------- |
| Frequency Based | Importance Based    |
| Equal Weight    | Weighted Importance |
| Simpler         | More Informative    |
| Less Accurate   | Better Performance  |

---

# Quick Revision

```text
Raw Text
    ↓
Preprocessing
    ↓
Bag of Words
    ↓
TF-IDF
    ↓
N-Grams
    ↓
Word2Vec
    ↓
Deep Learning NLP
```

---

# Next Topics

* Word2Vec
* CBOW
* Skip-Gram
* Embedding Layer
* GloVe
* Word Embeddings
* Semantic Relationships

```
```
