# Natural Language Processing (NLP) - Day 1 & Day 2 Notes

## Overview

Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables machines to understand, interpret, and generate human language.

### Applications of NLP

* Chatbots
* Language Translation
* Spam Detection
* Sentiment Analysis
* Text Summarization
* Search Engines
* Recommendation Systems

---

# NLP Learning Roadmap

```text
Text Preprocessing
       ↓
Bag of Words / TF-IDF
       ↓
Word2Vec
       ↓
Machine Learning NLP Projects
       ↓
RNN
       ↓
LSTM / GRU
       ↓
Attention Mechanism
       ↓
Transformers
       ↓
BERT
```

---

# Prerequisites

Before learning NLP, it is recommended to have knowledge of:

* Python
* Statistics
* Machine Learning
* Artificial Neural Networks (ANN)
* Loss Functions
* Optimizers

---

# Basic NLP Terminology

## Corpus

A collection of all text documents.

### Example

```text
Document 1
Document 2
Document 3
```

All documents together form a **Corpus**.

---

## Document

A single sentence, paragraph, or text sample.

### Example

```text
The food is good.
```

---

## Vocabulary

The set of all unique words present in the corpus.

### Example

```text
good
food
bad
pizza
```

Vocabulary Size = 4

---

## Word

The smallest meaningful token extracted from text.

---

# NLP Workflow

```text
Dataset
   ↓
Text Preprocessing
   ↓
Vectorization
   ↓
Machine Learning Model
   ↓
Prediction
```

---

# Text Preprocessing

Text preprocessing is the process of cleaning and transforming raw text into a machine-understandable format.

---

## 1. Tokenization

Tokenization converts a sentence into individual words (tokens).

### Example

```text
I love NLP
```

After Tokenization:

```text
["I", "love", "NLP"]
```

---

## 2. Stop Words Removal

Stop words are frequently occurring words that usually carry little meaning.

### Examples

```text
the
is
are
to
of
and
```

### Why Remove Them?

* Reduce noise
* Reduce vocabulary size
* Improve model performance

---

## 3. Stemming

Stemming reduces words to their root form.

### Examples

```text
going      → go
playing    → play
historical → histor
```

### Advantages

* Fast
* Computationally efficient

### Disadvantages

* May produce meaningless words
* Grammar may be lost

---

## 4. Lemmatization

Lemmatization converts words into meaningful root words using a dictionary.

### Examples

```text
going      → go
historical → history
finalized  → final
```

### Advantages

* Preserves meaning
* More accurate

### Disadvantages

* Slower than stemming

---

# When to Use?

## Stemming

Suitable for:

* Spam Classification
* Review Classification
* Sentiment Analysis

---

## Lemmatization

Suitable for:

* Chatbots
* Language Translation
* Text Summarization

---

# Converting Text into Numbers

Machine Learning algorithms cannot understand text directly.

Therefore, text must be converted into numerical vectors.

---

# One-Hot Encoding

Each word is represented by a binary vector.

## Example Vocabulary

```text
[a, man, eat, food]
```

## Vector Representation

```text
a      = [1,0,0,0]
man    = [0,1,0,0]
eat    = [0,0,1,0]
food   = [0,0,0,1]
```

---

## Advantages

* Easy to understand
* Easy to implement

---

## Disadvantages

### Sparse Matrix

Produces many zeros.

```text
[0,0,0,0,0,1,0,0,0]
```

---

### Out Of Vocabulary (OOV)

Cannot handle unseen words.

### Example

Training Vocabulary:

```text
cat
dog
```

Testing Word:

```text
elephant
```

Cannot be represented.

---

### No Semantic Meaning

Relationship between words is not captured.

```text
King ≠ Queen
```

Machine treats them as completely different.

---

# Bag of Words (BoW)

Bag of Words converts documents into vectors using word frequencies.

---

## Example Documents

```text
good boy
good girl
boy girl good
```

---

## Vocabulary

```text
good
boy
girl
```

---

## Vector Representation

| Document | Good | Boy | Girl |
| -------- | ---- | --- | ---- |
| D1       | 1    | 1   | 0    |
| D2       | 1    | 0   | 1    |
| D3       | 1    | 1   | 1    |

---

# Advantages of Bag of Words

* Easy to implement
* Easy to interpret
* Works well for simple text classification tasks

---

# Disadvantages of Bag of Words

## 1. Sparse Matrix

Large number of zeros.

---

## 2. Out Of Vocabulary Problem

Unknown words are ignored.

---

## 3. Word Order Lost

```text
Good Boy
Boy Good
```

Both produce similar vectors.

---

## 4. Semantic Meaning Lost

Relationship between words is not captured.

---

# N-Grams

N-Grams are used to preserve word order information.

---

## Unigram (N=1)

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

## Bigram (N=2)

```text
I love NLP
```

Output:

```text
I love
love NLP
```

---

## Trigram (N=3)

```text
I love NLP Course
```

Output:

```text
I love NLP
love NLP Course
```

---

# NLTK Library

NLTK (Natural Language Toolkit) is one of the most popular Python libraries for NLP.

---

## Important Imports

```python
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
```

---

# CountVectorizer (Bag of Words)

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus)
```

---

# Key Interview Questions

## Difference Between Stemming and Lemmatization

| Stemming      | Lemmatization   |
| ------------- | --------------- |
| Fast          | Slow            |
| Root Word     | Meaningful Word |
| Less Accurate | More Accurate   |

---

## Why Is One-Hot Encoding Not Preferred?

* Sparse Matrix
* OOV Problem
* No Semantic Meaning

---

## Advantages of Bag of Words

* Easy
* Fast
* Interpretable

---

## Disadvantages of Bag of Words

* Sparse Matrix
* OOV Problem
* Word Order Lost
* Semantic Meaning Lost

---

# Quick Revision Flow

```text
Raw Text
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming / Lemmatization
   ↓
One-Hot Encoding
   ↓
Bag of Words
   ↓
TF-IDF
   ↓
Word2Vec
   ↓
Deep Learning NLP
```

---



```
```
