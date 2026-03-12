# Data Types for Health Promotion
## Course Overview: Computational Text Analysis in Python

Welcome to the course! This overview gives you a roadmap of all the notebooks we will work through together. Each notebook introduces one method or concept, building on the previous one. You do not need any prior programming experience — we will go step by step.

---

## 📚 Table of Contents

1. [What is Computational Text Analysis?](#what-is-computational-text-analysis)
2. [How the Course is Structured](#how-the-course-is-structured)
3. [Notebook 1 — Text Preprocessing](#notebook-1--text-preprocessing)
4. [Notebook 2 — Data Visualization](#notebook-2--data-visualization)
5. [Notebook 3 — Dictionary-Based Analysis](#notebook-3--dictionary-based-analysis)
6. [Notebook 4 — Supervised Machine Learning](#notebook-4--supervised-machine-learning)
7. [Notebook 5 — Static Word Embeddings](#notebook-5--static-word-embeddings)
8. [Notebook 6 — Topic Modeling](#notebook-6--topic-modeling)
9. [Notebook 7 — Automated Annotation with LLMs](#notebook-7--automated-annotation-with-llms)
10. [The Dataset We Use](#the-dataset-we-use)
11. [Getting Started](#getting-started)

---

## What is Computational Text Analysis?

Computational text analysis means using programming and statistical methods to study large amounts of text automatically. Instead of reading hundreds or thousands of documents by hand, we write code that can process, categorize, and summarize text for us.

In the context of health communication, this is particularly powerful. Think about the sheer volume of social media posts, news articles, and online comments that touch on health topics every day — no human could read all of that. But with the right tools, we can identify patterns, track topics over time, and understand how health is discussed across different communities.

In this course, we focus on **health-related social media data**, working through a range of methods from simple word counting all the way to using Large Language Models (LLMs) for automated annotation.

---

## How the Course is Structured

The notebooks are designed to be worked through in order. Each one introduces a new method, explains the core idea in plain language, and then walks you through the code step by step.

```
Text Preprocessing  →  Visualization  →  Dictionary Analysis
        ↓
Supervised ML  →  Word Embeddings  →  Topic Modeling  →  LLM Annotation
```

You do not need to master one notebook before moving to the next, but the concepts do build on each other — especially preprocessing, which is used in almost every subsequent notebook.

---

## Notebook 1 — Text Preprocessing

📄 **File:** `hso_textpreprocessing.ipynb`

Before we can analyse text with a computer, we need to **clean and prepare it**. Raw text is messy — it contains punctuation, capital letters, filler words, and all kinds of noise. This notebook teaches you the fundamental techniques for getting text into a form that computers can work with.

### What you will learn

| Step | What it does | Example |
|------|-------------|---------|
| **Cleaning** | Removes punctuation, converts to lowercase | `"Hello, World!"` → `"hello world"` |
| **Tokenization** | Splits text into individual words | `"hello world"` → `["hello", "world"]` |
| **Stopword removal** | Removes common, meaningless words | removes `"the"`, `"is"`, `"and"` |
| **Stemming** | Cuts words down to their root | `"running"` → `"run"` |
| **Lemmatization** | Similar to stemming, but keeps valid words | `"better"` stays `"better"` |
| **Bag-of-Words** | Converts text to a numerical matrix | Counts how often each word appears |

> 💡 **Why does this matter?** Almost every method in the following notebooks starts with some form of text preprocessing. Getting this right lays the foundation for everything else.

---

## Notebook 2 — Data Visualization

📄 **File:** `hso_data_visualization.ipynb`

Once the text is cleaned, it helps to **look at it visually** before diving into more complex analysis. This notebook shows you different ways to plot and explore your text data.

### What you will learn

- **Word frequency plots** — which words appear most often?
- **Word clouds** — a visual representation of word importance
- **N-gram analysis** — which two-word phrases (bigrams) are most common?
- **Text length distributions** — how long are the documents in your dataset?
- **Sentiment distribution** — if you have labels, how are they distributed?

> 💡 **Why does this matter?** Visualizing your data before analysis helps you spot patterns, errors, and interesting features you might otherwise miss.

---

## Notebook 3 — Dictionary-Based Analysis

📄 **File:** `hso_dictionary_analysis.ipynb`

Dictionary-based analysis is one of the most intuitive methods in text analysis. You define a list of words that belong to a category (e.g., words related to anxiety, or words expressing trust), and then count how often those words appear in your texts.

### What you will learn

- How to build your own custom dictionary
- How to apply a dictionary to a text dataset and count matches
- How to use a **preloaded dictionary** (the LIWC Opinion Lexicon from NLTK)
- How to compare and summarize results across documents

### Example

```python
health_dictionary = {
    "positive_health": ["healthy", "wellness", "recovery", "thriving"],
    "negative_health": ["sick", "disease", "disorder", "pain"]
}
```

> 💡 **Why does this matter?** Dictionary methods are transparent, easy to interpret, and widely used in health communication research. They are a great starting point before moving to more complex approaches.

---

## Notebook 4 — Supervised Machine Learning

📄 **File:** `hso_supervised_machine_learning.ipynb`

In supervised machine learning, we **train a model on labelled examples** so it can predict labels for new, unseen texts. In this notebook, we use a dataset of social media posts labelled as either indicating a mental health disorder or not.

### What you will learn

We train and compare three classification models:

| Model | Key idea | Strengths |
|-------|----------|-----------|
| **Naive Bayes** | Probabilistic; uses word frequencies | Fast, simple, works well with text |
| **LASSO Logistic Regression** | Regression with feature selection | Prevents overfitting, interpretable |
| **Support Vector Machine (SVM)** | Finds the best boundary between classes | Powerful for high-dimensional data |

You will also learn about **hyperparameter tuning** with `GridSearchCV` and how to evaluate models using accuracy, precision, recall, and F1-score.

> 💡 **Why does this matter?** Supervised ML allows us to automatically classify large volumes of text once we have a set of annotated training examples — a core task in computational health communication research.

---

## Notebook 5 — Static Word Embeddings

📄 **File:** `hso_static_embeddings.ipynb`

Word embeddings are a way of **representing words as numbers** (vectors) that capture their meaning and relationships to other words. Words with similar meanings end up close together in this numerical space.

### What you will learn

- What word embeddings are and why they are useful
- How to load pre-trained **Word2Vec** embeddings
- How to load pre-trained **GloVe** embeddings
- How to train your own custom embeddings on a small dataset
- How to **visualize** word relationships using PCA and t-SNE

### Example

With word embeddings, you can do things like:
- Find words most similar to `"vaccine"` → `["immunization", "inoculation", "shot"]`
- Explore how health-related terms cluster together

> 💡 **Why does this matter?** Word embeddings form the foundation of modern NLP. Understanding them helps you grasp how more advanced models like BERT and LLMs work.

---

## Notebook 6 — Topic Modeling

📄 **File:** `hso_topic_modelling.ipynb`

Topic modeling is an **unsupervised method** for discovering hidden themes in a collection of documents. Unlike supervised ML, you do not need labelled data — the model finds the topics on its own.

### What you will learn

- How **LDA (Latent Dirichlet Allocation)** works: each document is a mix of topics, each topic is a mix of words
- How to preprocess text for topic modeling
- How to train an LDA model and interpret its output
- How to visualize topics interactively with `pyLDAvis`
- An introduction to **BERTopic** for structural topic modeling (STM), which can incorporate metadata like time or author group

> 💡 **Why does this matter?** Topic modeling is widely used in health communication research to map the landscape of how issues are discussed — for example, tracking which health topics dominate social media over time.

---

## Notebook 7 — Automated Annotation with LLMs

📄 **File:** `hso_data_annotation_with_llms.ipynb`

Large Language Models (LLMs) like GPT or Phi-4 can be used to **automatically annotate text** based on instructions you provide in plain language. This is sometimes called "zero-shot" or "few-shot" classification.

### What you will learn

- How to set up a local LLM using **Ollama** (runs on your own computer, no API key needed)
- How to write a **prompt** that instructs the model to classify text
- How to apply the model to a dataset and store the results
- How to evaluate annotation quality

### Example prompt

```
Classify the following text into one of the given categories: ['Disorder', 'No Disorder']

Comments should be coded as DISORDER when they include any indications 
that could refer to a mental health disorder or its symptoms.

Only include the selected category in your response and no further text.
```

> 💡 **Why does this matter?** LLM-based annotation is a powerful and increasingly popular approach, especially when you have a small budget for human annotators or need to classify text in nuanced ways that are hard to capture with simple rules.

---

## The Dataset We Use

Throughout the course we work with a dataset of social media posts (`mental_health_sentiment.csv`) labelled for mental health conditions. It contains:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each post |
| `text` | The raw text of the social media post |
| `sentiment` | Label: e.g., `Normal`, `Depression`, `Stress` |
| `condition_binary` | Simplified label: `0` = No disorder, `1` = Disorder |

This dataset is used across multiple notebooks so you can see how different methods handle the same data.

---

## Getting Started

### Prerequisites

Make sure you have Python installed, along with Jupyter Notebook or JupyterLab. You can install most required packages by running:

```bash
pip install nltk pandas scikit-learn matplotlib wordcloud gensim bertopic ollama
```

Each notebook also includes an install cell at the top that you can uncomment if needed.

### Recommended order

Work through the notebooks in this order:

1. `hso_textpreprocessing.ipynb`
2. `hso_data_visualization.ipynb`
3. `hso_dictionary_analysis.ipynb`
4. `hso_supervised_machine_learning.ipynb`
5. `hso_static_embeddings.ipynb`
6. `hso_topic_modelling.ipynb`
7. `hso_data_annotation_with_llms.ipynb`

### A note for beginners

Don't worry if not everything makes sense immediately. Computational text analysis combines programming, linguistics, and statistics — it takes time to get comfortable. Focus on understanding *what* each method does and *why* before worrying too much about *how* the code works in detail. The notebooks are designed so you can run them even if you don't yet understand every line.

---

*Course materials developed by Denise J. Roth, Wageningen University & Research.*
