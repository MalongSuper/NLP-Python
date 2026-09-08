# NLP Text Summarization with BART and FLAN-T5

This project implements a **multi-style text summarization system** using pretrained Transformer models from Hugging Face.

Instead of producing only one type of summary, the system can transform the same input text into several formats and writing styles, including **casual, formal, bullet-point, and paragraph summaries**.

The main idea is to separate **understanding the content** from **rewriting the content**:

- **BART** extracts and condenses the important information.
- **FLAN-T5** transforms that information into the requested style and format.
- A custom **n-gram copy-detection system** checks whether generated text is too similar to the original.
- Additional refinement passes are used when the generated summary does not satisfy the required constraints.

## Overview

The pipeline can be summarized as:

```text
Input Text
    │
    ▼
Text Preprocessing
    │
    ▼
BART
Extract Important Information
    │
    ▼
Core Idea / Factual Content
    │
    ▼
FLAN-T5
Generate Multiple Candidates
    │
    ├── Copy Detection
    ├── Length Checking
    ├── Meta-Summary Filtering
    └── Candidate Selection
    │
    ▼
Final Summary
```

For long documents, BART uses a **hierarchical summarization strategy**, splitting the document into smaller chunks before combining their summaries.

## Models

### BART

The project uses:

```text
facebook/bart-large-cnn
```

BART is responsible primarily for **content extraction and condensation**.

For normal-sized text, the input is summarized directly. For longer text, the document is divided into approximately 400-word chunks, each chunk is summarized independently, and the partial summaries are combined and summarized again.

This helps the system work around the model's input-length limitations while retaining information from different sections of a document.

### FLAN-T5

The project uses:

```text
google/flan-t5-base
```

FLAN-T5 acts as the **generative and rewriting component**.

It receives the extracted core information together with instructions describing:

- Desired writing style
- Required word count
- Sentence structure
- Paraphrasing requirements
- Formatting requirements

Multiple candidate outputs are generated and the system selects the candidate that best satisfies these constraints.

## Text Processing Utilities

Several helper functions prepare and normalize text before and after model generation.

The notebook includes utilities for:

- Counting words
- Extracting words using regular expressions
- Splitting documents into paragraphs
- Splitting paragraphs into sentences
- Splitting long text into chunks
- Ensuring proper sentence punctuation
- Removing unnecessary bullet markers
- Converting multiple sentences into one sentence
- Formatting long outputs with line breaks

These functions provide the structural layer around the Transformer models.

## Copy Detection

One of the main features of the project is a simple **n-gram-based copy detection mechanism**.

The system:

1. Normalizes the source and generated text.
2. Extracts four-word n-grams from both texts.
3. Finds n-grams shared between the source and candidate.
4. Calculates the proportion of copied n-grams.
5. Uses this score when selecting generated candidates.

For example:

```text
Source → "machine learning learns patterns from data"

Candidate → "machine learning learns patterns from examples"
```

The system checks how much of the candidate's wording is directly shared with the source.

If the overlap is too high, FLAN-T5 receives a stronger paraphrasing instruction and generates another set of candidates.

This is intended to encourage **rewriting rather than simple extraction or copying**.

## Core Idea Extraction

Before generating the final summary, the notebook creates an intermediate **core idea**.

For shorter text:

```text
Input
  ↓
BART summary
  ↓
FLAN-T5 factual content plan
  ↓
Core Idea
```

For longer text:

```text
Long Document
  ↓
400-word chunks
  ↓
BART summarizes each chunk
  ↓
Partial summaries combined
  ↓
BART produces a final condensed summary
  ↓
FLAN-T5 creates the Core Idea
```

The core idea acts as a common factual representation that can be passed into the different summary modes.

This means the casual and formal versions are intended to describe the **same underlying content**, while changing their presentation.

## Summary Modes

The system provides four main modes.

### 1. Casual

The casual mode produces a short and accessible explanation.

Characteristics:

- Approximately **20–25 words**
- Simple vocabulary
- Friendly wording
- Intended to be understandable to a younger student
- Focuses on the main idea rather than minor details

Example usage:

```python
summarize_text(text, mode="casual")
```

### 2. Formal

The formal mode produces a more academic-style summary.

Characteristics:

- Approximately **45–55 words**
- Formal and scientific vocabulary
- Greater emphasis on significance and implications
- Maintains the same core content as the casual version

Example:

```python
summarize_text(text, mode="formal")
```

### 3. Bullet

Bullet mode decomposes the source into individual sentences and paraphrases each sentence independently.

```text
Paragraph
   ↓
Sentences
   ↓
Individual paraphrasing
   ↓
Bullet points
```

Each generated bullet is checked for excessive copying and rewritten when necessary.

Example:

```python
summarize_text(text, mode="bullet")
```

This mode is designed for **granular notes and easy scanning** rather than aggressive compression.

### 4. Paragraph

Paragraph mode focuses on restructuring information into a more cohesive form.

The system takes the sentences within each paragraph and asks FLAN-T5 to **fuse their ideas into a single heavily paraphrased sentence**.

For very long paragraphs, hierarchical BART summarization is applied first.

Example:

```python
summarize_text(text, mode="paragraph")
```

## Candidate Selection

The system does not simply take the first generated response.

Multiple candidates are generated and evaluated using several criteria:

1. Avoid meta-summary responses
2. Satisfy the requested word-count range
3. Stay close to the desired target length
4. Minimize n-gram overlap with the source

The candidate with the best overall score is selected.

This provides a lightweight quality-control layer on top of the pretrained models.

## Iterative Refinement

If a generated summary does not satisfy the requirements, the system can perform additional generation passes.

The general process is:

```text
Generate Candidates
       ↓
Check Output
       ↓
Meets Requirements?
   ┌───┴───┐
  Yes      No
   │        │
   ▼        ▼
 Return   Stronger
 Result   Prompt
             ↓
        Generate Again
             ↓
        Check Again
             ↓
       Final Length Repair
```

The refinement process can address problems such as:

- Excessive copying
- Incorrect length
- Multiple sentences when one is required
- Meta-language such as "This passage is about..."
- Insufficient or excessive information

## Summary Router

The `summarize_text()` function provides a single interface for the entire system:

```python
summarize_text(text, mode="formal")
```

Available modes are:

```text
casual
formal
bullet
paragraph
```

Internally, the router sends the input to the corresponding summarization pipeline.

For example:

```python
summarize_text(text, "casual")
summarize_text(text, "formal")
summarize_text(text, "bullet")
summarize_text(text, "paragraph")
```

This makes it possible to use the same input text while experimenting with different presentation styles.

## Baseline: Lead-3

The notebook also implements a simple **Lead-3 baseline** for comparison.

The baseline simply:

1. Splits the document into sentences.
2. Takes the first three sentences.
3. Returns them as the summary.

```python
baseline_lead_n(text, n=3)
```

Unlike the Transformer-based system, Lead-3 does not understand or rewrite the content. It provides a simple reference point for evaluating whether the more complex summarization pipeline provides useful improvements.

## Main Technologies

The project primarily uses:

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **BART**
- **FLAN-T5**
- **Regular Expressions**

The models are loaded using Hugging Face's pretrained model and tokenizer interfaces.

## Usage

After loading the required models, provide a text passage and select the desired mode:

```python
text = """
Your input document goes here.
"""

summary = summarize_text(text, mode="formal")

print(summary)
```

To generate different versions:

```python
for mode in ["casual", "formal", "bullet", "paragraph"]:
    print(summarize_text(text, mode))
```

## Limitations

This project is primarily an **educational NLP experiment** rather than a production-ready summarization system.

Generated summaries may still:

- Contain factual errors or omissions
- Produce awkward wording
- Fail to perfectly satisfy word-count constraints
- Miss important information
- Generate different results between runs when sampling is enabled
- Produce summaries that are semantically similar even when surface-level copying is low

The n-gram detector only measures **surface-level word overlap**. It cannot determine whether two sentences express the same idea using completely different wording.

## Purpose

The notebook demonstrates how pretrained Transformer models can be combined with traditional text-processing techniques and custom validation logic to build a more controlled NLP application.

Rather than treating summarization as simply:

```text
Text → Summary
```

the project explores a more structured approach:

```text
Text
 ↓
Information Extraction
 ↓
Core Idea
 ↓
Controlled Generation
 ↓
Quality Checks
 ↓
Refinement
 ↓
Style-Specific Summary
```

The result is a flexible summarization system capable of producing multiple representations of the same source material while attempting to preserve its core information and reduce direct copying.