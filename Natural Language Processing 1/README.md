# Natural Language Processing 1

Lab 1 for the **Natural Language Processing course (5DV218 Språkteknologi)** at Umeå University.

This project explores core computational linguistics tasks and modern NLP pipelines using **spaCy**, evaluating rule-based and statistical methods across both English and Swedish text.

## What This Project Covers

- **Tokenization:** Word and sentence segmentation handling punctuation, hyphenated words, abbreviations, and numerical expressions.
- **Lemmatization:** Inflectional normalization and dictionary base-form extraction.
- **Part-of-Speech (POS) Tagging & Morphology:** Coarse-grained universal POS tags (`UPOS`), fine-grained Penn Treebank tags, and morphological features (tense, number, person, case).
- **Syntactic Dependency Parsing:** Grammatical relation extraction (subject, object, modifiers) and visualizing syntactic ambiguity (e.g., parsing *"time flies like an arrow"* under alternative linguistic interpretations).
- **Named Entity Recognition (NER):** Extracting real-world named entities (`PERSON`, `ORG`, `GPE`, `MONEY`, `DATE`).
- **Multilingual NLP (Swedish):** Extending the pipeline to Swedish using `sv_core_news_sm`, analyzing complex Swedish compounds and grammatical structures.
- **Dependency Visualization:** Rendering dependency parse trees with displaCy and exporting to vector SVG images.

## Files

- `lab1.py` - Main Python script executing all 9 laboratory tasks
- `Lab1_Results.pdf` - Written lab report containing linguistic analysis and answers
- `Lab1_Code.pdf` - Formatted code printout
- `output_run.txt` - Full terminal execution logs and token analysis tables
- `output/` - Generated displaCy vector SVG parse trees:
  - `task4_time_flies.svg`
  - `task6_dependency_tree.svg`
  - `task9_swedish_sentence1.svg`
  - `task9_swedish_sentence2.svg`

## Tech Stack

- Python 3
- spaCy (`en_core_web_sm`, `en_core_web_md`, `sv_core_news_sm`)
- displaCy (SVG visualization)

## How to Run

1. Install spaCy and download required language models:
   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   python -m spacy download sv_core_news_sm
   ```
2. Run the script:
   ```bash
   python lab1.py
   ```

## Notes

This is course assignment work intended for academic use.
