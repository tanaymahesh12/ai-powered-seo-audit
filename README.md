# Intent Checker

**Intent Checker** is an AI-powered Python tool that helps determine how well web content matches specific search queries. It uses transformer-based embeddings to semantically evaluate relevance, rank multiple URLs, and export results into a structured Excel report.

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Transformer Model](https://img.shields.io/badge/model-MiniLM--L6--v1-orange)

## 🔍 What It Does

- Scrapes paragraph text from one or more URLs.
- Accepts a list of user-defined queries (e.g., "best AI tools for SEO").
- Uses SentenceTransformers (MiniLM) to:
  - Find the best matching sentence from each webpage.
  - Score how well the query matches the content.
- Flags whether each query's intent is satisfied (based on score threshold).
- Ranks all URLs by average match score and intent match percentage.
- Exports results to `content_analysis_results.xlsx`.

## 🧠 Perfect For

- SEO professionals analyzing intent coverage across blogs/pages
- Competitive research on landing pages
- Content quality evaluation at scale

## 💾 Output Structure

Generates an Excel file with 3 sheets:
- **Detailed Results** — sentence-level relevance matches
- **Ranked URLs** — overall scoring by URL
- **Best Intent-Matched URL** — top performer summary

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:
- `requests`
- `beautifulsoup4`
- `nltk`
- `sentence-transformers`
- `pandas`
- `openpyxl`

If NLTK errors appear, run:
```python
import nltk
nltk.download('punkt')
```

## 🚀 Usage

Run the script in terminal:

```bash
python v2.py
```

You'll be prompted to input:
- One or more URLs (comma-separated)
- One or more queries (comma-separated)

Example input:

```
Enter URLs: https://example.com, https://another.com
Enter queries: best AI tools, seo automation
```

The Excel output will be saved in the same folder.

## 👨‍💻 Author

Created by [Tanay Mahesh](https://github.com/tanaymahesh12)

---

> "Don’t just guess content relevance — measure it with intent-aware AI."
