# Proposal Similarity Analysis Guide

## Overview

`evaluate_proposals_similarity.py` performs **textual similarity** and **topic modeling** analysis on research proposals using:
1. **TF-IDF cosine similarity** - Traditional text similarity
2. **OpenAI embeddings** - Semantic similarity using AI embeddings
3. **Keyword overlap** - Shared important terms (Jaccard similarity)
4. **Topic modeling (LDA)** - Latent themes and topic distributions

## Installation

### Required Dependencies

```bash
pip install scikit-learn numpy pandas openai
```

Or add to your `requirements.txt`:
```
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
openai>=1.0.0
```

## Usage

### Basic Usage

```bash
# Analyze human vs AI proposals
python evaluate_proposals_similarity.py --compare-type human-ai

# Analyze human proposals with each other
python evaluate_proposals_similarity.py --compare-type human-human

# Analyze AI proposals with each other
python evaluate_proposals_similarity.py --compare-type ai-ai
```

### With Filters

```bash
# Compare specific AI model
python evaluate_proposals_similarity.py \
  --compare-type human-ai \
  --ai-model gemini-2.5-pro

# Compare specific template/role
python evaluate_proposals_similarity.py \
  --compare-type ai-ai \
  --template single

# Test with small sample
python evaluate_proposals_similarity.py \
  --compare-type human-human \
  --max-proposals 3
```

### Custom Output

```bash
python evaluate_proposals_similarity.py \
  --compare-type human-ai \
  --output my_similarity_analysis.json
```

## Output Files

The script generates two output files in `similarity_analysis/`:

### 1. JSON File (Complete Results)

**Filename:** `similarity_{comparison_type}_{timestamp}.json`

**Example:** `similarity_human-ai_20251030_150000.json`

**Structure:**
```json
{
  "metadata": {
    "comparison_type": "human-ai",
    "total_analyses": 720,
    "generation_timestamp": "2025-10-30T15:00:00",
    "methods": {
      "tfidf": "TF-IDF based cosine similarity",
      "embeddings": "OpenAI text-embedding-3-small",
      "keywords": "TF-IDF top-20 keywords with Jaccard similarity",
      "topics": "Latent Dirichlet Allocation (LDA) with 5 topics"
    }
  },
  "results": [
    {
      "proposal_1_id": "human_1",
      "proposal_2_id": "ai_single_gemini_01",
      "proposal_1_title": "...",
      "proposal_2_title": "...",
      "proposal_1_who": "human",
      "proposal_2_who": "ai",
      "similarity_metrics": {
        "tfidf_cosine_similarity": 0.342,
        "embedding_cosine_similarity": 0.756,
        "keyword_overlap": {
          "overlapping_keywords": ["synthesis", "data", "network", "protein"],
          "num_overlapping": 4,
          "jaccard_similarity": 0.125,
          "proposal_1_unique": ["cytokinesis", "heat", "stress"],
          "proposal_2_unique": ["chromatin", "genome", "epigenetic"]
        },
        "topic_analysis": {
          "topic_similarity": 0.612,
          "proposal_1_topic_distribution": [0.15, 0.32, 0.08, 0.25, 0.20],
          "proposal_2_topic_distribution": [0.10, 0.28, 0.12, 0.35, 0.15],
          "topics": [
            {
              "topic_id": 0,
              "top_words": ["cell", "protein", "molecular", "data", "analysis"]
            },
            ...
          ]
        }
      }
    }
  ]
}
```

### 2. CSV File (Simplified View)

**Filename:** `similarity_{comparison_type}_{timestamp}.csv`

**Columns:**
- `proposal_1_id`, `proposal_2_id`
- `proposal_1_title`, `proposal_2_title`
- `proposal_1_who`, `proposal_2_who`
- `tfidf_similarity` (0-1)
- `embedding_similarity` (0-1)
- `keyword_overlap_count` (integer)
- `keyword_jaccard` (0-1)
- `topic_similarity` (0-1)

Easy to open in Excel/Google Sheets for analysis!

## Similarity Metrics Explained

### 1. TF-IDF Cosine Similarity

**What it measures:** Word-level similarity based on term frequency

**Range:** 0.0 (completely different) to 1.0 (identical)

**Interpretation:**
- **< 0.2:** Very different content
- **0.2-0.4:** Some overlap
- **0.4-0.6:** Moderate similarity
- **0.6-0.8:** High similarity
- **> 0.8:** Very similar content

**Pros:**
- Fast and efficient
- Good for detecting word-level overlap
- No API calls needed

**Cons:**
- Doesn't capture semantic meaning
- Sensitive to vocabulary differences

### 2. Embedding Cosine Similarity

**What it measures:** Semantic similarity using AI embeddings

**Range:** 0.0 (unrelated) to 1.0 (semantically identical)

**Interpretation:**
- **< 0.5:** Different topics
- **0.5-0.7:** Related but distinct
- **0.7-0.8:** Similar topics/themes
- **0.8-0.9:** Very similar content
- **> 0.9:** Nearly identical semantically

**Pros:**
- Captures semantic meaning
- Better at detecting paraphrasing
- Understands context

**Cons:**
- Requires OpenAI API (costs money)
- Slower than TF-IDF
- May be too lenient

**Cost:** ~$0.02 per 1M tokens with `text-embedding-3-small`

### 3. Keyword Overlap

**What it measures:** Shared important keywords

**Metrics:**
- `num_overlapping`: Number of shared keywords
- `jaccard_similarity`: Overlap ratio (0-1)

**Interpretation:**
- **0-3 keywords:** Minimal overlap
- **4-7 keywords:** Some common themes
- **8-12 keywords:** Significant overlap
- **> 12 keywords:** High topical overlap

**Jaccard similarity:**
- **< 0.1:** Different focus areas
- **0.1-0.3:** Some shared concepts
- **> 0.3:** Strong keyword overlap

### 4. Topic Similarity (LDA)

**What it measures:** Overlap in latent topics/themes

**Range:** 0.0 (different topics) to 1.0 (same topic distribution)

**Interpretation:**
- **< 0.4:** Different thematic focus
- **0.4-0.6:** Some shared themes
- **0.6-0.8:** Similar thematic content
- **> 0.8:** Very similar topic distributions

**Topics:** Extracts 5 latent topics with top 5 words each

## Example Workflows

### Workflow 1: Find Similar Human Proposals

```bash
# Analyze all human-human pairs
python evaluate_proposals_similarity.py --compare-type human-human

# Open the CSV to find high similarity pairs
# Look for:
# - tfidf_similarity > 0.4
# - keyword_overlap_count > 8
# - topic_similarity > 0.6
```

### Workflow 2: Find AI Proposals Similar to Human

```bash
# Analyze human vs AI
python evaluate_proposals_similarity.py \
  --compare-type human-ai \
  --output human_ai_similarity.json

# Filter results to find AI proposals most similar to each human proposal
python -c "
import json
import pandas as pd

with open('similarity_analysis/human_ai_similarity.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['results'])
# Extract similarity scores
df['embedding_sim'] = df['similarity_metrics'].apply(lambda x: x.get('embedding_cosine_similarity'))

# For each human proposal, find top 5 most similar AI proposals
for human_id in df['proposal_1_id'].unique():
    subset = df[df['proposal_1_id'] == human_id].sort_values('embedding_sim', ascending=False)
    print(f'\n{human_id}:')
    print(subset[['proposal_2_id', 'embedding_sim']].head(5))
"
```

### Workflow 3: Detect Duplicate/Overlapping Proposals

```bash
# High threshold for detecting near-duplicates
python evaluate_proposals_similarity.py --compare-type ai-ai

# Look for pairs with ALL of:
# - tfidf_similarity > 0.7
# - embedding_similarity > 0.85
# - keyword_overlap_count > 12
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv` | str | `all_proposals_combined.csv` | Input CSV file |
| `--compare-type` | choice | `human-ai` | Type: `human-human`, `ai-ai`, `human-ai` |
| `--ai-model` | str | `None` | Filter by specific AI model |
| `--template` | str | `single_scientist` | Filter by role (`single`, `group`, `group_int`) |
| `--max-proposals` | int | `None` | Limit number of proposals |
| `--output` | str | auto | Custom output filename |

## Configuration

### OpenAI API Key

The script uses `config.env` to load your OpenAI API key for embeddings:

```bash
# In config.env
OPENAI_API_KEY=your-key-here
```

**Without API key:** Script still works but skips embedding similarity.

## Performance

### Speed

- **TF-IDF:** ~0.1 seconds per pair
- **Embeddings:** ~1-2 seconds per pair (API call)
- **Keywords:** ~0.1 seconds per pair
- **Topics:** ~0.2 seconds per pair

### Estimates

| Comparison | Pairs | Time (with embeddings) | Cost |
|------------|-------|------------------------|------|
| Human-human (12 proposals) | 132 | ~5 minutes | $0.05 |
| Human-AI (12 × 60) | 720 | ~25 minutes | $0.30 |
| AI-AI (60 proposals) | 3,540 | ~2 hours | $1.50 |

**Cost calculation:** Assumes ~2000 tokens per proposal at $0.02/1M tokens

## Comparison with evaluate_proposals.py

| Feature | evaluate_proposals.py | evaluate_proposals_similarity.py |
|---------|----------------------|----------------------------------|
| **Method** | AI model evaluation (GPT/Gemini) | Statistical text analysis |
| **Output** | Qualitative assessment + scores | Quantitative similarity metrics |
| **Speed** | Slow (10-30s per pair) | Fast (1-2s per pair) |
| **Cost** | High ($0.01-0.05 per eval) | Low ($0.0004 per pair) |
| **Insight** | Structured evaluation criteria | Textual/topical similarity |
| **Use Case** | Quality assessment, scoring | Overlap detection, clustering |

**Use both:** They complement each other!
- Use `similarity.py` to find potentially overlapping proposals
- Use `evaluate_proposals.py` to deeply assess quality/merit

## Troubleshooting

### "OpenAI API key not found"

**Solution:**
```bash
# Add to config.env
echo 'OPENAI_API_KEY=sk-your-key-here' >> config.env
```

Or skip embeddings (still get TF-IDF, keywords, topics).

### "Empty text in one or both proposals"

Some proposals may have empty `full_draft` field. These are automatically skipped.

### Memory issues with large datasets

For large numbers of proposals (> 100), process in batches:
```bash
python evaluate_proposals_similarity.py --max-proposals 20 --compare-type ai-ai
```

## Future Enhancements

Potential additions:
- BERTopic for hierarchical topic modeling
- Sentence-level alignment analysis
- Citation network overlap
- Method similarity scoring
- Parallel processing for speed

## Summary

🎯 **Use this script to:**
- Find proposals with high textual overlap
- Detect potential duplicates
- Cluster similar proposals
- Identify unique vs. common themes
- Quantify similarity across different metrics

Fast, cost-effective, and complementary to AI-based evaluation! 📊

