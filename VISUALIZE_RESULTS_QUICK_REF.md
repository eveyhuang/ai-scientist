# Visualize Results - Quick Reference

## Quick Start

```bash
# Visualize evaluation results
python visualize_results.py --evaluation-file evaluations/evaluations_pairwise_human-human_20251030_155122.json

# Visualize similarity results
python visualize_results.py --similarity-file similarity_analysis/similarity_human-human_20251030_160213.csv

# Visualize both
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_human-human_20251030_155122.json \
    --similarity-file similarity_analysis/similarity_human-human_20251030_160213.csv
```

## Command-Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--evaluation-file` | Path to evaluation JSON file | No* | None |
| `--similarity-file` | Path to similarity CSV file | No* | None |
| `--output-dir` | Output directory for visualizations | No | `visuals` |
| `--all` | Process both files if provided | No | False |

*At least one file must be provided

## Output Files

### Evaluation Results → Generates:
- `overlap_matrices_combined.png` - All dimensions together
- `matrix_research_question_aims.png`
- `matrix_data_empirical_context.png`
- `matrix_methods_design.png`
- `matrix_intended_contribution_outcomes.png`
- `matrix_resources_timing_artifacts.png`
- `evaluation_statistics.txt` - Statistics report

### Similarity Results → Generates:
- `similarity_matrices_combined.png` - All metrics together
- `matrix_tfidf_similarity.png`
- `matrix_embedding_similarity.png`
- `matrix_keyword_jaccard.png`
- `matrix_topic_similarity.png`
- `metrics_correlation.png`
- `similarity_distributions.png`
- `similarity_statistics.txt` - Statistics report

## Common Use Cases

### Use Case 1: Visualize Human-Human Baseline
```bash
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_human-human_20251030_155122.json \
    --similarity-file similarity_analysis/similarity_human-human_20251030_160213.csv \
    --output-dir visuals/human_baseline
```

### Use Case 2: Compare AI-Generated Proposals (Same Model)
```bash
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini_20251031_120000.json \
    --output-dir visuals/ai_gemini_comparison
```

### Use Case 3: Analyze Human-AI Overlap
```bash
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_human-ai_20251031_140000.json \
    --similarity-file similarity_analysis/similarity_human-ai_20251031_140100.csv \
    --output-dir visuals/human_ai_overlap
```

## Interpreting Visualizations

### Evaluation Heatmaps (Score 0-4)
- **0** = Completely different
- **1** = Related theme, different core question
- **2** = Overlapping with distinctions
- **3** = Near-identical
- **4** = Identical

### Similarity Heatmaps (Score 0-1)
- **TF-IDF**: Word importance (typical: 0.1-0.3)
- **Embedding**: Semantic meaning (typical: 0.6-0.8)
- **Jaccard**: Exact keyword overlap (typical: 0.0-0.2)
- **Topic**: Topic model overlap (typical: 0.0-0.3)

## Batch Processing

```bash
# Process all evaluation files
for file in evaluations/evaluations_pairwise_*.json; do
    name=$(basename "$file" .json)
    python visualize_results.py --evaluation-file "$file" --output-dir "visuals/$name"
done
```

## Tips
- All images saved at 300 DPI (publication quality)
- Statistics printed to console AND saved to text files
- Output directory created automatically
- Handles missing data gracefully (NaN values)

---
📚 **Full Guide:** See `VISUALIZE_RESULTS_GUIDE.md` for detailed documentation

