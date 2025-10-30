# Evaluation Results Visualization Guide

## Overview

The `result_analysis.ipynb` notebook creates **heatmap matrices** to visualize pairwise proposal overlap scores across all dimensions.

## What the Notebook Does

### 1. **Loads Evaluation Data**
- Reads the JSON file with pairwise evaluations
- Extracts all proposal IDs
- Identifies evaluation dimensions

### 2. **Creates Matrices**
- One matrix per dimension (5 total):
  - Research Question / Aims
  - Data / Empirical Context
  - Methods / Design
  - Intended Contribution / Outcomes
  - Resources / Timing / Artifacts

### 3. **Generates Visualizations**

#### Combined View
- All 5 dimensions in one figure (3×2 grid)
- Saved as: `evaluations/overlap_matrices.png`

#### Individual Views
- Each dimension as a separate detailed heatmap
- Saved as: `evaluations/matrix_{dimension_name}.png`
  - `matrix_research_question_aims.png`
  - `matrix_data_empirical_context.png`
  - `matrix_methods_design.png`
  - `matrix_intended_contribution_outcomes.png`
  - `matrix_resources_timing_artifacts.png`

### 4. **Summary Statistics**
- Mean, median, std dev for each dimension
- Score distributions (0-4)
- Identifies high-overlap pairs (score ≥ 3)

## How to Use

### Run the Notebook

```bash
jupyter notebook result_analysis.ipynb
```

Then run all cells:
- Cell 1: Import libraries
- Cell 2: Load evaluation data
- Cell 3: Extract proposal IDs
- Cell 4: Extract dimensions
- Cell 5: Fill matrices with scores
- Cell 6: Create combined heatmap
- Cell 7: Create individual heatmaps
- Cell 8: Print summary statistics
- Cell 9: Find high-overlap pairs

### Interpreting the Heatmaps

**Color Scheme:**
- 🟢 Green (4): Identical/Very high overlap
- 🟡 Yellow (2-3): Moderate overlap
- 🟠 Orange (1): Minimal overlap
- 🔴 Red (0): No overlap

**Matrix Layout:**
- **Rows (Y-axis):** Proposal 1 (Reference)
- **Columns (X-axis):** Proposal 2 (Compared Against)
- **Cell Value:** Overlap score (0-4)

**Reading Example:**
```
Row: human_1, Column: human_2, Value: 2
→ When comparing human_1 to human_2, the overlap score is 2
```

### Score Interpretation

| Score | Meaning | Description |
|-------|---------|-------------|
| **0** | No overlap | Completely different |
| **1** | Minimal | Related theme but different approach |
| **2** | Moderate | Some shared elements |
| **3** | High | Near-identical with minor differences |
| **4** | Identical | Effectively the same |

## Output Files

All outputs saved to `evaluations/` directory:

```
evaluations/
├── overlap_matrices.png              # Combined view (all dimensions)
├── matrix_research_question_aims.png # Individual dimension
├── matrix_data_empirical_context.png
├── matrix_methods_design.png
├── matrix_intended_contribution_outcomes.png
└── matrix_resources_timing_artifacts.png
```

## Example Analysis

### Find Most Overlapping Proposals

```python
# In a new notebook cell:
for dim_name in dimension_names:
    matrix = matrices[dim_name]
    # Find max score (excluding diagonal/NaN)
    max_score = np.nanmax(matrix)
    if max_score >= 3:
        idx = np.where(matrix == max_score)
        print(f"{dim_name}: Max overlap = {max_score}")
        for i, j in zip(idx[0], idx[1]):
            print(f"  {proposal_ids[i]} vs {proposal_ids[j]}")
```

### Calculate Average Overlap Per Proposal

```python
# Average overlap score for each proposal
for i, prop_id in enumerate(proposal_ids):
    scores = []
    for dim_name in dimension_names:
        matrix = matrices[dim_name]
        # Get row i (comparing this proposal to others)
        row_scores = matrix[i, :]
        scores.extend(row_scores[~np.isnan(row_scores)])
    
    avg = np.mean(scores)
    print(f"{prop_id}: Average overlap = {avg:.2f}")
```

### Export to CSV

```python
# Create a DataFrame for easy export
import pandas as pd

data = []
for eval_data in evaluations:
    row = {
        'proposal_1': eval_data['proposal_1_id'],
        'proposal_2': eval_data['proposal_2_id'],
    }
    
    dimensions = eval_data['evaluation_response']['comparison']['dimensions']
    for dim in dimensions:
        row[dim['dimension']] = dim['score']
    
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('evaluations/overlap_scores.csv', index=False)
print("Saved to overlap_scores.csv")
```

## Customization

### Change Color Scheme

```python
# Use different colormap
sns.heatmap(..., cmap='coolwarm')  # Blue to Red
sns.heatmap(..., cmap='viridis')   # Purple to Yellow
sns.heatmap(..., cmap='Blues')     # White to Blue
```

### Adjust Figure Size

```python
# Make larger/smaller
fig, ax = plt.subplots(figsize=(16, 14))  # Width, Height in inches
```

### Change Score Range

```python
# If using different scoring scale
sns.heatmap(..., vmin=0, vmax=10)  # For 0-10 scale
```

### Filter by Proposal Type

```python
# Only show certain proposals
filtered_ids = [pid for pid in proposal_ids if pid.startswith('human_')]
filtered_indices = [id_to_idx[pid] for pid in filtered_ids]

# Subset matrix
filtered_matrix = matrix[np.ix_(filtered_indices, filtered_indices)]

# Plot
sns.heatmap(filtered_matrix, xticklabels=filtered_ids, yticklabels=filtered_ids, ...)
```

## Tips

1. **Save high-resolution images**: Use `dpi=300` or higher for publication quality

2. **Interactive exploration**: Use `%matplotlib widget` for interactive plots

3. **Compare multiple evaluations**: Load different JSON files and compare matrices

4. **Statistical tests**: Use scipy to test for significant differences:
   ```python
   from scipy import stats
   # Compare two dimensions
   dim1_scores = matrices[dimension_names[0]].flatten()
   dim2_scores = matrices[dimension_names[1]].flatten()
   t_stat, p_val = stats.ttest_ind(dim1_scores[~np.isnan(dim1_scores)], 
                                     dim2_scores[~np.isnan(dim2_scores)])
   ```

5. **Cluster analysis**: Use hierarchical clustering to group similar proposals:
   ```python
   from scipy.cluster.hierarchy import dendrogram, linkage
   from scipy.spatial.distance import squareform
   
   # Convert similarity to distance
   distance_matrix = 4 - matrix  # Max score (4) minus actual score
   
   # Cluster
   linkage_matrix = linkage(squareform(distance_matrix), method='ward')
   dendrogram(linkage_matrix, labels=proposal_ids)
   ```

## Troubleshooting

### "File not found"
Update the file path in Cell 2:
```python
evaluation_file = "evaluations/YOUR_FILE_NAME.json"
```

### "No dimensions found"
Check that your JSON file has the correct structure with `evaluation_response.comparison.dimensions`

### Empty/blank matrices
Verify that evaluations contain scores for all proposal pairs

### Missing matplotlib/seaborn
```bash
pip install matplotlib seaborn numpy pandas
```

## Similarity Analysis Visualization

The notebook also includes visualization of **textual similarity metrics** from the `evaluate_proposals_similarity.py` output.

### Similarity Metrics Visualized

1. **TF-IDF Cosine Similarity** - Statistical term frequency comparison
2. **Embedding Similarity (OpenAI)** - Semantic similarity using embeddings
3. **Keyword Jaccard Similarity** - Keyword overlap coefficient
4. **Topic Similarity (LDA)** - Topic modeling overlap

### Similarity Visualizations

**Cells 10-18** create:

1. **Combined Similarity Heatmap** (`similarity_matrices_combined.png`)
   - All 4 metrics in one figure (2×2 grid)
   - Scores range 0-1 (higher = more similar)

2. **Individual Metric Heatmaps**
   - `matrix_tfidf_similarity.png`
   - `matrix_embedding_similarity.png`
   - `matrix_keyword_jaccard.png`
   - `matrix_topic_similarity.png`

3. **Metric Correlation Analysis** (`metrics_correlation.png`)
   - Shows how different similarity metrics correlate with each other
   - Identifies which metrics provide similar/different information

4. **Distribution Plots** (`similarity_distributions.png`)
   - Histograms for each metric
   - Mean and median lines
   - Helps understand the range and spread of scores

### Similarity Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| **0.9 - 1.0** | Nearly identical |
| **0.7 - 0.9** | Very high similarity |
| **0.5 - 0.7** | Moderate similarity |
| **0.3 - 0.5** | Low similarity |
| **0.0 - 0.3** | Very different |

### Additional Analysis

**High Similarity Detection:** Identifies proposal pairs in the top 10% for each metric

**Summary Statistics:** Mean, median, std dev, percentiles for each metric

**Correlation Matrix:** Shows relationships between different similarity measures

## Summary

### Overlap Analysis (Cells 1-9)
✅ **Visual matrix representation** of all pairwise overlaps  
✅ **One heatmap per dimension** for detailed analysis  
✅ **Summary statistics** for each dimension  
✅ **High-overlap detection** (score ≥ 3)  
✅ **Publication-quality figures** (300 DPI)  

### Similarity Analysis (Cells 10-18)
✅ **4 similarity metrics** visualized as heatmaps  
✅ **Correlation analysis** between metrics  
✅ **Distribution plots** for each metric  
✅ **Top 10% similarity pairs** identified  
✅ **Comprehensive statistical summaries**

Perfect for identifying which proposals have high overlap and similarity across multiple dimensions! 📊

