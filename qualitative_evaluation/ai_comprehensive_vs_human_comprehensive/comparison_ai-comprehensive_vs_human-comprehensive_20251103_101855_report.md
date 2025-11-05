# Evaluation Score Comparison Report
Generated: 2025-11-03 10:18:55

## Input Files

**File 1:** `evaluations_single_ai_comprehensive_20251031_155119.json`
- Source: ai
- Evaluations: 20
- Template: comprehensive

**File 2:** `evaluations_single_human_comprehensive_20251031_155859.json`
- Source: human
- Evaluations: 12
- Template: comprehensive

## Detected Fields and Auto-Matching

**File 1 Fields:**
- `alignment_with_call`
- `collaborative_approach`
- `data_synthesis_approach`
- `feasibility`
- `impact`
- `innovation`
- `methodology`
- `open_science_commitment`
- `overall_score`
- `scientific_merit`
- `training_opportunities`

**File 2 Fields:**
- `alignment_with_call`
- `collaborative_approach`
- `data_synthesis_approach`
- `feasibility`
- `impact`
- `innovation`
- `methodology`
- `open_science_commitment`
- `overall_score`
- `scientific_merit`
- `training_opportunities`

**Auto-Matched Fields:**
- `alignment_with_call`: file1=`alignment_with_call` ↔ file2=`alignment_with_call`
- `collaborative_approach`: file1=`collaborative_approach` ↔ file2=`collaborative_approach`
- `data_synthesis_approach`: file1=`data_synthesis_approach` ↔ file2=`data_synthesis_approach`
- `feasibility`: file1=`feasibility` ↔ file2=`feasibility`
- `impact`: file1=`impact` ↔ file2=`impact`
- `innovation`: file1=`innovation` ↔ file2=`innovation`
- `methodology`: file1=`methodology` ↔ file2=`methodology`
- `open_science_commitment`: file1=`open_science_commitment` ↔ file2=`open_science_commitment`
- `overall_score`: file1=`overall_score` ↔ file2=`overall_score`
- `scientific_merit`: file1=`scientific_merit` ↔ file2=`scientific_merit`
- `training_opportunities`: file1=`training_opportunities` ↔ file2=`training_opportunities`

## Overall Score Comparison

### Summary Statistics

**ai:**
- Mean: 3.50
- Median: 3.60
- Std Dev: 1.39
- Range: 2.00 - 4.90
- Count: 20

**human:**
- Mean: 4.28
- Median: 4.45
- Std Dev: 0.48
- Range: 3.30 - 4.80
- Count: 12

**Difference:** -0.78 points

### Statistical Tests

**Independent Samples T-test:**
- t-statistic: -1.875
- p-value: 0.0705
- Significant: No (α=0.05)

**Mann-Whitney U Test (Non-parametric):**
- U-statistic: 117.500
- p-value: 0.9376
- Significant: No (α=0.05)

**Effect Size (Cohen's d):**
- d = -0.685
- Interpretation: medium

## Field-by-Field Comparison

| Field | Group 1 Mean | Group 2 Mean | Difference | p-value | Significant |
|-------|--------------|--------------|------------|---------|-------------|
| Alignment With Call | 3.85 | 4.75 | -0.90 | 0.0374 | Yes |
| Collaborative Approach | 3.15 | 3.92 | -0.77 | 0.1755 | No |
| Data Synthesis Approach | 3.25 | 4.75 | -1.50 | 0.0503 | No |
| Feasibility | 2.90 | 3.75 | -0.85 | 0.0584 | No |
| Impact | 3.85 | 5.00 | -1.15 | 0.0045 | Yes |
| Innovation | 3.40 | 4.83 | -1.43 | 0.0251 | Yes |
| Methodology | 3.10 | 4.17 | -1.07 | 0.3179 | No |
| Open Science Commitment | 3.95 | 4.33 | -0.38 | 0.4316 | No |
| Overall Score | 3.50 | 4.28 | -0.78 | 0.9376 | No |
| Scientific Merit | 3.50 | 4.75 | -1.25 | 0.0237 | Yes |
| Training Opportunities | 3.65 | 2.58 | +1.07 | 0.0510 | No |

## By AI Model Comparison

**Group1:**

- **gemini-2.5-pro**: Mean = 4.85, Count = 10
- **gpt-4**: Mean = 2.15, Count = 10

**Group2:**

- **human**: Mean = 4.28, Count = 12

