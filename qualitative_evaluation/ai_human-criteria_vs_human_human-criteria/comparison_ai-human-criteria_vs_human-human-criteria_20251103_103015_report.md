# Evaluation Score Comparison Report
Generated: 2025-11-03 10:30:15

## Input Files

**File 1:** `evaluations_single_ai_human-criteria_20251031_153158.json`
- Source: ai
- Evaluations: 20
- Template: human_criteria

**File 2:** `evaluations_single_human_human-criteria_20251031_151905.json`
- Source: human
- Evaluations: 12
- Template: human_criteria

## Detected Fields and Auto-Matching

**File 1 Fields:**
- `data_identification`
- `data_sources_and_limitations`
- `feasibility`
- `novelty_significance`
- `open_science_commitment`
- `open_science_compliance`
- `overall_score`
- `relevance_to_emergent_phenomena`
- `rigor_of_approach`
- `scientific_merit_and_innovation`
- `scope_timeline`
- `synthesis_focus`

**File 2 Fields:**
- `data_identification`
- `data_sources_and_limitations`
- `feasibility`
- `novelty_significance`
- `open_science_commitment`
- `open_science_compliance`
- `overall_score`
- `relevance_to_emergent_phenomena`
- `rigor_of_approach`
- `scientific_merit_and_innovation`
- `scope_timeline`
- `synthesis_focus`

**Auto-Matched Fields:**
- `data_identification`: file1=`data_identification` ↔ file2=`data_identification`
- `data_sources_and_limitations`: file1=`data_sources_and_limitations` ↔ file2=`data_sources_and_limitations`
- `feasibility`: file1=`feasibility` ↔ file2=`feasibility`
- `novelty_significance`: file1=`novelty_significance` ↔ file2=`novelty_significance`
- `open_science_commitment`: file1=`open_science_commitment` ↔ file2=`open_science_commitment`
- `open_science_compliance`: file1=`open_science_compliance` ↔ file2=`open_science_compliance`
- `overall_score`: file1=`overall_score` ↔ file2=`overall_score`
- `relevance_to_emergent_phenomena`: file1=`relevance_to_emergent_phenomena` ↔ file2=`relevance_to_emergent_phenomena`
- `rigor_of_approach`: file1=`rigor_of_approach` ↔ file2=`rigor_of_approach`
- `scientific_merit_and_innovation`: file1=`scientific_merit_and_innovation` ↔ file2=`scientific_merit_and_innovation`
- `scope_timeline`: file1=`scope_timeline` ↔ file2=`scope_timeline`
- `synthesis_focus`: file1=`synthesis_focus` ↔ file2=`synthesis_focus`

## Overall Score Comparison

### Summary Statistics

**ai:**
- Mean: 3.51
- Median: 3.65
- Std Dev: 1.24
- Range: 1.90 - 4.80
- Count: 20

**human:**
- Mean: 3.97
- Median: 4.25
- Std Dev: 0.56
- Range: 3.00 - 4.60
- Count: 12

**Difference:** -0.46 points

### Statistical Tests

**Independent Samples T-test:**
- t-statistic: -1.195
- p-value: 0.2416
- Significant: No (α=0.05)

**Mann-Whitney U Test (Non-parametric):**
- U-statistic: 118.000
- p-value: 0.9532
- Significant: No (α=0.05)

**Effect Size (Cohen's d):**
- d = -0.436
- Interpretation: small

## Field-by-Field Comparison

| Field | Group 1 Mean | Group 2 Mean | Difference | p-value | Significant |
|-------|--------------|--------------|------------|---------|-------------|
| Data Identification | 3.00 | 3.83 | -0.83 | 0.3455 | No |
| Data Sources And Limitations | 4.00 | 4.42 | -0.42 | 0.3455 | No |
| Feasibility | 2.80 | 3.33 | -0.53 | 0.3229 | No |
| Novelty Significance | 3.55 | 4.58 | -1.03 | 0.1434 | No |
| Open Science Commitment | 3.60 | 3.75 | -0.15 | 0.9834 | No |
| Open Science Compliance | 3.60 | 3.75 | -0.15 | 0.9834 | No |
| Overall Score | 3.51 | 3.97 | -0.46 | 0.9532 | No |
| Relevance To Emergent Phenomena | 4.00 | 4.33 | -0.33 | 0.5294 | No |
| Rigor Of Approach | 3.20 | 4.08 | -0.88 | 0.4299 | No |
| Scientific Merit And Innovation | 3.59 | 4.35 | -0.76 | 0.6037 | No |
| Scope Timeline | 2.80 | 3.33 | -0.53 | 0.3229 | No |
| Synthesis Focus | 5.00 | 5.00 | +0.00 | 1.0000 | No |

## By AI Model Comparison

**Group1:**

- **gemini-2.5-pro**: Mean = 4.71, Count = 10
- **gpt-4**: Mean = 2.31, Count = 10

**Group2:**

- **human**: Mean = 3.97, Count = 12

