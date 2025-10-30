# Evaluate Proposals - Complete Usage Guide

## Overview

`evaluate_proposals.py` is a comprehensive tool for evaluating research proposals using AI-powered evaluation prompts. It supports both **single proposal evaluation** and **pairwise proposal comparison**, with multiple evaluation templates and configurable evaluator perspectives.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Evaluation Modes](#evaluation-modes)
3. [Available Evaluation Templates](#available-evaluation-templates)
4. [Available Role Descriptions](#available-role-descriptions)
5. [Command-Line Arguments](#command-line-arguments)
6. [Usage Examples](#usage-examples)
7. [Output Files](#output-files)
8. [Advanced Usage](#advanced-usage)

---

## Quick Start

### List Available Options
```bash
# List all evaluation templates
python evaluate_proposals.py --list-templates

# List all role descriptions
python evaluate_proposals.py --list-roles
```

### Basic Evaluation
```bash
# Evaluate all proposals with comprehensive template
python evaluate_proposals.py --mode single --eval-templates comprehensive

# Compare human vs AI proposals
python evaluate_proposals.py --mode pairwise --compare-type human-ai
```

---

## Evaluation Modes

### 1. Single Proposal Evaluation (`--mode single`)
Evaluates individual proposals independently. Each proposal is assessed against specific criteria without comparison to other proposals.

**Use cases:**
- Assess scientific merit of proposals
- Identify strengths and weaknesses
- Score proposals on multiple dimensions
- Generate structured feedback

### 2. Pairwise Comparison (`--mode pairwise`)
Compares two proposals to assess overlap and similarity across multiple dimensions.

**Use cases:**
- Detect duplicate or highly similar proposals
- Assess competitive overlap between proposals
- Compare human vs AI-generated proposals
- Identify redundancies in research aims

---

## Available Evaluation Templates

### Single Proposal Templates (`--mode single`)

#### 1. **comprehensive**
**Template Key:** `eval_comprehensive`  
**Description:** Comprehensive evaluation across 10 criteria with structured JSON output

**Criteria Evaluated (1-5 scale):**
1. Scientific Merit
2. Alignment with Call
3. Methodology
4. Innovation
5. Feasibility
6. Data Synthesis Approach
7. Collaborative Approach
8. Open Science Commitment
9. Training Opportunities
10. Impact

**Output:** Overall score (average), overall assessment, and detailed scores with justifications for each criterion.

**Example:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive \
  --source both
```

---

#### 2. **strengths_weaknesses**
**Template Key:** `eval_strengths_weaknesses`  
**Description:** Detailed pros/cons analysis with structured JSON output

**Components:**
- Overall score (1-5)
- Recommendation (recommend/recommend with revisions/do not recommend)
- List of strengths (detailed descriptions)
- List of weaknesses (detailed descriptions)
- Alignment with priorities
- Recommendations for improvement
- Summary

**Example:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates strengths_weaknesses \
  --roles "program officer evaluating grant applications"
```

---

#### 3. **innovation_assessment**
**Template Key:** `eval_innovation_assessment`  
**Description:** Focus on novelty and innovation with structured JSON output

**Criteria Evaluated (1-5 scale):**
1. Novelty of Research Questions
2. Methodological Innovation
3. Data Integration Innovation
4. Potential for New Insights
5. Risk vs Reward Balance

**Output:** Overall innovation score, overall assessment, and detailed scores for each innovation criterion.

**Example:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates innovation_assessment \
  --source ai \
  --max-proposals 10
```

---

#### 4. **alignment_with_call**
**Template Key:** `eval_alignment_with_call`  
**Description:** Evaluate fit with funding requirements with structured JSON output

**Criteria Evaluated (1-5 scale):**
1. Community-Scale Synthesis
2. Collaboration Requirements
3. Transdisciplinary Approach
4. Compelling Scientific Question
5. Open Science
6. Training Component
7. Need for Support

**Output:** Overall alignment score, overall assessment, and detailed scores for each alignment criterion.

**Example:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates alignment_with_call \
  --source human
```

---

#### 5. **human_criteria**
**Template Key:** `eval_human_criteria`  
**Description:** Evaluate proposal based on human reviewer criteria with detailed scoring

**Categories & Subcriteria (1-5 scale):**

**1. Scientific Merit and Innovation**
   - Relevance to Emergent Phenomena
   - Novelty & Significance
   - Rigor of Approach

**2. Feasibility**
   - Scope & Timeline

**3. Data Sources and Limitations**
   - Synthesis Focus
   - Data Identification

**4. Open Science Compliance**
   - Open Science Commitment

**Output:** Category averages, final numeric score (average of category averages), and narrative summary (1-2 paragraphs).

**Example:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates human_criteria \
  --roles "expert scientific reviewer" \
  --source both
```

---

### Pairwise Comparison Templates (`--mode pairwise`)

#### 1. **proposal_overlap**
**Template Key:** `eval_proposal_overlap`  
**Description:** Compare two proposals on overlap dimensions with scoring

**Dimensions Evaluated (0-4 scale):**

1. **Research Question / Aims** (0-4)
   - 0 = Clearly different aims
   - 1 = Related theme, different core question
   - 2 = Overlapping aim with distinct sub-questions
   - 3 = Near-identical aims/hypotheses
   - 4 = Identical aims/hypotheses

2. **Data / Empirical Context** (0-4)
   - 0 = Different population/dataset/context
   - 1 = Same broad domain, different sample
   - 2 = Partial overlap
   - 3 = Same dataset with minor differences
   - 4 = Same dataset/population/site

3. **Methods / Design** (0-4)
   - 0 = Different methodological families
   - 1 = Different designs addressing similar question
   - 2 = Same family, different design details
   - 3 = Same design and similar analysis plan
   - 4 = Same design, measures, instruments, and analysis plan

4. **Intended Contribution / Outcomes** (0-4)
   - 0 = Distinct contributions/literatures
   - 1 = Adjacent literatures
   - 2 = Same literature, different claimed gap
   - 3 = Same gap/contribution
   - 4 = Duplicate novelty claim

5. **Resources / Timing / Artifacts** (0-4)
   - 0 = Unique partners/resources/timeline
   - 1 = Partial overlap
   - 2 = Multiple shared resources
   - 3 = Same partners/instruments/timeline
   - 4 = Effectively the same operational plan

**Output:** Overall assessment, scores and justifications for each dimension.

**Example:**
```bash
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type human-ai \
  --eval-templates proposal_overlap
```

---

## Available Role Descriptions

Use `--roles` flag with single proposal evaluation to specify the perspective of the evaluator:

1. **"expert scientific reviewer"** (default)
   - General expert perspective
   - Balanced evaluation across all criteria

2. **"program officer evaluating grant applications"**
   - Focus on funding priorities
   - Emphasis on alignment with call

3. **"interdisciplinary scientist"**
   - Cross-disciplinary perspective
   - Focus on integration across fields

4. **"data science expert"**
   - Focus on data and computational methods
   - Emphasis on analytical rigor

5. **"molecular biologist"**
   - Domain-specific perspective
   - Focus on biological relevance

6. **"computational biologist"**
   - Focus on computational approaches
   - Emphasis on methodological innovation

7. **"methodological expert in synthesis research"**
   - Focus on synthesis approaches
   - Emphasis on data integration

8. **"early career researcher"**
   - Fresh perspective
   - Focus on feasibility and training

9. **"senior principal investigator"**
   - Experienced perspective
   - Focus on impact and significance

**Example with multiple roles:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive \
  --roles "expert scientific reviewer" "interdisciplinary scientist" "data science expert"
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--mode` | `single` \| `pairwise` | Evaluation mode |

### Data Source Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `all_proposals_combined.csv` | Path to combined proposals CSV file |
| `--source` | `both` | Which proposals to evaluate: `human`, `ai`, or `both` (single mode only) |
| `--compare-type` | `human-ai` | Type of comparison: `human-human`, `ai-ai`, or `human-ai` (pairwise mode only) |
| `--template` | `single_scientist` | Filter AI proposals by role: `single`, `group`, `group_int` |
| `--ai-model` | `None` | Specific AI model name to filter proposals (e.g., `gemini-2.5-pro`) |
| `--max-proposals` | `None` | Maximum number of proposals to evaluate |

### Evaluation Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--evaluator-model` | `gemini-2.5-pro` | AI model to use for evaluation |
| `--eval-templates` | varies by mode | Evaluation templates to use (can specify multiple for single mode) |
| `--roles` | `["expert scientific reviewer"]` | Role descriptions for evaluators (single mode only, can specify multiple) |

### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `None` | Custom output filename for evaluations |

### Utility Arguments

| Argument | Description |
|----------|-------------|
| `--list-templates` | List available evaluation templates and exit |
| `--list-roles` | List available role descriptions and exit |

---

## Usage Examples

### Basic Single Proposal Evaluation

```bash
# Evaluate all proposals with default comprehensive template
python evaluate_proposals.py --mode single

# Evaluate only human proposals
python evaluate_proposals.py --mode single --source human

# Evaluate only AI proposals
python evaluate_proposals.py --mode single --source ai
```

### Multiple Evaluation Templates

```bash
# Use multiple templates
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive strengths_weaknesses innovation_assessment \
  --source both

# Evaluate with all single-proposal templates
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive strengths_weaknesses innovation_assessment alignment_with_call human_criteria
```

### Multiple Role Perspectives

```bash
# Evaluate from multiple perspectives
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive \
  --roles "expert scientific reviewer" "program officer evaluating grant applications" "interdisciplinary scientist"
```

### Filtered Evaluations

```bash
# Evaluate only proposals from a specific AI model
python evaluate_proposals.py \
  --mode single \
  --source ai \
  --ai-model gemini-2.5-pro

# Evaluate only "single scientist" AI proposals
python evaluate_proposals.py \
  --mode single \
  --source ai \
  --template single

# Evaluate only first 5 proposals
python evaluate_proposals.py \
  --mode single \
  --max-proposals 5
```

### Pairwise Comparisons

```bash
# Compare human vs AI proposals (default)
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type human-ai

# Compare human proposals with each other
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type human-human

# Compare AI proposals with each other
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type ai-ai

# Compare specific AI model proposals with human proposals
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type human-ai \
  --ai-model gpt-4 \
  --template single
```

### Custom Output

```bash
# Specify custom output filename
python evaluate_proposals.py \
  --mode single \
  --eval-templates human_criteria \
  --output my_evaluation_results.json
```

### Using Custom CSV File

```bash
# Use a custom proposals CSV file
python evaluate_proposals.py \
  --mode single \
  --csv /path/to/custom_proposals.csv \
  --eval-templates comprehensive
```

### Complex Workflow Example

```bash
# Comprehensive evaluation workflow:
# 1. Evaluate all proposals with multiple templates and perspectives
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive human_criteria innovation_assessment \
  --roles "expert scientific reviewer" "interdisciplinary scientist" \
  --source both \
  --output comprehensive_evaluation.json

# 2. Compare all human vs AI proposals for overlap
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type human-ai \
  --eval-templates proposal_overlap \
  --output overlap_analysis.json

# 3. Compare AI proposals from different models
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type ai-ai \
  --ai-model gemini-2.5-pro \
  --max-proposals 10 \
  --output ai_overlap.json
```

---

## Output Files

The script generates two types of output files in the `evaluations/` directory:

### 1. Evaluation Results (JSON)

**Filename format:** `evaluations_{evaluation_type}_{timestamp}.json`

**Structure:**
```json
{
  "metadata": {
    "evaluation_type": "single_both | pairwise_human-ai | etc.",
    "total_evaluations": 120,
    "generation_timestamp": "2025-10-30T14:30:00"
  },
  "evaluations": [
    {
      "evaluation_id": "...",
      "proposal_id": "...",
      "proposal_title": "...",
      "evaluator_model": "gemini-2.5-pro",
      "evaluation_template": "comprehensive",
      "evaluation_response": "{...JSON evaluation...}",
      ...
    }
  ]
}
```

### 2. Summary Report (Markdown)

**Filename format:** `summary_{evaluation_type}_{timestamp}.md`

**Contents:**
- Total number of evaluations
- Evaluations grouped by template
- Sample evaluations (first 5 per template)
- Proposal information and evaluator details

---

## Advanced Usage

### Batch Processing Strategy

For large-scale evaluations, consider breaking the work into batches:

```bash
# Batch 1: Evaluate human proposals
python evaluate_proposals.py \
  --mode single \
  --source human \
  --eval-templates comprehensive human_criteria \
  --output batch1_human.json

# Batch 2: Evaluate AI proposals from gemini
python evaluate_proposals.py \
  --mode single \
  --source ai \
  --ai-model gemini-2.5-pro \
  --eval-templates comprehensive human_criteria \
  --output batch2_ai_gemini.json

# Batch 3: Evaluate AI proposals from gpt
python evaluate_proposals.py \
  --mode single \
  --source ai \
  --ai-model gpt-4 \
  --eval-templates comprehensive human_criteria \
  --output batch3_ai_gpt.json

# Batch 4: Compare all combinations
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type human-ai \
  --output batch4_comparisons.json
```

### Combining Templates for Different Purposes

**For Initial Screening:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates alignment_with_call \
  --source both
```

**For Detailed Review:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive strengths_weaknesses human_criteria \
  --roles "expert scientific reviewer" "program officer evaluating grant applications"
```

**For Innovation Focus:**
```bash
python evaluate_proposals.py \
  --mode single \
  --eval-templates innovation_assessment \
  --roles "interdisciplinary scientist" "senior principal investigator"
```

**For Overlap Detection:**
```bash
python evaluate_proposals.py \
  --mode pairwise \
  --compare-type ai-ai \
  --eval-templates proposal_overlap
```

### Integration with Analysis Scripts

The JSON output can be processed by `analyze_evaluations.py` or custom scripts:

```bash
# Generate evaluations
python evaluate_proposals.py \
  --mode single \
  --eval-templates comprehensive \
  --output my_evaluations.json

# Analyze results
python analyze_evaluations.py evaluations/my_evaluations.json
```

---

## Data Source: all_proposals_combined.csv

The script loads proposals from `all_proposals_combined.csv` (or custom CSV specified with `--csv`). This file is generated by `combine_proposals.py` and contains:

**Columns:**
- `proposal_id`: Unique identifier
- `title`: Proposal title
- `abstract`: Proposal abstract
- `full_draft`: Complete proposal text
- `authors`: Authors (for human proposals)
- `who`: Source (`human` or `ai`)
- `role`: Role type (`human`, `single`, `group`, `group_int`)
- `model`: Model name (for AI proposals)

**Filtering Logic:**
- `--source`: Filters by `who` column
- `--template`: Filters by `role` column
- `--ai-model`: Filters by `model` column
- `--max-proposals`: Limits number of proposals loaded

---

## Troubleshooting

### Common Issues

**1. "No proposals loaded"**
- Check that `all_proposals_combined.csv` exists
- Verify filters aren't too restrictive
- Use `--csv` to specify correct file path

**2. "Template not found"**
- Use `--list-templates` to see available templates
- Check spelling of template name
- Ensure template is appropriate for mode (single vs pairwise)

**3. API/Model Errors**
- Verify `config.env` is properly configured
- Check API keys and quotas
- Ensure evaluator model is available

**4. Memory Issues with Large Evaluations**
- Use `--max-proposals` to limit batch size
- Process proposals in smaller batches
- Consider filtering by specific criteria

---

## Tips and Best Practices

1. **Start Small**: Use `--max-proposals 5` for testing before full runs
2. **List Options First**: Always run `--list-templates` and `--list-roles` to see available options
3. **Meaningful Output Names**: Use `--output` with descriptive names for different evaluation runs
4. **Multiple Perspectives**: Use multiple `--roles` to get diverse feedback
5. **Template Combinations**: Combine templates for comprehensive analysis
6. **Monitor Progress**: Check logs for evaluation progress and any errors
7. **Save Outputs**: Store evaluation results for future analysis
8. **Version Control**: Keep track of which proposals and templates were used

---

## Summary of All Templates

### Single Proposal Evaluation Templates
1. `comprehensive` - 10 criteria, overall assessment
2. `strengths_weaknesses` - Pros/cons with recommendations
3. `innovation_assessment` - 5 innovation-focused criteria
4. `alignment_with_call` - 7 funding call alignment criteria
5. `human_criteria` - 7 subcriteria in 4 categories with narrative summary

### Pairwise Comparison Templates
1. `proposal_overlap` - 5 dimensions comparing two proposals (0-4 scale)

### Total Possible Combinations
- Single mode: 5 templates × 9 roles = 45 evaluation perspectives per proposal
- Pairwise mode: 1 template for comparing any two proposals

---

## Version Info

- **Script**: `evaluate_proposals.py`
- **Prompt Templates**: `prompt_templates.py`
- **Data Source**: `all_proposals_combined.csv` (generated by `combine_proposals.py`)
- **Default Evaluator Model**: `gemini-2.5-pro`

For questions or issues, refer to the inline documentation in `evaluate_proposals.py` or check the template definitions in `prompt_templates.py`.

