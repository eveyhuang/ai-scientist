# Usage Examples - Updated Defaults

All scripts now default to `generate_ideas_no_role` template (no role descriptions).

## Quick Commands

### 1. Generate Ideas (No Role - Default)
```bash
python generate_research_ideas.py
# Uses generate_ideas_no_role by default
```

### 2. Generate Proposals from Those Ideas
```bash
python generate_proposals.py
# Uses generate_ideas_no_role by default
```

### 3. Combine All Proposals
```bash
python combine_proposals.py
```

### 4. Evaluate Proposals (AI-AI, Gemini only, No Role)
```bash
# Pairwise evaluation
python evaluate_proposals.py \
    --mode pairwise \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gemini-2.5-pro \
    --evaluator-model gemini-2.5-pro
```

### 5. Similarity Analysis (AI-AI, Gemini only, No Role)
```bash
python evaluate_proposals_similarity.py \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gemini-2.5-pro
```

### 6. Visualize Results
```bash
# Will auto-create subfolder like: visuals/ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini/
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini_20251031_120000.json
```

---

## Detailed Examples

### Example 1: Complete Workflow for No-Role Ideas (Gemini Only)

```bash
# Step 1: Generate ideas with no role (uses gemini-2.5-pro and gpt-4 by default)
python generate_research_ideas.py

# Step 2: Generate proposals from those ideas (uses gemini-2.5-pro and gpt-4)
python generate_proposals.py

# Step 3: Combine all proposals
python combine_proposals.py

# Step 4: Evaluate Gemini proposals against each other
python evaluate_proposals.py \
    --mode pairwise \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gemini-2.5-pro \
    --evaluator-model gemini-2.5-pro

# Step 5: Analyze similarity between Gemini proposals
python evaluate_proposals_similarity.py \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gemini-2.5-pro

# Step 6: Visualize both results
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini_*.json \
    --similarity-file similarity_analysis/similarity_ai-ai_generate_ideas_no_role_genby_gemini_*.csv
```

**Output Structure:**
```
visuals/
└── ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini/
    ├── overlap_matrices_combined.png
    ├── matrix_research_question_aims.png
    ├── matrix_data_empirical_context.png
    ├── matrix_methods_design.png
    ├── matrix_intended_contribution_outcomes.png
    ├── matrix_resources_timing_artifacts.png
    ├── evaluation_statistics.txt
    ├── similarity_matrices_combined.png
    ├── matrix_tfidf_similarity.png
    ├── matrix_embedding_similarity.png
    ├── matrix_keyword_jaccard.png
    ├── matrix_topic_similarity.png
    ├── metrics_correlation.png
    ├── similarity_distributions.png
    └── similarity_statistics.txt
```

---

### Example 2: Compare GPT vs Gemini (Separate Analyses)

```bash
# Analyze GPT-generated proposals only
python evaluate_proposals.py \
    --mode pairwise \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gpt-4 \
    --evaluator-model gemini-2.5-pro

python evaluate_proposals_similarity.py \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gpt-4

# Analyze Gemini-generated proposals only
python evaluate_proposals.py \
    --mode pairwise \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gemini-2.5-pro \
    --evaluator-model gemini-2.5-pro

python evaluate_proposals_similarity.py \
    --compare-type ai-ai \
    --template generate_ideas_no_role \
    --ai-model gemini-2.5-pro
```

**This creates separate checkpoint and output files:**
- `checkpoint_ai-ai_generate_ideas_no_role_genby_gpt_eval_gemini_*.json`
- `checkpoint_ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini_*.json`
- `similarity_ai-ai_generate_ideas_no_role_genby_gpt_*.csv`
- `similarity_ai-ai_generate_ideas_no_role_genby_gemini_*.csv`

---

### Example 3: Human-AI Comparison with No-Role AI Proposals

```bash
# Compare human proposals with no-role AI proposals
python evaluate_proposals.py \
    --mode pairwise \
    --compare-type human-ai \
    --template generate_ideas_no_role \
    --evaluator-model gemini-2.5-pro

python evaluate_proposals_similarity.py \
    --compare-type human-ai \
    --template generate_ideas_no_role

# Visualize
python visualize_results.py \
    --evaluation-file evaluations/evaluations_pairwise_human-ai_generate_ideas_no_role_eval_gemini_*.json \
    --similarity-file similarity_analysis/similarity_human-ai_generate_ideas_no_role_*.csv
```

---

### Example 4: Using Other Templates (Single Scientist, Groups, etc.)

```bash
# Generate ideas with single scientist role
python generate_research_ideas.py --template single_scientist

# Generate proposals
python generate_proposals.py --template single_scientist

# Re-combine proposals (to include new ones)
python combine_proposals.py

# Evaluate
python evaluate_proposals.py \
    --mode pairwise \
    --compare-type ai-ai \
    --template single \
    --ai-model gemini-2.5-pro \
    --evaluator-model gemini-2.5-pro

# Similarity
python evaluate_proposals_similarity.py \
    --compare-type ai-ai \
    --template single \
    --ai-model gemini-2.5-pro
```

---

## Filter Specifications

### Template/Role Options:
- `generate_ideas_no_role` - No role description (new default)
- `single` - Single scientist
- `group` - Groups of scientists
- `group_int` - Groups of interdisciplinary scientists
- `None` or omit - All templates

### AI Model Options:
- `gemini-2.5-pro`
- `gpt-4`
- `None` or omit - All models

### Comparison Types:
- `human-human` - Compare human proposals
- `human-ai` - Compare human vs AI proposals
- `ai-ai` - Compare AI proposals

---

## File Naming Conventions

### Checkpoint Files:
```
checkpoint_{comparison_type}_{role}_{genby_model}_{eval_model}_{timestamp}.json
```
**Examples:**
- `checkpoint_ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini_20251031_120000.json`
- `checkpoint_human-ai_generate_ideas_no_role_eval_gemini_20251031_130000.json`

### Evaluation Files:
```
evaluations_pairwise_{comparison_type}_{timestamp}.json
```

### Similarity Files:
```
similarity_{comparison_type}_{role}_{genby_model}_{timestamp}.csv
similarity_{comparison_type}_{role}_{genby_model}_{timestamp}.json
```
**Examples:**
- `similarity_ai-ai_generate_ideas_no_role_genby_gemini_20251031_120000.csv`
- `similarity_human-ai_generate_ideas_no_role_20251031_130000.csv`

### Visual Subfolders:
```
visuals/{comparison_type}_{role}_{genby_model}_{eval_model}/
```
**Examples:**
- `visuals/ai-ai_generate_ideas_no_role_genby_gemini_eval_gemini/`
- `visuals/human-ai_generate_ideas_no_role_eval_gemini/`
- `visuals/human-human/`

---

## Tips

1. **Always run `combine_proposals.py` after generating new proposals** to update the unified CSV.

2. **Use `--template` to filter** specific role types when you have multiple templates.

3. **Use `--ai-model` to ensure same-model comparisons** (e.g., only Gemini vs Gemini).

4. **Checkpoint files are automatically created and resumed** - you can safely interrupt long-running evaluations.

5. **Output files are now descriptive** - filenames include the comparison type, role, and models used.

6. **Visualizations auto-organize into subfolders** - no need to manually organize output files.

---

## Comparison with Old Defaults

### Before (single_scientist default):
```bash
# Had to specify no_role explicitly every time
python generate_research_ideas.py --template generate_ideas_no_role
python generate_proposals.py --template generate_ideas_no_role
python evaluate_proposals.py --template generate_ideas_no_role --ai-model gemini-2.5-pro
```

### Now (no_role default):
```bash
# Defaults to no_role - much shorter!
python generate_research_ideas.py
python generate_proposals.py
python evaluate_proposals.py --compare-type ai-ai --ai-model gemini-2.5-pro
```

---

**Last Updated:** October 31, 2025

