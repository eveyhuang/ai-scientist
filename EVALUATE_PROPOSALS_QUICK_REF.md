# Evaluate Proposals - Quick Reference Card

> **Note:** By default, evaluations run **without role descriptions** for unbiased assessment. Use `--roles` only if you need perspective-based evaluations.

## 🚀 Common Commands

```bash
# List available options
python evaluate_proposals.py --list-templates
python evaluate_proposals.py --list-roles

# Basic single evaluation
python evaluate_proposals.py --mode single --eval-templates comprehensive

# Compare human vs AI
python evaluate_proposals.py --mode pairwise --compare-type human-ai

# Multiple templates
python evaluate_proposals.py --mode single --eval-templates comprehensive human_criteria innovation_assessment

# With role descriptions (optional)
python evaluate_proposals.py --mode single --roles "expert scientific reviewer" "interdisciplinary scientist"

# Without role (default - no bias)
python evaluate_proposals.py --mode single --eval-templates comprehensive

# Filter by source
python evaluate_proposals.py --mode single --source human  # or: ai, both
python evaluate_proposals.py --mode single --source ai --ai-model gemini-2.5-pro

# Limit proposals
python evaluate_proposals.py --mode single --max-proposals 5
```

## 📋 Available Templates

### Single Proposal Evaluation
- `comprehensive` - 10 criteria comprehensive evaluation
- `strengths_weaknesses` - Pros/cons analysis with recommendations
- `innovation_assessment` - 5 innovation-focused criteria
- `alignment_with_call` - 7 funding alignment criteria  
- `human_criteria` - 7 subcriteria in 4 categories + narrative

### Pairwise Comparison
- `proposal_overlap` - 5 dimensions (0-4 scale) comparing two proposals

## 👤 Available Roles (for single mode - OPTIONAL)
**Default:** None (no role description - unbiased evaluation)

If you want to add role perspective, use `--roles`:
- `expert scientific reviewer`
- `program officer evaluating grant applications`
- `interdisciplinary scientist`
- `data science expert`
- `molecular biologist`
- `computational biologist`
- `methodological expert in synthesis research`
- `early career researcher`
- `senior principal investigator`

## 🎯 Arguments Cheat Sheet

| Argument | Values | Description |
|----------|--------|-------------|
| `--mode` | `single`, `pairwise` | **Required** - Evaluation mode |
| `--source` | `human`, `ai`, `both` | Filter proposals (single mode) |
| `--compare-type` | `human-human`, `ai-ai`, `human-ai` | Comparison type (pairwise mode) |
| `--eval-templates` | template names | Evaluation templates (space-separated) |
| `--roles` | role names | Evaluator roles (space-separated, single mode) |
| `--ai-model` | model name | Filter by specific AI model |
| `--template` | `single`, `group`, `group_int` | Filter by AI role type |
| `--max-proposals` | number | Limit number of proposals |
| `--evaluator-model` | model name | AI model for evaluation (default: gemini-2.5-pro) |
| `--csv` | file path | Custom proposals CSV (default: all_proposals_combined.csv) |
| `--output` | filename | Custom output filename |

## 📁 Output Files

Generated in `evaluations/` directory:
- `evaluations_{type}_{timestamp}.json` - Full evaluation results
- `summary_{type}_{timestamp}.md` - Summary report

## 💡 Quick Tips

1. **Start small**: Use `--max-proposals 5` for testing
2. **Check options**: Run `--list-templates` and `--list-roles` first
3. **Combine templates**: Get comprehensive analysis with multiple templates
4. **Multiple perspectives**: Use multiple roles for diverse feedback
5. **Meaningful names**: Use `--output` with descriptive filenames

## 📖 Full Documentation

See `EVALUATE_PROPOSALS_GUIDE.md` for complete documentation with detailed examples.

## 🔄 Typical Workflows

### Initial Screening (no role bias)
```bash
python evaluate_proposals.py --mode single --eval-templates alignment_with_call --source both
```

### Detailed Review (no role bias)
```bash
python evaluate_proposals.py --mode single \
  --eval-templates comprehensive human_criteria
```

### Detailed Review with Multiple Perspectives (optional)
```bash
python evaluate_proposals.py --mode single \
  --eval-templates comprehensive human_criteria \
  --roles "expert scientific reviewer" "program officer evaluating grant applications"
```

### Overlap Detection
```bash
python evaluate_proposals.py --mode pairwise --compare-type human-ai --eval-templates proposal_overlap
```

### Compare Specific Models
```bash
python evaluate_proposals.py --mode pairwise --compare-type ai-ai \
  --ai-model gemini-2.5-pro --max-proposals 10
```

