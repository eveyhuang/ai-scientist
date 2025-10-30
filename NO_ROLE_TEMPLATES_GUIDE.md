# No-Role Templates - Complete Usage Guide

## Overview

Two new templates have been added that generate research content **without any role perspective**:

1. **`generate_ideas_no_role`** - Generate research ideas without role framing
2. **`generate_proposals_no_role`** - Generate research proposals without role framing

These templates allow for neutral, unbiased content generation without "You are a scientist..." prefixes.

---

## Template Details

### 1. `generate_ideas_no_role`

**Location:** `prompt_templates.py` (lines 87-122)

**Purpose:** Generate innovative research ideas without role perspective

**Parameters:** `['research_call']` only

**Prompt starts with:**
```
Generate innovative research ideas based on the following research call.
```

**vs. Regular `generate_ideas` which starts with:**
```
You are {role_description}, you are tasked with generating innovative research ideas...
```

---

### 2. `generate_proposals_no_role`

**Location:** `prompt_templates.py` (lines 195-264)

**Purpose:** Generate comprehensive research proposals without role perspective

**Parameters:** `['title', 'abstract', 'research_call']` only

**Prompt starts with:**
```
Write a comprehensive research proposal based on the provided title and abstract...
```

**vs. Regular `generate_proposals` which starts with:**
```
You are {role_description} writing a comprehensive research proposal...
```

---

## How to Use

### Step 1: Generate Ideas Without Role

```bash
python generate_research_ideas.py \
    --template generate_ideas_no_role \
    --models gemini-2.5-pro gpt-4
```

**Output:** Ideas saved to `generated_ideas/generate_ideas_no_role/`

---

### Step 2: Generate Proposals Without Role

```bash
python generate_proposals.py \
    --template generate_ideas_no_role \
    --model gemini-2.5-pro
```

**How it works:**
- Script detects `generate_ideas_no_role` template
- Automatically uses `generate_proposals_no_role` for proposal generation
- No role description is applied to either ideas or proposals

**Output:** Proposals saved to `generated_ideas/generate_ideas_no_role/proposals/`

---

## Complete Workflow Example

```bash
# 1. Generate ideas without role
python generate_research_ideas.py --template generate_ideas_no_role --models gemini-2.5-pro

# 2. Generate proposals from those ideas (automatically uses no-role template)
python generate_proposals.py --template generate_ideas_no_role --model gemini-2.5-pro

# 3. Combine with other proposals
python combine_proposals.py

# 4. Evaluate
python evaluate_proposals.py --mode pairwise --compare-type ai-ai
```

---

## Integration Details

### `prompt_templates.py` Changes

**Added templates:**
- `generate_ideas_no_role` (lines 87-122)
- `generate_proposals_no_role` (lines 195-264)

**Updated `format_prompt()` method:**
- Only adds `role_description` if template requires it
- Templates without `role_description` in parameters skip role handling

### `ai_models_interface.py` Changes

**Updated `generate_research_ideas()` method:**
```python
# For templates ending in _no_role, don't use any role
if template_name.endswith('_no_role'):
    role_to_use = None
```

### `generate_research_ideas.py` Changes

**Added to template choices:**
```python
choices=['single_scientist', 'groups_of_scientists', 
         'groups_of_interdisciplinary_scientists', 'generate_ideas_no_role']
```

### `generate_proposals.py` Changes

**Updated `generate_proposal()` method:**
```python
if template_name == 'generate_ideas_no_role':
    proposals_template_name = 'generate_proposals_no_role'
    role_to_use = None
```

---

## Comparison Table

| Aspect | With Role | Without Role (NEW) |
|--------|-----------|-------------------|
| **Ideas Template** | `generate_ideas` | `generate_ideas_no_role` |
| **Proposals Template** | `generate_proposals` | `generate_proposals_no_role` |
| **Prompt Prefix** | "You are {role}..." | Direct instruction |
| **Role Parameter** | Required | Not used |
| **Output Directory** | `generated_ideas/{role}/` | `generated_ideas/generate_ideas_no_role/` |
| **Use Case** | Role-specific perspective | Neutral baseline |

---

## Command-Line Reference

### Generate Ideas

```bash
# With role (existing)
python generate_research_ideas.py --template single_scientist --models gemini-2.5-pro

# Without role (NEW)
python generate_research_ideas.py --template generate_ideas_no_role --models gemini-2.5-pro
```

### Generate Proposals

```bash
# With role (existing)
python generate_proposals.py --template single_scientist --model gemini-2.5-pro

# Without role (NEW)
python generate_proposals.py --template generate_ideas_no_role --model gemini-2.5-pro
```

### List Available Options

```bash
# See all available templates
python generate_research_ideas.py --help

# See all templates with ideas
python generate_proposals.py --list-templates
```

---

## Use Cases

### 1. Baseline Comparison

Generate ideas/proposals with and without role framing to compare:
- Creativity differences
- Quality differences
- Content differences

```bash
# Baseline (no role)
python generate_research_ideas.py --template generate_ideas_no_role --models gemini-2.5-pro

# With single scientist role
python generate_research_ideas.py --template single_scientist --models gemini-2.5-pro

# With group role
python generate_research_ideas.py --template groups_of_scientists --models gemini-2.5-pro
```

### 2. Control Experiments

Test whether role descriptions actually impact output:
1. Generate with no role
2. Generate with different roles
3. Compare results using evaluation scripts

### 3. Neutral Synthesis

When you want the AI to generate content without any persona bias:
- Literature reviews
- Data synthesis
- Methodological assessments

---

## File Structure

```
generated_ideas/
├── single_scientist/              # With role
│   ├── processed_ideas/
│   └── proposals/
├── groups_of_scientists/          # With role
│   ├── processed_ideas/
│   └── proposals/
└── generate_ideas_no_role/        # Without role (NEW)
    ├── processed_ideas/
    │   └── gemini-2.5-pro_generate_ideas_no_role_ideas.json
    └── proposals/
        └── proposals_gemini-2.5-pro_generate_ideas_no_role_20241030.json
```

---

## Python API Usage

### Generate Ideas Without Role

```python
from ai_models_interface import AIModelsInterface

ai = AIModelsInterface(config_path='config.env')

response = ai.generate_research_ideas(
    research_call="Your research call text...",
    model_name='gemini-2.5-pro',
    prompt_template='generate_ideas_no_role'  # No role template
)

print(response.generated_ideas)
```

### Generate Proposals Without Role

```python
from prompt_templates import PromptManager

manager = PromptManager()

prompt = manager.format_prompt(
    template_name='generate_proposals_no_role',
    data={
        'research_call': 'Your research call...',
        'title': 'Research Title',
        'abstract': 'Research abstract...'
    },
    role=None  # No role needed
)

# Use prompt with AI interface
response = ai_interface.generate_content(prompt, model_name='gemini-2.5-pro')
```

---

## Testing

### Verify Templates Work

```python
from prompt_templates import PromptManager

manager = PromptManager()

# Test ideas template
prompt_ideas = manager.format_prompt(
    'generate_ideas_no_role',
    {'research_call': 'Test call'},
    role=None
)
print("Ideas template - Contains 'You are':", "You are" in prompt_ideas)
# Should print: False

# Test proposals template
prompt_proposals = manager.format_prompt(
    'generate_proposals_no_role',
    {
        'research_call': 'Test call',
        'title': 'Test',
        'abstract': 'Test abstract'
    },
    role=None
)
print("Proposals template - Contains 'You are':", "You are" in prompt_proposals)
# Should print: False
```

---

## Troubleshooting

### Error: "Role 'generate_ideas_no_role' not found"

**Cause:** Old code trying to use template name as a role

**Solution:** Update to latest versions of:
- `prompt_templates.py`
- `ai_models_interface.py`
- `generate_research_ideas.py`
- `generate_proposals.py`

### Error: "Template 'generate_ideas_no_role' not found"

**Cause:** Template not in `prompt_templates.py`

**Solution:** Ensure you have the latest `prompt_templates.py` with both no-role templates

### Proposals still have "You are..." prefix

**Cause:** Using wrong template in `generate_proposals.py`

**Solution:** Script should automatically detect `generate_ideas_no_role` and use `generate_proposals_no_role`. Check line 92-95 in `generate_proposals.py`

---

## Summary

✅ **Two new templates added:**
- `generate_ideas_no_role` for neutral idea generation
- `generate_proposals_no_role` for neutral proposal generation

✅ **Fully integrated with existing scripts:**
- `generate_research_ideas.py` has the option
- `generate_proposals.py` automatically uses correct template
- `combine_proposals.py` includes them
- `evaluate_proposals.py` can evaluate them

✅ **Backward compatible:**
- All existing role-based templates still work
- No breaking changes to existing workflows

✅ **Use cases:**
- Baseline comparisons
- Control experiments  
- Neutral content generation

Perfect for studying the impact of role framing on AI-generated research content! 🚀

