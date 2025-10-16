# AI Research Idea Generation System

A comprehensive system for generating research ideas using multiple AI models (GPT-4, Gemini, Claude, Grok, Qwen) and comparing them with original research proposals.

## Features

- **Multi-AI Model Support**: Generate ideas using GPT-4, Google Gemini, Anthropic Claude, Grok, and Qwen
- **Automated Comparison**: Compare AI-generated ideas with original proposals using various similarity metrics
- **Quality Analysis**: Analyze idea quality, methodology coverage, and interdisciplinary aspects
- **Visualization**: Create interactive charts and reports for analysis
- **Batch Processing**: Process multiple proposals simultaneously

## Quick Start

### 1. Setup

```bash
# Run the setup script
python setup.py
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 3. Generate Research Ideas

```bash
# Generate ideas for all proposals using all available models
python generate_research_ideas.py
```

### 4. Compare and Analyze

```bash
# Compare generated ideas with original proposals
python compare_ideas.py
```

## File Structure

```
AI-human scientist/
├── ai_models_interface.py      # Unified interface for AI models
├── generate_research_ideas.py  # Main script for idea generation
├── compare_ideas.py           # Comparison and analysis script
├── setup.py                   # Setup script
├── requirements.txt           # Python dependencies
├── config.env                 # API keys configuration
├── human-proposals-y1.json    # Original proposals data
├── generated_ideas/           # Output directory
│   ├── raw_responses/         # Raw AI responses
│   ├── processed_ideas/       # Processed ideas
│   └── comparisons/           # Comparison results
└── README.md                  # This file
```

## Usage Examples

### Generate Ideas for Specific Models

```python
from ai_models_interface import AIModelsInterface
import asyncio

# Initialize interface
ai_interface = AIModelsInterface("config.env")

# Load a proposal
proposal = {
    "proposal_id": "1",
    "proposal_title": "Example Research",
    "abstract": "Research abstract...",
    # ... other fields
}

# Generate ideas using specific models
async def generate_ideas():
    responses = await ai_interface.generate_ideas_for_all_models(
        proposal, 
        temperature=0.7,
        max_tokens=2000
    )
    return responses

# Run the async function
responses = asyncio.run(generate_ideas())
```

## Output Files

### Generated Ideas
- `generated_ideas/raw_responses/`: Raw JSON responses from AI models
- `generated_ideas/processed_ideas/`: Processed and cleaned ideas
- `generated_ideas/generation_summary.json`: Summary of generation process


## Customization

### Adding New AI Models

1. Add the model to `ai_models_interface.py`:
```python
def _call_new_model(self, prompt: str, **kwargs) -> str:
    # Implementation for new model
    pass
```

2. Register the model in `setup_models()`:
```python
self.models['new-model'] = self._call_new_model
```

### Custom Prompts

Modify the `_create_prompt()` method in `ai_models_interface.py` to customize how proposals are presented to AI models.

### Custom Analysis

Extend the `ResearchIdeaComparator` class to add new analysis methods and metrics.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all API keys are correctly set in `config.env`
2. **Rate Limiting**: Some APIs have rate limits; the system includes error handling
3. **Memory Issues**: For large datasets, consider processing proposals in batches
4. **Model Availability**: Not all models may be available; check API status

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```


