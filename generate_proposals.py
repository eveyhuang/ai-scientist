#!/usr/bin/env python3
"""
Script to generate full research proposals from generated research ideas.
Reads ideas from generated_ideas subfolders and creates comprehensive proposals.
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any
import logging
from pathlib import Path

from ai_models_interface import AIModelsInterface
from prompt_templates import PromptManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProposalGenerator:
    """Generate full research proposals from research ideas"""
    
    def __init__(self, config_path: str = "config.env"):
        """Initialize the proposal generator"""
        self.ai_interface = AIModelsInterface(config_path)
        self.base_dir = Path("generated_ideas")
        
        # Initialize prompt manager for proposal templates
        self.prompt_manager = PromptManager()
        
        # Load the research call from the original proposals file
        self.research_call = self._load_research_call()
    
    def _load_research_call(self) -> str:
        """Load the research call from the original proposals file"""
        try:
            with open("human-proposals-y1.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('call', '')
        except Exception as e:
            logger.error(f"Error loading research call: {e}")
            return ''
    
    def load_ideas_from_template(self, template_name: str) -> List[Dict[str, Any]]:
        """Load all research ideas from a specific template folder"""
        template_dir = self.base_dir / template_name / "processed_ideas"
        
        if not template_dir.exists():
            logger.warning(f"Template directory not found: {template_dir}")
            return []
        
        all_ideas = []
        
        # Find all idea files in the template directory
        idea_files = list(template_dir.glob("*_ideas.json"))
        
        for idea_file in idea_files:
            try:
                with open(idea_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract ideas from the file
                    for response in data.get('responses', []):
                        for idea in response.get('research_ideas', []):
                            idea_data = {
                                'idea_id': idea.get('idea_id', ''),
                                'title': idea.get('title', ''),
                                'abstract': idea.get('abstract', ''),
                                'model_name': response.get('model_name', ''),
                                'template_name': response.get('template_name', ''),
                                'source_file': idea_file.name
                            }
                            all_ideas.append(idea_data)
                            
            except Exception as e:
                logger.error(f"Error loading ideas from {idea_file}: {e}")
        
        logger.info(f"Loaded {len(all_ideas)} ideas from {template_name}")
        return all_ideas
    
    def generate_proposal(self, idea: Dict[str, Any], template_name: str, model_name: str = None) -> Dict[str, Any]:
        """Generate a full proposal for a single research idea"""
        # Determine if this template uses roles or is a no-role template
        available_roles = self.prompt_manager.get_available_roles()
        
        # Check if template_name is a valid role or a special template
        if template_name == 'generate_ideas_no_role':
            # No role - use unified template with None
            role_to_use = None
        elif template_name in available_roles:
            # Use the unified template with the specified role
            role_to_use = template_name
        else:
            raise ValueError(f"Invalid template '{template_name}'. Available roles: {available_roles}, or use 'generate_ideas_no_role'")
        
        # Use specified model or default to first available
        if model_name is None:
            available_models = self.ai_interface.get_available_models()
            if not available_models:
                raise ValueError("No AI models available")
            model_name = available_models[0]
        
        # Format the proposal template with the idea using PromptManager
        # Always use 'generate_proposals' - it now handles both with-role and no-role cases
        proposal_prompt = self.prompt_manager.format_prompt(
            'generate_proposals',
            {
                'title': idea['title'],
                'abstract': idea['abstract'],
                'research_call': self.research_call
            },
            role_to_use  # Pass None for no-role, or the role name
        )
        
        # Generate the proposal
        try:
            response = self.ai_interface.generate_research_ideas(
                research_call=proposal_prompt,
                model_name=model_name,
                prompt_template='generate_proposals'  # Always use the unified template
            )
            
            # Parse the JSON response
            proposal_data = json.loads(response.generated_ideas)
            
            return {
                'idea_id': idea['idea_id'],
                'original_title': idea['title'],
                'original_abstract': idea['abstract'],
                'model_name': model_name,
                'template_name': template_name,
                'generation_timestamp': datetime.now().isoformat(),
                'proposal': proposal_data.get('proposal', {})
            }
            
        except Exception as e:
            logger.error(f"Error generating proposal for idea {idea['idea_id']}: {e}")
            return {
                'idea_id': idea['idea_id'],
                'original_title': idea['title'],
                'original_abstract': idea['abstract'],
                'model_name': model_name,
                'template_name': template_name,
                'generation_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def generate_proposals_for_template(self, template_name: str, model_name: str = None, max_proposals: int = None):
        """Generate proposals for all ideas in a template folder"""
        logger.info(f"Generating proposals for template: {template_name}")
        
        # Load ideas
        ideas = self.load_ideas_from_template(template_name)
        
        if not ideas:
            logger.warning(f"No ideas found for template: {template_name}")
            return
        
        # Group ideas by model to ensure consistency
        ideas_by_model = {}
        for idea in ideas:
            model = idea.get('model_name', 'unknown')
            if model not in ideas_by_model:
                ideas_by_model[model] = []
            ideas_by_model[model].append(idea)
        
        logger.info(f"Found ideas from models: {list(ideas_by_model.keys())}")
        
        # If specific model requested, only process ideas from that model
        if model_name:
            if model_name not in ideas_by_model:
                logger.warning(f"No ideas found for model {model_name}. Available models: {list(ideas_by_model.keys())}")
                return
            ideas_by_model = {model_name: ideas_by_model[model_name]}
        
        # Process each model separately
        all_proposals = []
        for model, model_ideas in ideas_by_model.items():
            logger.info(f"Processing {len(model_ideas)} ideas from model: {model}")
            
            # Limit number of proposals if specified
            if max_proposals:
                model_ideas = model_ideas[:max_proposals]
            
            # Generate proposals using the same model that generated the ideas
            proposals = []
            for i, idea in enumerate(model_ideas, 1):
                logger.info(f"Generating proposal {i}/{len(model_ideas)}: {idea['title'][:50]}...")
                
                proposal = self.generate_proposal(idea, template_name, model)  # Use same model
                proposals.append(proposal)
            
            # Save proposals for this model
            self._save_proposals(proposals, template_name, model)
            all_proposals.extend(proposals)
        
        logger.info(f"Generated {len(all_proposals)} total proposals for {template_name}")
    
    def _save_proposals(self, proposals: List[Dict[str, Any]], template_name: str, model_name: str = None):
        """Save generated proposals to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Create proposals directory
        proposals_dir = self.base_dir / template_name / "proposals"
        proposals_dir.mkdir(exist_ok=True)
        
        # Get model name from first proposal if not provided
        if model_name is None:
            model_name = proposals[0].get('model_name', 'unknown') if proposals else 'unknown'
        
        # Save proposals with same session ID format as generate_research_ideas.py
        session_id = f"{model_name}_{template_name}"
        filename = f"proposals_{session_id}_{timestamp}.json"
        filepath = proposals_dir / filename
        
        proposals_data = {
            'session_id': session_id,
            'template_name': template_name,
            'generation_timestamp': datetime.now().isoformat(),
            'total_proposals': len(proposals),
            'proposals': proposals
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(proposals_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(proposals)} proposals to {filepath}")
    
    def list_available_templates(self) -> List[str]:
        """List all available template folders"""
        if not self.base_dir.exists():
            return []
        
        templates = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "processed_ideas").exists():
                templates.append(item.name)
        
        return templates
    
    def list_available_models_for_template(self, template_name: str) -> List[str]:
        """List all models that have generated ideas for a specific template"""
        ideas = self.load_ideas_from_template(template_name)
        models = list(set(idea.get('model_name', 'unknown') for idea in ideas))
        return models

def main():
    """Main function to run proposal generation"""
    parser = argparse.ArgumentParser(description='Generate full research proposals from research ideas')
    parser.add_argument('--template', '-t', 
                       default='generate_ideas_no_role',
                       help='Template name to process (default: generate_ideas_no_role). Options: single_scientist, groups_of_scientists, groups_of_interdisciplinary_scientists, generate_ideas_no_role')
    parser.add_argument('--model', '-m',
                       help='AI model to use for proposal generation')
    parser.add_argument('--max-proposals', type=int,
                       help='Maximum number of proposals to generate')
    parser.add_argument('--list-templates', action='store_true',
                       help='List available templates and exit')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ProposalGenerator()
    
    # List templates if requested
    if args.list_templates:
        templates = generator.list_available_templates()
        print("Available templates:")
        for template in templates:
            models = generator.list_available_models_for_template(template)
            print(f"  - {template} (models: {', '.join(models)})")
        return
    
    # Process the specified template (now has default)
        generator.generate_proposals_for_template(
            args.template, 
                args.model, 
                args.max_proposals
            )

if __name__ == "__main__":
    main()
