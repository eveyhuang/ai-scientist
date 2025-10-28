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
        # Validate that template_name is a valid role
        if template_name not in self.prompt_manager.get_available_roles():
            raise ValueError(f"Invalid role '{template_name}'. Available: {self.prompt_manager.get_available_roles()}")
        
        # Use specified model or default to first available
        if model_name is None:
            available_models = self.ai_interface.get_available_models()
            if not available_models:
                raise ValueError("No AI models available")
            model_name = available_models[0]
        
        # Format the proposal template with the idea using PromptManager
        proposal_prompt = self.prompt_manager.format_prompt(
            'generate_proposals',
            {
                'title': idea['title'],
                'abstract': idea['abstract']
            },
            template_name  # Use template_name as the role
        )
        
        # Generate the proposal
        try:
            response = self.ai_interface.generate_research_ideas(
                research_call=proposal_prompt,
                model_name=model_name,
                prompt_template="standard_extension"  # Use a simple template for proposal generation
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
        
        # Limit number of proposals if specified
        if max_proposals:
            ideas = ideas[:max_proposals]
        
        logger.info(f"Generating {len(ideas)} proposals...")
        
        # Generate proposals
        proposals = []
        for i, idea in enumerate(ideas, 1):
            logger.info(f"Generating proposal {i}/{len(ideas)}: {idea['title'][:50]}...")
            
            proposal = self.generate_proposal(idea, template_name, model_name)
            proposals.append(proposal)
        
        # Save proposals
        self._save_proposals(proposals, template_name)
        
        logger.info(f"Generated {len(proposals)} proposals for {template_name}")
    
    def _save_proposals(self, proposals: List[Dict[str, Any]], template_name: str):
        """Save generated proposals to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create proposals directory
        proposals_dir = self.base_dir / template_name / "proposals"
        proposals_dir.mkdir(exist_ok=True)
        
        # Save proposals
        filename = f"proposals_{template_name}_{timestamp}.json"
        filepath = proposals_dir / filename
        
        proposals_data = {
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

def main():
    """Main function to run proposal generation"""
    parser = argparse.ArgumentParser(description='Generate full research proposals from research ideas')
    parser.add_argument('--template', '-t', 
                       help='Template name to process (e.g., single_scientist, groups_of_scientists)')
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
            print(f"  - {template}")
        return
    
    # Process specific template
    if args.template:
        generator.generate_proposals_for_template(
            args.template, 
            args.model, 
            args.max_proposals
        )
    else:
        # Process all available templates
        templates = generator.list_available_templates()
        if not templates:
            logger.error("No templates found. Run with --list-templates to see available options.")
            return
        
        logger.info(f"Processing all templates: {templates}")
        for template in templates:
            generator.generate_proposals_for_template(
                template, 
                args.model, 
                args.max_proposals
            )

if __name__ == "__main__":
    main()
