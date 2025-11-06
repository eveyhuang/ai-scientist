#!/usr/bin/env python3
"""
Main script to generate research ideas from proposals using multiple AI models.
"""

import json
import asyncio
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any
import logging
from pathlib import Path

from ai_models_interface import AIModelsInterface, AIResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchIdeaGenerator:
    """Main class for generating and managing research ideas"""
    
    def __init__(self, config_path: str = "config.env", prompt_template: str = "generate_ideas_no_role"):
        """Initialize the generator"""
        self.ai_interface = AIModelsInterface(config_path, prompt_template)
        self.prompt_template = prompt_template
        self.output_dir = Path("generated_ideas")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization with template-based folders
        self.template_dir = self.output_dir / prompt_template
        self.template_dir.mkdir(exist_ok=True)
        
        (self.template_dir / "raw_responses").mkdir(exist_ok=True)
        (self.template_dir / "processed_ideas").mkdir(exist_ok=True)
    
    def _get_template_dir_for_diverse(self):
        """Get or create the template directory for diverse ideas"""
        diverse_template_dir = self.output_dir / "generate_diverse_ideas"
        diverse_template_dir.mkdir(exist_ok=True)
        (diverse_template_dir / "raw_responses").mkdir(exist_ok=True)
        (diverse_template_dir / "processed_ideas").mkdir(exist_ok=True)
        return diverse_template_dir

    
    def load_research_call(self, file_path: str) -> str:
        """Load research call from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                research_call = data.get('call', '')
                logger.info(f"Loaded research call from {file_path}")
                return research_call
        except Exception as e:
            logger.error(f"Error loading research call: {e}")
            return ""
    
    async def generate_ideas_for_research_call(self, research_call: str, 
                                              models: List[str] = None,
                                              **kwargs) -> List[AIResponse]:
        """Generate ideas for the research call using specified models"""
        if models is None:
            models = self.ai_interface.get_available_models()
        
        logger.info(f"Generating ideas for research call using models: {models}")
        
        responses = []
        for model_name in models:
            try:
                response = self.ai_interface.generate_research_ideas(
                    research_call, model_name, **kwargs
                )
                responses.append(response)
                logger.info(f"Generated ideas using {model_name}")
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
        
        return responses
    
    async def generate_ideas_for_research_call_main(self, research_call: str, 
                                                   models: List[str] = None,
                                                   **kwargs) -> List[AIResponse]:
        """Generate ideas for the research call"""
        if models is None:
            models = self.ai_interface.get_available_models()
        
        logger.info("Processing research call")
        
        all_responses = []
        
        # Generate ideas for each model and save individually
        for model_name in models:
            try:
                response = self.ai_interface.generate_research_ideas(
                    research_call, model_name, **kwargs
                )
                all_responses.append(response)
                
                # Save individual response with model and template info
                session_id = f"{model_name}_{response.metadata.get('prompt_template', 'unknown')}"
                self._save_responses(session_id, [response])
                
                logger.info(f"Generated ideas using {model_name}")
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
        
        return all_responses
    
    def _save_responses(self, session_id: str, responses: List[AIResponse]):
        """Save responses to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw responses
        raw_data = {
            'session_id': session_id,
            'timestamp': timestamp,
            'responses': []
        }
        
        for response in responses:
            raw_data['responses'].append({
                'model_name': response.model_name,
                'generated_ideas': response.generated_ideas,
                'timestamp': response.timestamp,
                'metadata': response.metadata
            })
        
        # Save to file
        filename = f"{session_id}_ideas_{timestamp}.json"
        filepath = self.template_dir / "raw_responses" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved responses for session {session_id} to {filepath}")
        
        # Also save processed ideas as individual JSON files
        self._save_processed_ideas(session_id, responses)
    
    def _save_processed_ideas(self, session_id: str, responses: List[AIResponse]):
        """Parse and save all research ideas from a session as one consolidated JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create consolidated data structure
        consolidated_data = {
            'session_id': session_id,
            'timestamp': timestamp,
            'generation_timestamp': datetime.now().isoformat(),
            'total_responses': len(responses),
            'responses': []
        }
        
        for response in responses:
            try:
                # Parse the JSON response
                ideas_data = json.loads(response.generated_ideas)
                
                if 'research_ideas' in ideas_data:
                    # Create structured data for this response
                    response_data = {
                        'model_name': response.model_name,
                        'template_name': response.metadata.get('prompt_template', 'unknown'),
                        'timestamp': response.timestamp,
                        'metadata': {
                            'temperature': response.metadata.get('temperature', 'unknown'),
                            'max_tokens': response.metadata.get('max_tokens', 'unknown'),
                            'parsed_successfully': response.metadata.get('parsed_successfully', False),
                            'research_call': response.metadata.get('research_call', '')
                        },
                        'research_ideas': []
                    }
                    
                    # Process each idea in this response
                    for i, idea in enumerate(ideas_data['research_ideas']):
                        idea_data = {
                            'idea_id': f"{session_id}_{response.model_name}_{i+1:02d}",
                            'title': idea.get('title', ''),
                            'abstract': idea.get('abstract', '')
                        }
                        response_data['research_ideas'].append(idea_data)
                    
                    consolidated_data['responses'].append(response_data)
                    logger.info(f"Processed {len(ideas_data['research_ideas'])} ideas from {response.model_name}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for session {session_id}: {e}")
            except Exception as e:
                logger.error(f"Error processing ideas for session {session_id}: {e}")
        
        # Save consolidated file
        filename = f"{session_id}_ideas_{timestamp}.json"
        filepath = self.template_dir / "processed_ideas" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved consolidated ideas for session {session_id} to {filepath}")
    
    def process_and_save_all(self, research_call: str, 
                            models: List[str] = None,
                            **kwargs):
        """Process research call and save results"""
        # Run async function
        all_responses = asyncio.run(
            self.generate_ideas_for_research_call_main(research_call, models, **kwargs)
        )
        
        # Create summary
        self._create_summary(all_responses, research_call)
        
        # Create consolidated ideas summary
        self._create_ideas_summary()
        
        return all_responses
    
    def _create_summary(self, all_responses: List[AIResponse], 
                       research_call: str):
        """Create a summary of all generated ideas"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'research_call_length': len(research_call),
            'available_models': self.ai_interface.get_available_models(),
            'total_responses': len(all_responses),
            'models_used': list(set(r.model_name for r in all_responses)),
            'templates_used': list(set(r.metadata.get('prompt_template', 'unknown') for r in all_responses)),
            'sessions': []
        }
        
        for response in all_responses:
            session_summary = {
                'session_id': f"{response.model_name}_{response.metadata.get('prompt_template', 'unknown')}",
                'model_name': response.model_name,
                'template_name': response.metadata.get('prompt_template', 'unknown'),
                'timestamp': response.timestamp,
                'ideas_length': len(response.generated_ideas),
                'word_count': len(response.generated_ideas.split()),
                'temperature': response.metadata.get('temperature', 'unknown'),
                'max_tokens': response.metadata.get('max_tokens', 'unknown')
            }
            summary['sessions'].append(session_summary)
        
        # Save summary
        summary_file = self.template_dir / "generation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created summary at {summary_file}")
    
    def _create_ideas_summary(self):
        """Create a consolidated summary of all processed ideas"""
        processed_ideas_dir = self.template_dir / "processed_ideas"
        
        if not processed_ideas_dir.exists():
            logger.warning("No processed ideas directory found")
            return
        
        # Collect all processed idea files
        processed_files = list(processed_ideas_dir.glob("processed_ideas_*.json"))
        
        if not processed_files:
            logger.warning("No processed idea files found")
            return
        
        # Load and organize all ideas
        all_ideas = []
        sessions_data = []
        
        for processed_file in processed_files:
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    sessions_data.append(session_data)
                    
                    # Extract individual ideas from this session
                    for response in session_data.get('responses', []):
                        for idea in response.get('research_ideas', []):
                            all_ideas.append({
                                'idea_id': idea.get('idea_id', ''),
                                'title': idea.get('title', ''),
                                'abstract': idea.get('abstract', ''),
                                'model_name': response.get('model_name', 'unknown'),
                                'template_name': response.get('template_name', 'unknown'),
                                'session_id': session_data.get('session_id', 'unknown'),
                                'timestamp': response.get('timestamp', '')
                            })
            except Exception as e:
                logger.warning(f"Failed to load processed file {processed_file}: {e}")
        
        # Create consolidated summary
        ideas_summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_ideas': len(all_ideas),
            'total_sessions': len(sessions_data),
            'sessions': {},
            'models': {},
            'templates': {},
            'ideas': all_ideas
        }
        
        # Organize by session, model, and template
        for idea in all_ideas:
            session_id = idea.get('session_id', 'unknown')
            model_name = idea.get('model_name', 'unknown')
            template_name = idea.get('template_name', 'unknown')
            
            # Count by session
            if session_id not in ideas_summary['sessions']:
                ideas_summary['sessions'][session_id] = 0
            ideas_summary['sessions'][session_id] += 1
            
            # Count by model
            if model_name not in ideas_summary['models']:
                ideas_summary['models'][model_name] = 0
            ideas_summary['models'][model_name] += 1
            
            # Count by template
            if template_name not in ideas_summary['templates']:
                ideas_summary['templates'][template_name] = 0
            ideas_summary['templates'][template_name] += 1
        
        # Save consolidated summary
        summary_file = self.template_dir / "processed_ideas" / "all_ideas_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(ideas_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created ideas summary at {summary_file}")
        logger.info(f"Total ideas processed: {len(all_ideas)}")
        logger.info(f"Total sessions: {len(sessions_data)}")
    
    def load_existing_ideas(self, model_name: str, template_name: str = "generate_ideas_no_role") -> List[Dict[str, Any]]:
        """
        Load existing ideas generated by a specific model and template
        
        Args:
            model_name: The model that generated the ideas (e.g., 'claude-sonnet-4-5')
            template_name: The template used (default: 'generate_ideas_no_role')
        
        Returns:
            List of idea dictionaries with 'title' and 'abstract'
        """
        processed_ideas_dir = self.output_dir / template_name / "processed_ideas"
        
        if not processed_ideas_dir.exists():
            logger.warning(f"Processed ideas directory not found: {processed_ideas_dir}")
            return []
        
        # Find all files matching the model and template
        pattern = f"{model_name}_{template_name}_ideas_*.json"
        matching_files = list(processed_ideas_dir.glob(pattern))
        
        if not matching_files:
            logger.warning(f"No existing ideas found for model '{model_name}' with template '{template_name}'")
            logger.info(f"Searched for pattern: {pattern} in {processed_ideas_dir}")
            return []
        
        logger.info(f"Found {len(matching_files)} file(s) with existing ideas for {model_name}")
        
        # Load all ideas from matching files
        all_ideas = []
        for idea_file in matching_files:
            try:
                with open(idea_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract ideas from the file structure
                    for response in data.get('responses', []):
                        for idea in response.get('research_ideas', []):
                            all_ideas.append({
                                'title': idea.get('title', ''),
                                'abstract': idea.get('abstract', '')
                            })
                
                logger.info(f"Loaded {len(data.get('responses', [{}])[0].get('research_ideas', []))} ideas from {idea_file.name}")
            except Exception as e:
                logger.error(f"Error loading ideas from {idea_file}: {e}")
        
        logger.info(f"Total ideas loaded: {len(all_ideas)}")
        return all_ideas
    
    def format_existing_ideas_for_prompt(self, ideas: List[Dict[str, Any]]) -> str:
        """
        Format existing ideas into a string for the generate_diverse_ideas template
        
        Args:
            ideas: List of idea dictionaries with 'title' and 'abstract'
        
        Returns:
            Formatted string of existing ideas
        """
        if not ideas:
            return "No existing ideas provided."
        
        formatted_ideas = []
        for i, idea in enumerate(ideas, 1):
            formatted_ideas.append(f"""
                Existing Idea {i}:
                Title: {idea.get('title', 'N/A')}
                Abstract: {idea.get('abstract', 'N/A')}
                """)
        
        return "\n".join(formatted_ideas)
    
    def generate_diverse_ideas(self, 
                              research_call: str,
                              model_name: str,
                              existing_ideas_model: str = None,
                              existing_ideas_template: str = "generate_ideas_no_role",
                              **kwargs) -> AIResponse:
        """
        Generate diverse ideas that are different from existing ideas
        
        Args:
            research_call: The research call text
            model_name: Model to use for generating NEW diverse ideas
            existing_ideas_model: Model whose existing ideas to avoid (default: same as model_name)
            existing_ideas_template: Template used for existing ideas (default: 'generate_ideas_no_role')
            **kwargs: Additional parameters for AI generation
        
        Returns:
            AIResponse containing the newly generated diverse ideas
        """
        # If no existing_ideas_model specified, use the same model
        if existing_ideas_model is None:
            existing_ideas_model = model_name
        
        logger.info(f"Generating diverse ideas using {model_name}")
        logger.info(f"Loading existing ideas from {existing_ideas_model} with template {existing_ideas_template}")
        
        # Load existing ideas
        existing_ideas = self.load_existing_ideas(existing_ideas_model, existing_ideas_template)
        
        if not existing_ideas:
            logger.warning("No existing ideas found. Will generate ideas without diversity constraint.")
            existing_ideas_str = "No existing ideas to avoid."
        else:
            existing_ideas_str = self.format_existing_ideas_for_prompt(existing_ideas)
            logger.info(f"Formatted {len(existing_ideas)} existing ideas for prompt. Example of first idea: {existing_ideas[0]}")
        
        # Get the generate_diverse_ideas template
        prompt_manager = self.ai_interface.prompt_manager
        template_obj = prompt_manager.get_template('generate_diverse_ideas')
        
        # Format the prompt with research call and existing ideas
        prompt = template_obj.template.format(
            research_call=research_call,
            existing_ideas=existing_ideas_str
        )
        
        # Generate diverse ideas using the AI model
        try:
            response = self.ai_interface.models[model_name](prompt, **kwargs)
            
            # Parse the JSON response
            parsed_ideas = self.ai_interface._parse_json_response(response)
            
            # Create AIResponse object
            ai_response = AIResponse(
                model_name=model_name,
                session_id=f'{model_name}_generate_diverse_ideas',
                generated_ideas=parsed_ideas,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'temperature': kwargs.get('temperature', 0),
                    'max_tokens': kwargs.get('max_tokens', 5000),
                    'prompt_template': 'generate_diverse_ideas',
                    'existing_ideas_model': existing_ideas_model,
                    'existing_ideas_template': existing_ideas_template,
                    'existing_ideas_count': len(existing_ideas),
                    'research_call': research_call,
                    'raw_response': response,
                    'parsed_successfully': parsed_ideas is not None
                }
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to generate_diverse_ideas template directory
            original_template_dir = self.template_dir
            self.template_dir = self._get_template_dir_for_diverse()
            
            # Save the response
            session_id = f"{model_name}_generate_diverse_ideas_{timestamp}"
            self._save_responses(session_id, [ai_response])
            
            # Restore original template directory
            self.template_dir = original_template_dir
            
            logger.info(f"Successfully generated diverse ideas using {model_name}")
            logger.info(f"Saved to: {self._get_template_dir_for_diverse()}")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating diverse ideas with {model_name}: {e}")
            raise
    
    def process_diverse_ideas(self,
                             research_call: str,
                             models: List[str],
                             existing_ideas_template: str = "generate_ideas_no_role",
                             **kwargs) -> List[AIResponse]:
        """
        Generate diverse ideas for multiple models, each avoiding its own existing ideas
        
        Args:
            research_call: The research call text
            models: List of models to generate diverse ideas for
            existing_ideas_template: Template used for existing ideas (default: 'generate_ideas_no_role')
            **kwargs: Additional parameters for AI generation
        
        Returns:
            List of AIResponse objects
        """
        all_responses = []
        
        for model_name in models:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing diverse ideas for model: {model_name}")
                logger.info(f"{'='*60}")
                
                response = self.generate_diverse_ideas(
                    research_call=research_call,
                    model_name=model_name,
                    existing_ideas_model=model_name,  # Each model avoids its own ideas
                    existing_ideas_template=existing_ideas_template,
                    **kwargs
                )
                
                all_responses.append(response)
                
            except Exception as e:
                logger.error(f"Failed to generate diverse ideas for {model_name}: {e}")
        
        logger.info(f"\nCompleted diverse idea generation for {len(all_responses)}/{len(models)} models")
        
        return all_responses

def main():
    """Main function to run the research idea generation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate research ideas using AI models')
    parser.add_argument('--template', '-t', 
                        choices=['single_scientist', 'groups_of_scientists', 'groups_of_interdisciplinary_scientists', 'generate_ideas_no_role'],
                        default='generate_ideas_no_role',
                        help='Prompt template to use (default: generate_ideas_no_role). Use generate_ideas_no_role for ideas without role description')
    parser.add_argument('--models', '-m', nargs='+',
                       help='Specific models to use (e.g., gpt-4 gemini-2.5-flash)')
    parser.add_argument('--temperature', type=float, default=0,
                       help='Temperature for generation (default: 0)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    parser.add_argument('--diverse', action='store_true',
                       help='Generate diverse ideas that are different from existing ideas')
    parser.add_argument('--existing-template', type=str, default='generate_ideas_no_role',
                       help='Template used for existing ideas to avoid (default: generate_ideas_no_role)')
    
    args = parser.parse_args()
    
    # Configuration
    PROPOSALS_FILE = "human-proposals-y1.json"
    CONFIG_FILE = "config.env"
    
    # Initialize generator with specified template
    generator = ResearchIdeaGenerator(CONFIG_FILE, args.template)
    
    # Check available models
    available_models = generator.ai_interface.get_available_models()
    
    # Handle list-models option
    if args.list_models:
        print("Available AI models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")
        print(f"\nTotal: {len(available_models)} models available")
        return
    
    # Load research call
    research_call = generator.load_research_call(PROPOSALS_FILE)
    if not research_call:
        logger.error("No research call loaded. Exiting.")
        return
    
    logger.info(f"Available models: {available_models}")
    
    if not available_models:
        logger.error("No AI models available. Check your API keys.")
        return
    
    # Determine which models to use
    if args.models:
        models_to_use = [model for model in args.models if model in available_models]
        if not models_to_use:
            logger.error(f"None of the specified models are available. Available: {available_models}")
            return
    else:
        models_to_use = available_models
    
    logger.info(f"Using models: {models_to_use}")
    logger.info(f"Using template: {args.template}")
    
    # Check if diverse mode is enabled
    if args.diverse:
        logger.info("\n" + "="*70)
        logger.info("DIVERSE IDEAS MODE: Generating ideas different from existing ones")
        logger.info("="*70)
        logger.info(f"Existing ideas template: {args.existing_template}")
        
        # Generate diverse ideas
        all_responses = generator.process_diverse_ideas(
            research_call=research_call,
            models=models_to_use,
            existing_ideas_template=args.existing_template,
            temperature=args.temperature
        )
        
        logger.info("\nDiverse idea generation completed!")
        logger.info(f"Results saved in: {generator.template_dir}")
        
    else:
        # Generate ideas for research call (standard mode)
        logger.info("Starting research idea generation...")
    
    all_responses = generator.process_and_save_all(
        research_call, 
        models=models_to_use,
        temperature=args.temperature
    )
    
    logger.info("Research idea generation completed!")
    logger.info(f"Results saved in: {generator.template_dir}")

if __name__ == "__main__":
    main()
