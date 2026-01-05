#!/usr/bin/env python3
"""
Evaluate individual research proposals
Evaluates proposals using various evaluation prompts and criteria
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import itertools
import pandas as pd

from ai_models_interface import AIModelsInterface
from prompt_templates import PromptManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProposalEvaluator:
    """Evaluate individual research proposals"""
    
    def __init__(self, config_path: str = "config.env", proposals_csv: str = "all_proposals_combined.csv", proposals_json: str = None):
        """Initialize the proposal evaluator"""
        self.ai_interface = AIModelsInterface(config_path)
        self.prompt_manager = PromptManager()
        self.evaluations_dir = Path("qualitative_evaluation")
        self.evaluations_dir.mkdir(exist_ok=True)
        
        # Load proposals from CSV or JSON
        self.proposals_csv = proposals_csv
        self.proposals_json = proposals_json
        self.proposals_df = None
        self.proposals_dict = {}  # For JSON-based lookup by proposal_id
        
        if proposals_json:
            self._load_proposals_from_json()
        else:
            self._load_proposals_from_csv()
        
        # Load research call
        self.research_call = self._load_research_call()
        
    def _load_research_call(self) -> str:
        """Load the research call from the human proposals file"""
        try:
            with open("/Users/eveyhuang/Documents/NICO/ai-scientist/human-proposals/human-proposals-y1.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('call', '')
        except Exception as e:
            logger.error(f"Error loading research call: {e}")
            return ''
    
    def _load_proposals_from_csv(self):
        """Load all proposals from the combined CSV file"""
        try:
            self.proposals_df = pd.read_csv(self.proposals_csv)
            logger.info(f"Loaded {len(self.proposals_df)} proposals from {self.proposals_csv}")
            logger.info(f"  - Human: {len(self.proposals_df[self.proposals_df['who'] == 'human'])}")
            logger.info(f"  - AI: {len(self.proposals_df[self.proposals_df['who'] == 'ai'])}")
        except Exception as e:
            logger.error(f"Error loading proposals from CSV: {e}")
            self.proposals_df = pd.DataFrame()
    
    def _load_proposals_from_json(self):
        """Load all proposals from the combined JSON file"""
        try:
            with open(self.proposals_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            proposals_list = data.get('proposals', [])
            
            # Create dictionary for fast lookup by proposal_id
            self.proposals_dict = {p['proposal_id']: p for p in proposals_list}
            
            # Also create DataFrame for compatibility
            self.proposals_df = pd.DataFrame(proposals_list)
            
            logger.info(f"Loaded {len(proposals_list)} proposals from {self.proposals_json}")
            human_count = sum(1 for p in proposals_list if p.get('who') == 'human')
            ai_count = sum(1 for p in proposals_list if p.get('who') == 'ai')
            logger.info(f"  - Human: {human_count}")
            logger.info(f"  - AI: {ai_count}")
        except Exception as e:
            logger.error(f"Error loading proposals from JSON: {e}")
            self.proposals_dict = {}
            self.proposals_df = pd.DataFrame()
    
    def load_proposal_ids_from_selection_json(self, selection_json: str) -> List[str]:
        """
        Load proposal IDs from a selection JSON file (e.g., top_22_diverse_ai_proposals.json)
        
        Args:
            selection_json: Path to JSON file containing 'selected_proposals' field
        
        Returns:
            List of proposal IDs
        """
        try:
            with open(selection_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Try different possible field names
            if 'selected_proposals' in data:
                proposal_ids = data['selected_proposals']
            elif 'proposal_ids' in data:
                proposal_ids = data['proposal_ids']
            elif 'proposals' in data and isinstance(data['proposals'], list) and len(data['proposals']) > 0:
                # Check if it's a list of strings (IDs) or list of dicts
                if isinstance(data['proposals'][0], str):
                    proposal_ids = data['proposals']
                else:
                    proposal_ids = [p.get('proposal_id') for p in data['proposals'] if p.get('proposal_id')]
            else:
                logger.error(f"Could not find proposal IDs in {selection_json}")
                return []
            
            logger.info(f"Loaded {len(proposal_ids)} proposal IDs from {selection_json}")
            return proposal_ids
        
        except Exception as e:
            logger.error(f"Error loading proposal IDs from {selection_json}: {e}")
            return []
    
    def load_proposals_by_ids(self, proposal_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load specific proposals by their IDs
        
        Args:
            proposal_ids: List of proposal IDs to load
        
        Returns:
            List of proposal dictionaries
        """
        proposals = []
        missing_ids = []
        
        for pid in proposal_ids:
            # Try dictionary lookup first (faster for JSON-loaded data)
            if self.proposals_dict and pid in self.proposals_dict:
                proposals.append(self.proposals_dict[pid])
            # Fall back to DataFrame lookup
            elif self.proposals_df is not None and len(self.proposals_df) > 0:
                matches = self.proposals_df[self.proposals_df['proposal_id'] == pid]
                if len(matches) > 0:
                    proposals.append(matches.iloc[0].to_dict())
                else:
                    missing_ids.append(pid)
            else:
                missing_ids.append(pid)
        
        if missing_ids:
            logger.warning(f"Could not find {len(missing_ids)} proposals: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")
        
        logger.info(f"Loaded {len(proposals)} proposals by ID")
        return proposals
    
    def load_proposals(self, 
                      who: str = None,
                      role: str = None, 
                      model: str = None,
                      max_proposals: int = None) -> List[Dict[str, Any]]:
        """
        Load proposals from the combined CSV with optional filters
        
        Args:
            who: Filter by source ('human' or 'ai')
            role: Filter by role ('human', 'single', 'group', 'group_int')
            model: Filter by model name
            max_proposals: Maximum number of proposals to return
        
        Returns:
            List of proposal dictionaries
        """
        if self.proposals_df is None or len(self.proposals_df) == 0:
            logger.warning("No proposals loaded")
            return []
        
        df = self.proposals_df.copy()
        
        # Apply filters
        if who:
            df = df[df['who'] == who]
        if role:
            df = df[df['role'] == role]
        if model:
            df = df[df['model'] == model]
        
        # Limit number of proposals
        if max_proposals:
            df = df.head(max_proposals)
        
        # Convert to list of dictionaries
        proposals = df.to_dict('records')
        
        logger.info(f"Loaded {len(proposals)} proposals with filters: who={who}, role={role}, model={model}")
        
        return proposals
    
    def create_evaluation_prompt(self,
                                proposal: Dict[str, Any],
                                evaluation_template: str,
                                role_description: str = None) -> str:
        """Create an evaluation prompt for a single proposal"""
        
        # Map user-friendly template names to template keys in PromptManager
        template_name_mapping = {
            "comprehensive": "eval_comprehensive",
            "strengths_weaknesses": "eval_strengths_weaknesses",
            "innovation_assessment": "eval_innovation_assessment",
            "alignment_with_call": "eval_alignment_with_call",
            "human_criteria": "eval_human_criteria"
        }
        
        # Get the template key
        template_key = template_name_mapping.get(evaluation_template, "eval_comprehensive")
        
        # Get the template from PromptManager
        try:
            template_obj = self.prompt_manager.get_template(template_key)
            template = template_obj.template
        except ValueError:
            logger.warning(f"Template '{template_key}' not found in PromptManager, using default")
            template_key = "eval_comprehensive"
            template_obj = self.prompt_manager.get_template(template_key)
            template = template_obj.template
        
        # Extract proposal content from CSV format
        proposal_id = proposal.get('proposal_id', 'unknown')
        proposal_title = proposal.get('title', 'N/A')
        proposal_abstract = proposal.get('abstract', 'N/A')
        proposal_full = proposal.get('full_draft', '')
        
        # Format the prompt (only include role_description if provided)
        format_kwargs = {
            'research_call': self.research_call,
            'proposal_id': proposal_id,
            'proposal_title': proposal_title,
            'proposal_abstract': proposal_abstract,
            'proposal_full': proposal_full
        }
        
        # Only add role_description if provided and template requires it
        if role_description and 'role_description' in template:
            format_kwargs['role_description'] = role_description
        
        prompt = template.format(**format_kwargs)
        
        return prompt
    
    def evaluate_proposal(self,
                         proposal: Dict[str, Any],
                         evaluation_template: str = "comprehensive",
                         role_description: str = None,
                         evaluator_model: str = "gemini-2.5-pro") -> Dict[str, Any]:
        """Evaluate a single proposal"""
        
        proposal_id = proposal.get('proposal_id', 'unknown')
        proposal_title = proposal.get('title', 'N/A')
        proposal_who = proposal.get('who', 'unknown')
        proposal_role = proposal.get('role', 'unknown')
        proposal_model = proposal.get('model', 'unknown')
        
        logger.info(f"Evaluating proposal: '{proposal_title}' (ID: {proposal_id}, Source: {proposal_who}/{proposal_role})")
        
        # Create evaluation prompt
        prompt = self.create_evaluation_prompt(
            proposal=proposal,
            evaluation_template=evaluation_template,
            role_description=role_description
        )
        
        # Get evaluation from AI model
        try:
            evaluation_response = self.ai_interface.generate_content(
                prompt=prompt,
                model_name=evaluator_model
            )
            
            # Parse JSON response
            try:
                evaluation_data = json.loads(evaluation_response)
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse JSON response: {je}")
                logger.warning(f"Raw response: {evaluation_response[:200]}...")
                evaluation_data = {
                    "error": "Failed to parse JSON",
                    "raw_response": evaluation_response
                }
        except Exception as e:
            logger.error(f"Error generating evaluation: {e}")
            evaluation_data = {
                "error": str(e)
            }
        
        # Create evaluation result
        evaluation_id = f"{proposal_id}_{evaluation_template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            'evaluation_id': evaluation_id,
            'proposal_id': proposal_id,
            'proposal_title': proposal_title,
            'proposal_who': proposal_who,
            'proposal_role': proposal_role,
            'proposal_model': proposal_model,
            'evaluator_model': evaluator_model,
            'evaluation_template': evaluation_template,
            'role_description': role_description,
            'timestamp': datetime.now().isoformat(),
            'evaluation_response': evaluation_data
        }
        
        return result
    
    def evaluate_all(self,
                    proposals: List[Dict[str, Any]],
                    evaluation_templates: List[str] = None,
                    role_descriptions: List[str] = None,
                    evaluator_model: str = "gemini-2.5-pro") -> List[Dict[str, Any]]:
        """Evaluate all proposals individually"""
        
        # Set defaults
        if evaluation_templates is None:
            evaluation_templates = ["comprehensive"]
        
        if role_descriptions is None:
            role_descriptions = [None]  # No role description by default
        
        logger.info(f"Evaluating {len(proposals)} proposals individually")
        
        # Generate all combinations
        all_evaluations = []
        total_evaluations = len(proposals) * len(evaluation_templates) * len(role_descriptions)
        current = 0
        
        for prop, eval_template, role in itertools.product(
            proposals, evaluation_templates, role_descriptions
        ):
            current += 1
            logger.info(f"Processing evaluation {current}/{total_evaluations}")
            
            result = self.evaluate_proposal(
                proposal=prop,
                evaluation_template=eval_template,
                role_description=role,
                evaluator_model=evaluator_model
            )
            
            all_evaluations.append(result)
        
        return all_evaluations
    
    def save_evaluations(self, 
                        evaluations: List[Dict[str, Any]],
                        source_type: str,
                        output_filename: str = None,
                        evaluation_templates: List[str] = None,
                        proposal_role: str = None,
                        proposal_ai_model: str = None):
        """Save evaluation results to a JSON file in organized subfolders"""
        
        # Create subfolder name based on evaluation parameters
        subfolder_parts = ["single", source_type]
        
        # Add template name(s)
        if evaluation_templates and len(evaluation_templates) > 0:
            if len(evaluation_templates) == 1:
                subfolder_parts.append(evaluation_templates[0])
            else:
                subfolder_parts.append("multi")
        
        # Add AI model if specified (for AI proposals)
        if proposal_role:
            subfolder_parts.append(proposal_role)
        if proposal_ai_model:
            model_short = proposal_ai_model.split('-')[0]
            subfolder_parts.append(f"genby_{model_short}")
        
        subfolder_name = "_".join(subfolder_parts)
        output_dir = self.evaluations_dir / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include evaluation template(s) in filename
            if evaluation_templates and len(evaluation_templates) > 0:
                # For single template, use its name
                if len(evaluation_templates) == 1:
                    template_str = evaluation_templates[0]
                # For multiple templates, use "multi" or list first few
                else:
                    template_str = "multi"
                output_filename = f"evaluations_single_{source_type}_{template_str}_{timestamp}.json"
            else:
                output_filename = f"evaluations_single_{source_type}_{timestamp}.json"
        
        output_path = output_dir / output_filename
        
        # Create evaluations dict with enhanced metadata
        evaluations_dict = {
            "metadata": {
                "evaluation_type": f"single_{source_type}",
                "evaluation_templates": evaluation_templates if evaluation_templates else [],
                "proposal_role": proposal_role,
                "proposal_ai_model": proposal_ai_model,
                "total_evaluations": len(evaluations),
                "generation_timestamp": datetime.now().isoformat(),
            },
            "evaluations": evaluations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluations_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(evaluations)} evaluations to {output_path}")
        
    def generate_summary_report(self, evaluations: List[Dict[str, Any]]) -> str:
        """Generate a summary report from evaluations"""
        
        report = ["# Proposal Evaluation Summary Report\n"]
        report.append(f"Total Evaluations: {len(evaluations)}\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group by evaluation template
        by_template = {}
        for eval_result in evaluations:
            template = eval_result.get('evaluation_template', 'unknown')
            if template not in by_template:
                by_template[template] = []
            by_template[template].append(eval_result)
        
        for template, evals in by_template.items():
            report.append(f"## Evaluation Template: {template}\n")
            report.append(f"Number of evaluations: {len(evals)}\n\n")
            
            for eval_result in evals[:5]:  # Show first 5 as examples
                report.append(f"### {eval_result.get('evaluation_id', 'N/A')}\n")
                report.append(f"- Proposal: {eval_result.get('proposal_title', 'N/A')} ({eval_result.get('proposal_who', 'unknown')}/{eval_result.get('proposal_role', 'unknown')})\n")
                report.append(f"- Role: {eval_result.get('role_description', 'N/A')}\n")
                report.append(f"- Evaluator: {eval_result.get('evaluator_model', 'N/A')}\n\n")
        
        return "".join(report)
    
    def list_available_evaluation_templates(self) -> List[str]:
        """List available templates for single proposal evaluation"""
        return [
            "comprehensive",
            "strengths_weaknesses",
            "innovation_assessment",
            "alignment_with_call",
            "human_criteria"
        ]
    
    def list_available_role_descriptions(self) -> List[str]:
        """List available role descriptions for evaluators"""
        return [
            "expert scientific reviewer",
            "program officer evaluating grant applications",
            "interdisciplinary scientist",
            "data science expert",
            "molecular biologist",
            "computational biologist",
            "methodological expert in synthesis research",
            "early career researcher",
            "senior principal investigator"
        ]


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate individual research proposals"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="all_proposals_combined.csv",
        help="Path to combined proposals CSV file (default: all_proposals_combined.csv)"
    )
    parser.add_argument(
        "--proposals-json",
        type=str,
        default=None,
        help="Path to combined proposals JSON file (alternative to CSV, e.g., all_proposals_combined_no_role.json)"
    )
    parser.add_argument(
        "--selection-json",
        type=str,
        default=None,
        help="Path to JSON file with selected proposal IDs (e.g., top_22_diverse_ai_proposals.json). Uses 'selected_proposals' field."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["human", "ai", "both"],
        default="both",
        help="Which proposals to evaluate (default: both)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Filter AI proposals by template/role (e.g., 'generate_ideas_no_role', 'single', 'group', 'group_int'). Default: None (all templates)"
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help="Specific AI model name to filter proposals (optional)"
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gemini-2.5-pro",
        help="AI model to use for evaluation"
    )
    parser.add_argument(
        "--eval-templates",
        nargs="+",
        default=["comprehensive"],
        help="Evaluation templates to use (default: comprehensive)"
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=None,
        help="Role descriptions for evaluators. Default: None (no role description)"
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=None,
        help="Maximum number of proposals to evaluate"
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available evaluation templates and exit"
    )
    parser.add_argument(
        "--list-roles",
        action="store_true",
        help="List available role descriptions and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for evaluations"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator with CSV or JSON file
    if args.proposals_json:
        evaluator = ProposalEvaluator(proposals_json=args.proposals_json)
    else:
        evaluator = ProposalEvaluator(proposals_csv=args.csv)
    
    # Handle list commands
    if args.list_templates:
        print("Available Evaluation Templates:")
        for template in evaluator.list_available_evaluation_templates():
            print(f"  - {template}")
        return
    
    if args.list_roles:
        print("Available Role Descriptions:")
        for role in evaluator.list_available_role_descriptions():
            print(f"  - {role}")
        return
    
    # Load selected proposals if selection JSON is provided
    selected_proposal_ids = None
    if args.selection_json:
        selected_proposal_ids = evaluator.load_proposal_ids_from_selection_json(args.selection_json)
        if not selected_proposal_ids:
            logger.error("No proposal IDs loaded from selection JSON. Exiting.")
            return
        logger.info(f"Will evaluate {len(selected_proposal_ids)} selected proposals")
    
    logger.info(f"Starting proposal evaluation")
    logger.info(f"Evaluation templates: {args.eval_templates}")
    logger.info(f"Role descriptions: {args.roles}")
    
    # Determine source type for output naming
    source_type = args.source
    
    # Load proposals - either from selection JSON or with filters
    if selected_proposal_ids:
        proposals = evaluator.load_proposals_by_ids(selected_proposal_ids)
        if args.max_proposals:
            proposals = proposals[:args.max_proposals]
        
        # Auto-detect source type based on loaded proposals
        who_types = set(p.get('who') for p in proposals)
        if len(who_types) == 1:
            if 'human' in who_types:
                source_type = "human"
            else:
                source_type = "ai"
        else:
            source_type = "both"
        logger.info(f"Auto-detected source type: {source_type}")
    else:
        # Load proposals from CSV with filters
        who_filter = None if args.source == "both" else args.source
        role_filter = args.template
        model_filter = args.ai_model
        
        proposals = evaluator.load_proposals(
            who=who_filter,
            role=role_filter,
            model=model_filter,
            max_proposals=args.max_proposals
        )
    
    # Evaluate all proposals
    all_evaluations = evaluator.evaluate_all(
        proposals=proposals,
        evaluation_templates=args.eval_templates,
        role_descriptions=args.roles,
        evaluator_model=args.evaluator_model
    )
    
    # Save evaluations
    evaluator.save_evaluations(
        evaluations=all_evaluations,
        source_type=source_type,
        output_filename=args.output,
        evaluation_templates=args.eval_templates,
        proposal_role=args.template,
        proposal_ai_model=args.ai_model
    )
    
    # Generate and save summary report (in the same subfolder as evaluations)
    summary = evaluator.generate_summary_report(all_evaluations)
    
    # Determine the correct subfolder (same logic as save_evaluations)
    subfolder_parts = ["single", source_type]
    
    if args.eval_templates and len(args.eval_templates) > 0:
        if len(args.eval_templates) == 1:
            subfolder_parts.append(args.eval_templates[0])
        else:
            subfolder_parts.append("multi")
    
    if args.template:
        subfolder_parts.append(args.template)
    if args.ai_model:
        model_short = args.ai_model.split('-')[0]
        subfolder_parts.append(f"genby_{model_short}")
    
    subfolder_name = "_".join(subfolder_parts)
    summary_dir = evaluator.evaluations_dir / subfolder_name
    summary_path = summary_dir / f"summary_single_{source_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Summary report saved to {summary_path}")
    logger.info(f"Completed {len(all_evaluations)} evaluations")


if __name__ == "__main__":
    main()
