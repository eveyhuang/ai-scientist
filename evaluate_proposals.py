#!/usr/bin/env python3
"""
Evaluate and compare research proposals from different sources
Compares human proposals with AI-generated proposals using various evaluation prompts
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
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


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    evaluation_id: str
    human_proposal_id: str
    ai_proposal_id: str
    evaluator_model: str
    evaluation_template: str
    role_description: str
    timestamp: str
    human_proposal_title: str
    ai_proposal_title: str
    evaluation_response: str
    scores: Optional[Dict[str, Any]] = None
    comparison_summary: Optional[str] = None


class ProposalEvaluator:
    """Evaluate and compare research proposals"""
    
    def __init__(self, config_path: str = "config.env", proposals_csv: str = "all_proposals_combined.csv"):
        """Initialize the proposal evaluator"""
        self.ai_interface = AIModelsInterface(config_path)
        self.prompt_manager = PromptManager()
        self.evaluations_dir = Path("evaluations")
        self.evaluations_dir.mkdir(exist_ok=True)
        
        # Load proposals from CSV
        self.proposals_csv = proposals_csv
        self.proposals_df = None
        self._load_proposals_from_csv()
        
        # Load research call
        self.research_call = self._load_research_call()
        
    def _load_research_call(self) -> str:
        """Load the research call from the human proposals file"""
        try:
            with open("human-proposals-y1.json", 'r', encoding='utf-8') as f:
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
    
    def create_single_evaluation_prompt(self,
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
    
    def create_pairwise_evaluation_prompt(self, 
                                         proposal_1: Dict[str, Any],
                                         proposal_2: Dict[str, Any],
                                         evaluation_template: str) -> str:
        """Create an evaluation prompt comparing two proposals"""
        
        # Map user-friendly template names to template keys in PromptManager
        template_name_mapping = {
            "proposal_overlap": "eval_proposal_overlap"
        }
        
        # Get the template key
        template_key = template_name_mapping.get(evaluation_template, "eval_proposal_overlap")
        
        # Get the template from PromptManager
        try:
            template_obj = self.prompt_manager.get_template(template_key)
            template = template_obj.template
        except ValueError:
            logger.warning(f"Template '{template_key}' not found in PromptManager, using default")
            template_key = "eval_proposal_overlap"
            template_obj = self.prompt_manager.get_template(template_key)
            template = template_obj.template
        
        # Extract proposal 1 content from CSV format
        proposal_1_id = proposal_1.get('proposal_id', 'unknown')
        proposal_1_title = proposal_1.get('title', 'N/A')
        proposal_1_abstract = proposal_1.get('abstract', 'N/A')
        proposal_1_full = proposal_1.get('full_draft', '')
        
        # Extract proposal 2 content from CSV format
        proposal_2_id = proposal_2.get('proposal_id', 'unknown')
        proposal_2_title = proposal_2.get('title', 'N/A')
        proposal_2_abstract = proposal_2.get('abstract', 'N/A')
        proposal_2_full = proposal_2.get('full_draft', '')
        
        # Format the prompt
        prompt = template.format(
            research_call=self.research_call,
            proposal_1_id=proposal_1_id,
            proposal_1_title=proposal_1_title,
            proposal_1_abstract=proposal_1_abstract,
            proposal_1_full=proposal_1_full,
            proposal_2_id=proposal_2_id,
            proposal_2_title=proposal_2_title,
            proposal_2_abstract=proposal_2_abstract,
            proposal_2_full=proposal_2_full
        )
        
        return prompt
    
    def evaluate_single_proposal(self,
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
        
        logger.info(f"Evaluating single proposal: '{proposal_title}' (ID: {proposal_id}, Source: {proposal_who}/{proposal_role})")
        
        # Create evaluation prompt
        prompt = self.create_single_evaluation_prompt(
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
    
    def evaluate_proposal_pair(self,
                              proposal_1: Dict[str, Any],
                              proposal_2: Dict[str, Any],
                              evaluation_template: str = "proposal_overlap",
                              evaluator_model: str = "gemini-2.5-pro") -> Dict[str, Any]:
        """Evaluate a pair of proposals (for overlap/comparison)"""
        
        proposal_1_id = proposal_1.get('proposal_id', 'unknown')
        proposal_2_id = proposal_2.get('proposal_id', 'unknown')
        proposal_1_title = proposal_1.get('title', 'N/A')
        proposal_2_title = proposal_2.get('title', 'N/A')
        proposal_1_who = proposal_1.get('who', 'unknown')
        proposal_2_who = proposal_2.get('who', 'unknown')
        proposal_1_role = proposal_1.get('role', 'unknown')
        proposal_2_role = proposal_2.get('role', 'unknown')
        
        logger.info(f"Comparing proposals: '{proposal_1_title}' ({proposal_1_who}/{proposal_1_role}) vs '{proposal_2_title}' ({proposal_2_who}/{proposal_2_role})")
        
        # Create evaluation prompt
        prompt = self.create_pairwise_evaluation_prompt(
            proposal_1=proposal_1,
            proposal_2=proposal_2,
            evaluation_template=evaluation_template
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
        evaluation_id = f"{proposal_1_id}_vs_{proposal_2_id}_{evaluation_template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            'evaluation_id': evaluation_id,
            'proposal_1_id': proposal_1_id,
            'proposal_2_id': proposal_2_id,
            'proposal_1_title': proposal_1_title,
            'proposal_2_title': proposal_2_title,
            'proposal_1_who': proposal_1_who,
            'proposal_2_who': proposal_2_who,
            'proposal_1_role': proposal_1_role,
            'proposal_2_role': proposal_2_role,
            'evaluator_model': evaluator_model,
            'evaluation_template': evaluation_template,
            'timestamp': datetime.now().isoformat(),
            'evaluation_response': evaluation_data
        }
        
        return result
    
    def evaluate_all_single(self,
                           proposals: List[Dict[str, Any]],
                           evaluation_templates: List[str] = None,
                           role_descriptions: List[str] = None,
                           evaluator_model: str = "gemini-2.5-pro") -> List[Dict[str, Any]]:
        """Evaluate all proposals individually"""
        
        # Set defaults
        if evaluation_templates is None:
            evaluation_templates = ["comprehensive"]
        
        if role_descriptions is None:
            role_descriptions = ["expert scientific reviewer"]
        
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
            
            result = self.evaluate_single_proposal(
                proposal=prop,
                evaluation_template=eval_template,
                role_description=role,
                evaluator_model=evaluator_model
            )
            
            all_evaluations.append(result)
        
        return all_evaluations
    
    def evaluate_all_pairs(self,
                          proposals_1: List[Dict[str, Any]],
                          proposals_2: List[Dict[str, Any]],
                          evaluation_template: str = "proposal_overlap",
                          evaluator_model: str = "gemini-2.5-pro",
                          checkpoint_interval: int = 10,
                          resume_from_checkpoint: bool = True,
                          comparison_type: str = "human-ai",
                          proposal_role: str = None,
                          proposal_ai_model: str = None) -> List[Dict[str, Any]]:
        """
        Evaluate all pairs of proposals (e.g., for overlap comparison)
        
        Args:
            proposals_1: First set of proposals
            proposals_2: Second set of proposals
            evaluation_template: Template to use for evaluation
            evaluator_model: Model to use for evaluation
            checkpoint_interval: Save progress every N evaluations (default: 10)
            resume_from_checkpoint: Resume from last checkpoint if available (default: True)
            comparison_type: Type of comparison (human-human, ai-ai, human-ai)
            proposal_role: Role filter used for proposals (for checkpoint naming)
            proposal_ai_model: AI model filter used for proposals (for checkpoint naming)
        
        Returns:
            List of all evaluation results
        """
        
        # Generate all combinations with indices for tracking
        all_pairs = list(itertools.product(enumerate(proposals_1), enumerate(proposals_2)))
        
        # Count non-self comparisons for accurate total
        total_evaluations = sum(
            1 for (_, p1), (_, p2) in all_pairs 
            if p1.get('proposal_id') != p2.get('proposal_id')
        )
        
        logger.info(f"Comparing {len(proposals_1)} x {len(proposals_2)} proposal pairs")
        logger.info(f"Comparison type: {comparison_type}")
        logger.info(f"Evaluator model: {evaluator_model}")
        logger.info(f"Total comparisons (excluding self-comparisons): {total_evaluations}")
        
        # Setup checkpoint file with comparison type, role, proposal AI model, and evaluator model
        # Extract short model name (e.g., "gemini" from "gemini-2.5-pro", "gpt" from "gpt-4")
        evaluator_short = evaluator_model.split('-')[0]
        
        # Build checkpoint filename with all distinguishing parameters
        checkpoint_parts = [f"checkpoint_{comparison_type}"]
        if proposal_role:
            checkpoint_parts.append(proposal_role)
        if proposal_ai_model:
            proposal_model_short = proposal_ai_model.split('-')[0]
            checkpoint_parts.append(f"genby_{proposal_model_short}")
        checkpoint_parts.append(f"eval_{evaluator_short}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = "_".join(checkpoint_parts) + f"_{timestamp}.json"
        checkpoint_file = self.evaluations_dir / checkpoint_name
        
        # Try to resume from checkpoint (only matching comparison type, role, and models)
        all_evaluations = []
        completed_pairs = set()
        start_index = 0
        
        if resume_from_checkpoint:
            # Build glob pattern for finding matching checkpoints
            glob_parts = [f"checkpoint_{comparison_type}"]
            if proposal_role:
                glob_parts.append(proposal_role)
            if proposal_ai_model:
                proposal_model_short = proposal_ai_model.split('-')[0]
                glob_parts.append(f"genby_{proposal_model_short}")
            glob_parts.append(f"eval_{evaluator_short}")
            glob_pattern = "_".join(glob_parts) + "_*.json"
            
            checkpoint_files = sorted(self.evaluations_dir.glob(glob_pattern))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                try:
                    with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        all_evaluations = checkpoint_data.get('evaluations', [])
                        completed_pairs = set(checkpoint_data.get('completed_pairs', []))
                        start_index = len(all_evaluations)
                        checkpoint_file = latest_checkpoint  # Continue using same file
                        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                        logger.info(f"Already completed: {len(all_evaluations)}/{total_evaluations} evaluations")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
        
        # Track statistics
        errors_count = 0
        successful_count = len(all_evaluations)
        
        try:
            for idx, ((i1, prop_1), (i2, prop_2)) in enumerate(all_pairs):
                # Skip self-comparisons (same proposal)
                if prop_1.get('proposal_id') == prop_2.get('proposal_id'):
                    logger.debug(f"Skipping self-comparison: {prop_1.get('proposal_id')}")
                    continue
                
                # Skip if already completed
                pair_key = f"{i1}_{i2}"
                if pair_key in completed_pairs:
                    continue
                
                current = idx + 1
                logger.info(f"Processing comparison {current}/{total_evaluations}")
                
                try:
                    result = self.evaluate_proposal_pair(
                        proposal_1=prop_1,
                        proposal_2=prop_2,
                        evaluation_template=evaluation_template,
                        evaluator_model=evaluator_model
                    )
                    
                    all_evaluations.append(result)
                    completed_pairs.add(pair_key)
                    successful_count += 1
                    
                    # Check if evaluation had an error in the response
                    if isinstance(result.get('evaluation_response'), dict) and 'error' in result['evaluation_response']:
                        errors_count += 1
                        logger.warning(f"Evaluation {current} completed with error: {result['evaluation_response'].get('error')}")
                    
                except Exception as e:
                    # Log error but continue processing
                    errors_count += 1
                    logger.error(f"Failed to evaluate pair {current}/{total_evaluations}: {e}")
                    
                    # Create error result
                    error_result = {
                        'evaluation_id': f"error_{i1}_{i2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'proposal_1_id': prop_1.get('proposal_id', 'unknown'),
                        'proposal_2_id': prop_2.get('proposal_id', 'unknown'),
                        'proposal_1_title': prop_1.get('title', 'N/A'),
                        'proposal_2_title': prop_2.get('title', 'N/A'),
                        'evaluator_model': evaluator_model,
                        'evaluation_template': evaluation_template,
                        'timestamp': datetime.now().isoformat(),
                        'evaluation_response': {
                            'error': f"Exception during evaluation: {str(e)}"
                        }
                    }
                    all_evaluations.append(error_result)
                    completed_pairs.add(pair_key)
                
                # Save checkpoint at regular intervals
                if current % checkpoint_interval == 0 or current == total_evaluations:
                    self._save_checkpoint(checkpoint_file, all_evaluations, list(completed_pairs), comparison_type, evaluator_model)
                    logger.info(f"Checkpoint saved: {successful_count} successful, {errors_count} errors")
        
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user. Saving checkpoint...")
            self._save_checkpoint(checkpoint_file, all_evaluations, list(completed_pairs), comparison_type, evaluator_model)
            logger.info(f"Progress saved. Completed {len(all_evaluations)}/{total_evaluations} evaluations")
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during batch evaluation: {e}")
            self._save_checkpoint(checkpoint_file, all_evaluations, list(completed_pairs), comparison_type, evaluator_model)
            logger.info(f"Emergency checkpoint saved. Completed {len(all_evaluations)}/{total_evaluations} evaluations")
            raise
        
        # Final summary
        logger.info(f"\n=== Evaluation Complete ===")
        logger.info(f"Total evaluations: {len(all_evaluations)}")
        logger.info(f"Successful: {successful_count}")
        logger.info(f"Errors: {errors_count}")
        
        return all_evaluations
    
    def _save_checkpoint(self, checkpoint_file: Path, evaluations: List[Dict[str, Any]], completed_pairs: List[str], comparison_type: str = None, evaluator_model: str = None):
        """Save checkpoint to file"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'comparison_type': comparison_type,
            'evaluator_model': evaluator_model,
            'total_evaluations': len(evaluations),
            'completed_pairs': completed_pairs,
            'evaluations': evaluations
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    
    def save_evaluations(self, 
                        evaluations: List[Dict[str, Any]],
                        evaluation_type: str,
                        output_filename: str = None,
                        evaluation_templates: List[str] = None):
        """Save evaluation results to a JSON file"""
        
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
                output_filename = f"evaluations_{evaluation_type}_{template_str}_{timestamp}.json"
            else:
                output_filename = f"evaluations_{evaluation_type}_{timestamp}.json"
        
        output_path = self.evaluations_dir / output_filename
        
        # Create evaluations dict
        evaluations_dict = {
            "metadata": {
                "evaluation_type": evaluation_type,
                "evaluation_templates": evaluation_templates if evaluation_templates else [],
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
                
                # Handle single proposal vs pair comparison
                if 'proposal_1_id' in eval_result:
                    report.append(f"- Proposal 1: {eval_result.get('proposal_1_title', 'N/A')} ({eval_result.get('proposal_1_who', 'unknown')}/{eval_result.get('proposal_1_role', 'unknown')})\n")
                    report.append(f"- Proposal 2: {eval_result.get('proposal_2_title', 'N/A')} ({eval_result.get('proposal_2_who', 'unknown')}/{eval_result.get('proposal_2_role', 'unknown')})\n")
                else:
                    report.append(f"- Proposal: {eval_result.get('proposal_title', 'N/A')} ({eval_result.get('proposal_who', 'unknown')}/{eval_result.get('proposal_role', 'unknown')})\n")
                    report.append(f"- Role: {eval_result.get('role_description', 'N/A')}\n")
                
                report.append(f"- Evaluator: {eval_result.get('evaluator_model', 'N/A')}\n\n")
        
        return "".join(report)
    
    def list_available_single_evaluation_templates(self) -> List[str]:
        """List available templates for single proposal evaluation"""
        return [
            "comprehensive",
            "strengths_weaknesses",
            "innovation_assessment",
            "alignment_with_call",
            "human_criteria"
        ]
    
    def list_available_pairwise_evaluation_templates(self) -> List[str]:
        """List available templates for pairwise proposal comparison"""
        return [
            "proposal_overlap"
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
        description="Evaluate and compare research proposals"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "pairwise"],
        required=True,
        help="Evaluation mode: 'single' for individual proposals, 'pairwise' for comparisons"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="all_proposals_combined.csv",
        help="Path to combined proposals CSV file (default: all_proposals_combined.csv)"
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
        default=None,
        help="Evaluation templates to use (defaults depend on mode)"
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=None,
        help="Role descriptions for evaluators (single mode only). Default: None (no role description)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["human", "ai", "both"],
        default="both",
        help="Which proposals to evaluate (single mode only)"
    )
    parser.add_argument(
        "--compare-type",
        type=str,
        choices=["human-human", "ai-ai", "human-ai"],
        default="human-ai",
        help="Type of comparison (pairwise mode only)"
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save progress every N evaluations (default: 10)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator with CSV file
    evaluator = ProposalEvaluator(proposals_csv=args.csv)
    
    # Handle list commands
    if args.list_templates:
        print("Available Single Evaluation Templates:")
        for template in evaluator.list_available_single_evaluation_templates():
            print(f"  - {template}")
        print("\nAvailable Pairwise Evaluation Templates:")
        for template in evaluator.list_available_pairwise_evaluation_templates():
            print(f"  - {template}")
        return
    
    if args.list_roles:
        print("Available Role Descriptions:")
        for role in evaluator.list_available_role_descriptions():
            print(f"  - {role}")
        return
    
    # Set default eval templates based on mode
    if args.eval_templates is None:
        if args.mode == "single":
            args.eval_templates = ["comprehensive"]
        else:
            args.eval_templates = ["proposal_overlap"]
    
    # Run evaluations based on mode
    if args.mode == "single":
        logger.info(f"Starting single proposal evaluation")
        logger.info(f"Evaluation templates: {args.eval_templates}")
        logger.info(f"Role descriptions: {args.roles}")
        
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
        all_evaluations = evaluator.evaluate_all_single(
            proposals=proposals,
            evaluation_templates=args.eval_templates,
            role_descriptions=args.roles,
            evaluator_model=args.evaluator_model
        )
        
        evaluation_type = f"single_{args.source}"
        
    else:  # pairwise mode
        logger.info(f"Starting pairwise proposal comparison")
        logger.info(f"Comparison type: {args.compare_type}")
        logger.info(f"Evaluation template: {args.eval_templates[0]}")
        
        # Load proposals based on comparison type
        if args.compare_type == "human-human":
            proposals_1 = evaluator.load_proposals(who="human", max_proposals=args.max_proposals)
            proposals_2 = evaluator.load_proposals(who="human", max_proposals=args.max_proposals)
        elif args.compare_type == "ai-ai":
            proposals_1 = evaluator.load_proposals(
                who="ai",
                role=args.template,
                model=args.ai_model,
                max_proposals=args.max_proposals
            )
            proposals_2 = evaluator.load_proposals(
                who="ai",
                role=args.template,
                model=args.ai_model,
                max_proposals=args.max_proposals
            )
        else:  # human-ai
            proposals_1 = evaluator.load_proposals(who="human", max_proposals=args.max_proposals)
            proposals_2 = evaluator.load_proposals(
                who="ai",
                role=args.template,
                model=args.ai_model,
                max_proposals=args.max_proposals
            )
        
        all_evaluations = evaluator.evaluate_all_pairs(
            proposals_1=proposals_1,
            proposals_2=proposals_2,
            evaluation_template=args.eval_templates[0],
            evaluator_model=args.evaluator_model,
            checkpoint_interval=args.checkpoint_interval,
            resume_from_checkpoint=not args.no_resume,
            comparison_type=args.compare_type,
            proposal_role=args.template,
            proposal_ai_model=args.ai_model
        )
        
        evaluation_type = f"pairwise_{args.compare_type}"
    
    # Save evaluations
    evaluator.save_evaluations(
        evaluations=all_evaluations,
        evaluation_type=evaluation_type,
        output_filename=args.output,
        evaluation_templates=args.eval_templates
    )
    
    # Generate and save summary report
    summary = evaluator.generate_summary_report(all_evaluations)
    summary_path = evaluator.evaluations_dir / f"summary_{evaluation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Summary report saved to {summary_path}")
    logger.info(f"Completed {len(all_evaluations)} evaluations")


if __name__ == "__main__":
    main()

