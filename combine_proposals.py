#!/usr/bin/env python3
"""
Combine all human and AI-generated proposals into a single CSV file.
Creates a unified dataset with proposal_id, role, who (human/ai), model, title, abstract, authors, and full_draft.

Supports proposals from:
- Human proposals (human-proposals-y1.json)
- AI proposals from generate_ideas_no_role template
- AI proposals from generate_diverse_ideas template
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProposalCombiner:
    """Combine human and AI proposals into a unified dataframe"""
    
    def __init__(self):
        self.base_dir = Path("generated_ideas")
        self.proposals = []
        
        # Map template names to role types
        # For generate_ideas_no_role and generate_diverse_ideas, use the template name as the role
        self.role_mapping = {
            'generate_ideas_no_role': 'generate_ideas_no_role',
            'generate_diverse_ideas': 'generate_diverse_ideas'
        }
    
    def load_human_proposals(self) -> List[Dict[str, Any]]:
        """Load human proposals from human-proposals-y1.json"""
        logger.info("Loading human proposals...")
        
        try:
            with open("human-proposals-y1.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                human_proposals = data.get('proposals', [])
            
            logger.info(f"Loaded {len(human_proposals)} human proposals")
            
            # Process human proposals
            processed = []
            for i, proposal in enumerate(human_proposals, 1):
                processed_proposal = {
                    'proposal_id': f"human_{proposal.get('proposal_id', i)}",
                    'role': 'human',
                    'who': 'human',
                    'model': 'human',
                    'title': proposal.get('proposal_title', ''),
                    'abstract': proposal.get('abstract', ''),
                    'authors': '; '.join(proposal.get('authors', [])),
                    'full_draft': proposal.get('full_draft', ''),
                    'proposal_status': proposal.get('proposal_status', ''),
                    'ranking': proposal.get('ranking', None)
                }
                processed.append(processed_proposal)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error loading human proposals: {e}")
            return []
    
    def load_ai_proposals_from_template(self, template_name: str) -> List[Dict[str, Any]]:
        """Load AI proposals from a specific template folder"""
        proposals_dir = self.base_dir / template_name / "proposals"
        
        if not proposals_dir.exists():
            logger.warning(f"Proposals directory not found: {proposals_dir}")
            return []
        
        # Find all proposal files
        proposal_files = list(proposals_dir.glob("proposals_*.json"))
        
        if not proposal_files:
            logger.warning(f"No proposal files found in {proposals_dir}")
            return []
        
        logger.info(f"Found {len(proposal_files)} proposal files for template '{template_name}'")
        
        all_proposals = []
        # Track used IDs to prevent duplicates
        used_ids = set()
        
        for proposal_file in proposal_files:
            try:
                with open(proposal_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    model_name = data.get('session_id', '').split('_')[0] if data.get('session_id') else 'unknown'
                    proposals = data.get('proposals', [])
                    
                    # Process each proposal
                    for proposal in proposals:
                        # Merge proposal sections into full_draft
                        full_draft = self._merge_proposal_sections(proposal.get('proposal', {}))
                        
                        # Get role from template name
                        role = self.role_mapping.get(template_name, template_name)
                        
                        # Create processed proposal with simplified ID
                        # Extract just the numeric part from idea_id if available
                        idea_id = proposal.get('idea_id', 'unknown')
                        if isinstance(idea_id, str) and '_' in idea_id:
                            # If idea_id is like "gemini-2.5-pro_single_scientist_01", extract "01"
                            idea_num = idea_id.split('_')[-1]
                        else:
                            idea_num = str(idea_id)
                        
                        # Create simplified ID: ai_{role}_{model_short}_{num}
                        model_short = proposal.get('model_name', model_name).split('-')[0]  # e.g., "gemini" or "gpt"
                        simplified_id = f"ai_{role}_{model_short}_{idea_num}"
                        
                        # If ID already exists, append a counter to make it unique
                        original_id = simplified_id
                        counter = 1
                        while simplified_id in used_ids:
                            counter += 1
                            simplified_id = f"{original_id}_{counter}"
                        
                        used_ids.add(simplified_id)
                        
                        # Warn if full_draft is empty
                        if not full_draft or not full_draft.strip():
                            logger.warning(f"Proposal {simplified_id} has empty full_draft")
                        
                        processed_proposal = {
                            'proposal_id': simplified_id,
                            'role': role,
                            'who': 'ai',
                            'model': proposal.get('model_name', model_name),
                            'title': proposal.get('original_title', proposal.get('proposal', {}).get('title', '')),
                            'abstract': proposal.get('original_abstract', proposal.get('proposal', {}).get('abstract', '')),
                            'authors': '',  # AI proposals don't have authors
                            'full_draft': full_draft,
                            'proposal_status': None,
                            'ranking': None
                        }
                        all_proposals.append(processed_proposal)
                        
            except Exception as e:
                logger.error(f"Error loading proposals from {proposal_file}: {e}")
        
        logger.info(f"Loaded {len(all_proposals)} AI proposals from template '{template_name}'")
        return all_proposals
    
    def _merge_proposal_sections(self, proposal_dict: Dict[str, Any]) -> str:
        """Merge all proposal sections into a single full_draft text"""
        if not proposal_dict:
            return ''
        
        # Define section order
        section_order = [
            'title',
            'abstract',
            'background_and_significance',
            'research_questions_and_hypotheses',
            'methods_and_approach',
            'expected_outcomes_and_impact',
            'budget_and_resources'
        ]
        
        sections = []
        for section_key in section_order:
            if section_key in proposal_dict and isinstance(proposal_dict[section_key], str):
                # Remove line breaks from section content
                section_content = proposal_dict[section_key].replace('\n', ' ')
                # Format section header (skip title and abstract as they're in separate columns)
                if section_key not in ['title', 'abstract']:
                    section_header = section_key.replace('_', ' ').title()
                    sections.append(f"{section_header}\n\n{section_content}")
        
        return '\n\n'.join(sections)
    
    def load_all_ai_proposals(self) -> List[Dict[str, Any]]:
        """Load AI proposals from specified template folders only"""
        logger.info("Loading AI proposals from specified templates...")
        
        all_ai_proposals = []
        
        # Only load templates explicitly defined in role_mapping
        for template_name in self.role_mapping.keys():
            template_proposals = self.load_ai_proposals_from_template(template_name)
            all_ai_proposals.extend(template_proposals)
        
        logger.info(f"Loaded total of {len(all_ai_proposals)} AI proposals")
        return all_ai_proposals
    
    def combine_all_proposals(self) -> List[Dict[str, Any]]:
        """Combine all human and AI proposals into a single list"""
        logger.info("Combining all proposals...")
        
        # Load human proposals
        human_proposals = self.load_human_proposals()
        
        # Load AI proposals
        ai_proposals = self.load_all_ai_proposals()
        
        # Combine all proposals
        all_proposals = human_proposals + ai_proposals
        
        logger.info(f"Total proposals: {len(all_proposals)} ({len(human_proposals)} human + {len(ai_proposals)} AI)")
        
        # Add some statistics
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Total proposals: {len(all_proposals)}")
        
        who_counts = Counter(p['who'] for p in all_proposals)
        logger.info(f"\nBy source (who):")
        for key, count in who_counts.items():
            logger.info(f"  {key}: {count}")
        
        role_counts = Counter(p['role'] for p in all_proposals)
        logger.info(f"\nBy role:")
        for key, count in role_counts.items():
            logger.info(f"  {key}: {count}")
        
        model_counts = Counter(p['model'] for p in all_proposals)
        logger.info(f"\nBy model:")
        for key, count in model_counts.items():
            logger.info(f"  {key}: {count}")
        
        return all_proposals
    
    def _clean_text_field(self, text: Any) -> str:
        """Remove newlines and clean text for CSV export"""
        if text is None:
            return ''
        if not isinstance(text, str):
            return str(text)
        # Replace all newlines with spaces
        return text.replace('\n', ' ').replace('\r', ' ')
    
    def save_to_csv(self, proposals: List[Dict[str, Any]], output_filename: str = "all_proposals_combined_no_role.csv"):
        """Save proposals to CSV file with proper quoting for special characters"""
        output_path = Path(output_filename)
        
        # Define column order
        fieldnames = [
            'proposal_id',
            'who',
            'role',
            'model',
            'title',
            'abstract',
            'authors',
            'full_draft',
            'proposal_status',
            'ranking'
        ]
        
        # Clean all text fields to remove newlines
        cleaned_proposals = []
        for proposal in proposals:
            cleaned_proposal = {}
            for key, value in proposal.items():
                if isinstance(value, str):
                    cleaned_proposal[key] = self._clean_text_field(value)
                else:
                    cleaned_proposal[key] = value
            cleaned_proposals.append(cleaned_proposal)
        
        # Write CSV with proper quoting
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,  # Quote all fields
                doublequote=True,       # Escape quotes by doubling them
                lineterminator='\n'      # Use Unix-style line endings
            )
            writer.writeheader()
            writer.writerows(cleaned_proposals)
        
        logger.info(f"\n✓ Saved combined proposals to {output_path}")
        logger.info(f"  - Rows: {len(proposals)}")
        logger.info(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine all human and AI proposals into a single CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_proposals_combined_no_role.csv",
        help="Output CSV filename (default: all_proposals_combined_no_role.csv)"
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Display sample rows from the combined dataset"
    )
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = ProposalCombiner()
    
    # Combine all proposals
    proposals = combiner.combine_all_proposals()
    
    # Show sample if requested
    if args.show_sample:
        print("\n=== Sample Rows ===")
        for i, p in enumerate(proposals[:10], 1):
            print(f"{i}. {p['proposal_id']:50s} | {p['who']:8s} | {p['role']:25s} | {p['model']:20s} | {p['title'][:50]}")
    
    # Save to CSV
    combiner.save_to_csv(proposals, args.output)


if __name__ == "__main__":
    main()

