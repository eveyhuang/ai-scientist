#!/usr/bin/env python3
"""
Analyze evaluation results from proposal comparisons
Extract scores, generate statistics, and create visualizations
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationAnalyzer:
    """Analyze evaluation results"""
    
    def __init__(self, evaluations_file: str):
        """Initialize with evaluation results file"""
        self.evaluations_file = Path(evaluations_file)
        self.data = self.load_evaluations()
        self.evaluations = self.data.get('evaluations', [])
        self.metadata = self.data.get('metadata', {})
        
    def load_evaluations(self) -> Dict[str, Any]:
        """Load evaluation results from JSON file"""
        try:
            with open(self.evaluations_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluations: {e}")
            return {}
    
    def extract_scores(self, evaluation_text: str) -> Dict[str, Dict[str, float]]:
        """
        Extract numerical scores from evaluation text
        Returns dict with structure: {criterion: {human: score, ai: score}}
        """
        scores = {}
        
        # Common patterns for score extraction
        patterns = [
            # Pattern: "Criterion: Human X/10, AI Y/10"
            r'(\w+(?:\s+\w+)*)\s*:\s*(?:Human|Proposal\s*A)[^\d]*(\d+(?:\.\d+)?)\s*/\s*10[^\d]*(?:AI|Proposal\s*B)[^\d]*(\d+(?:\.\d+)?)\s*/\s*10',
            # Pattern: "Human: X/10" and "AI: Y/10" on separate lines
            r'(?:Human|Proposal\s*A)[^\d]*(\d+(?:\.\d+)?)\s*/\s*10',
            # Pattern: "Score: X" format
            r'[Ss]core\s*:\s*(\d+(?:\.\d+)?)',
        ]
        
        # Try to extract structured scores
        for pattern in patterns:
            matches = re.finditer(pattern, evaluation_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 3:
                    criterion = match.group(1).strip()
                    human_score = float(match.group(2))
                    ai_score = float(match.group(3))
                    scores[criterion] = {'human': human_score, 'ai': ai_score}
        
        return scores
    
    def extract_overall_preference(self, evaluation_text: str) -> Optional[str]:
        """Extract which proposal was preferred overall"""
        text_lower = evaluation_text.lower()
        
        # Look for clear preference statements
        if re.search(r'\b(?:recommend|prefer|stronger|better)\s+(?:the\s+)?(?:human|proposal\s*a)', text_lower):
            return 'human'
        elif re.search(r'\b(?:recommend|prefer|stronger|better)\s+(?:the\s+)?(?:ai|proposal\s*b)', text_lower):
            return 'ai'
        
        # Look for "Proposal A/B is preferred"
        if re.search(r'proposal\s*a\s+is\s+(?:preferred|recommended|stronger)', text_lower):
            return 'human'
        elif re.search(r'proposal\s*b\s+is\s+(?:preferred|recommended|stronger)', text_lower):
            return 'ai'
        
        return None
    
    def analyze_by_template(self) -> Dict[str, Any]:
        """Analyze evaluations grouped by evaluation template"""
        by_template = defaultdict(list)
        
        for eval_result in self.evaluations:
            template = eval_result.get('evaluation_template', 'unknown')
            by_template[template].append(eval_result)
        
        analysis = {}
        for template, evals in by_template.items():
            # Extract preferences
            preferences = []
            for eval_result in evals:
                pref = self.extract_overall_preference(eval_result.get('evaluation_response', ''))
                if pref:
                    preferences.append(pref)
            
            pref_counts = Counter(preferences)
            
            analysis[template] = {
                'total_evaluations': len(evals),
                'preferences': dict(pref_counts),
                'human_preferred_pct': pref_counts['human'] / len(preferences) * 100 if preferences else 0,
                'ai_preferred_pct': pref_counts['ai'] / len(preferences) * 100 if preferences else 0,
                'unclear_pct': (len(evals) - len(preferences)) / len(evals) * 100 if evals else 0
            }
        
        return analysis
    
    def analyze_by_role(self) -> Dict[str, Any]:
        """Analyze evaluations grouped by role description"""
        by_role = defaultdict(list)
        
        for eval_result in self.evaluations:
            role = eval_result.get('role_description', 'unknown')
            by_role[role].append(eval_result)
        
        analysis = {}
        for role, evals in by_role.items():
            preferences = []
            for eval_result in evals:
                pref = self.extract_overall_preference(eval_result.get('evaluation_response', ''))
                if pref:
                    preferences.append(pref)
            
            pref_counts = Counter(preferences)
            
            analysis[role] = {
                'total_evaluations': len(evals),
                'preferences': dict(pref_counts),
                'human_preferred_pct': pref_counts['human'] / len(preferences) * 100 if preferences else 0,
                'ai_preferred_pct': pref_counts['ai'] / len(preferences) * 100 if preferences else 0
            }
        
        return analysis
    
    def analyze_by_proposal_pair(self) -> Dict[str, Any]:
        """Analyze evaluations grouped by proposal pairs"""
        by_pair = defaultdict(list)
        
        for eval_result in self.evaluations:
            pair_key = f"{eval_result.get('human_proposal_id')}_{eval_result.get('ai_proposal_id')}"
            by_pair[pair_key].append(eval_result)
        
        analysis = {}
        for pair_key, evals in by_pair.items():
            preferences = []
            for eval_result in evals:
                pref = self.extract_overall_preference(eval_result.get('evaluation_response', ''))
                if pref:
                    preferences.append(pref)
            
            pref_counts = Counter(preferences)
            
            # Get titles
            sample_eval = evals[0]
            
            analysis[pair_key] = {
                'human_title': sample_eval.get('human_proposal_title'),
                'ai_title': sample_eval.get('ai_proposal_title'),
                'total_evaluations': len(evals),
                'preferences': dict(pref_counts),
                'consensus_score': max(pref_counts.values()) / len(preferences) if preferences else 0
            }
        
        return analysis
    
    def get_score_statistics(self) -> Dict[str, Any]:
        """Calculate statistics on extracted scores"""
        all_scores = {'human': [], 'ai': []}
        
        for eval_result in self.evaluations:
            scores = self.extract_scores(eval_result.get('evaluation_response', ''))
            for criterion, score_dict in scores.items():
                if 'human' in score_dict:
                    all_scores['human'].append(score_dict['human'])
                if 'ai' in score_dict:
                    all_scores['ai'].append(score_dict['ai'])
        
        if not all_scores['human'] or not all_scores['ai']:
            return {
                'note': 'No numerical scores could be extracted from evaluations'
            }
        
        import statistics
        
        return {
            'human': {
                'mean': statistics.mean(all_scores['human']),
                'median': statistics.median(all_scores['human']),
                'stdev': statistics.stdev(all_scores['human']) if len(all_scores['human']) > 1 else 0,
                'min': min(all_scores['human']),
                'max': max(all_scores['human']),
                'count': len(all_scores['human'])
            },
            'ai': {
                'mean': statistics.mean(all_scores['ai']),
                'median': statistics.median(all_scores['ai']),
                'stdev': statistics.stdev(all_scores['ai']) if len(all_scores['ai']) > 1 else 0,
                'min': min(all_scores['ai']),
                'max': max(all_scores['ai']),
                'count': len(all_scores['ai'])
            }
        }
    
    def generate_summary_report(self, output_file: str = None) -> str:
        """Generate a comprehensive summary report"""
        
        report = []
        report.append("# Evaluation Analysis Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Source: {self.evaluations_file}\n\n")
        
        # Metadata
        report.append("## Metadata\n")
        for key, value in self.metadata.items():
            report.append(f"- **{key}**: {value}\n")
        report.append("\n")
        
        # Overall statistics
        report.append("## Overall Statistics\n")
        report.append(f"- Total evaluations: {len(self.evaluations)}\n")
        
        all_prefs = []
        for eval_result in self.evaluations:
            pref = self.extract_overall_preference(eval_result.get('evaluation_response', ''))
            if pref:
                all_prefs.append(pref)
        
        pref_counts = Counter(all_prefs)
        report.append(f"- Human preferred: {pref_counts['human']} ({pref_counts['human']/len(all_prefs)*100:.1f}%)\n")
        report.append(f"- AI preferred: {pref_counts['ai']} ({pref_counts['ai']/len(all_prefs)*100:.1f}%)\n")
        report.append(f"- Unclear: {len(self.evaluations) - len(all_prefs)}\n\n")
        
        # Score statistics
        report.append("## Score Statistics\n")
        score_stats = self.get_score_statistics()
        if 'note' in score_stats:
            report.append(f"_{score_stats['note']}_\n\n")
        else:
            report.append("### Human Proposals\n")
            for key, value in score_stats['human'].items():
                report.append(f"- {key}: {value:.2f}\n")
            report.append("\n### AI Proposals\n")
            for key, value in score_stats['ai'].items():
                report.append(f"- {key}: {value:.2f}\n")
            report.append("\n")
        
        # Analysis by template
        report.append("## Analysis by Evaluation Template\n")
        template_analysis = self.analyze_by_template()
        for template, stats in template_analysis.items():
            report.append(f"\n### {template}\n")
            report.append(f"- Total evaluations: {stats['total_evaluations']}\n")
            report.append(f"- Human preferred: {stats['human_preferred_pct']:.1f}%\n")
            report.append(f"- AI preferred: {stats['ai_preferred_pct']:.1f}%\n")
            report.append(f"- Unclear: {stats['unclear_pct']:.1f}%\n")
        
        # Analysis by role
        report.append("\n## Analysis by Role Description\n")
        role_analysis = self.analyze_by_role()
        for role, stats in role_analysis.items():
            report.append(f"\n### {role}\n")
            report.append(f"- Total evaluations: {stats['total_evaluations']}\n")
            report.append(f"- Human preferred: {stats['human_preferred_pct']:.1f}%\n")
            report.append(f"- AI preferred: {stats['ai_preferred_pct']:.1f}%\n")
        
        # Analysis by proposal pair
        report.append("\n## Analysis by Proposal Pair\n")
        pair_analysis = self.analyze_by_proposal_pair()
        
        # Sort by consensus score
        sorted_pairs = sorted(pair_analysis.items(), 
                            key=lambda x: x[1]['consensus_score'], 
                            reverse=True)
        
        for pair_key, stats in sorted_pairs[:10]:  # Top 10
            report.append(f"\n### Pair: {pair_key}\n")
            report.append(f"- **Human**: {stats['human_title'][:100]}...\n")
            report.append(f"- **AI**: {stats['ai_title'][:100]}...\n")
            report.append(f"- Evaluations: {stats['total_evaluations']}\n")
            report.append(f"- Preferences: {stats['preferences']}\n")
            report.append(f"- Consensus: {stats['consensus_score']:.2f}\n")
        
        report_text = "".join(report)
        
        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def export_to_csv(self, output_file: str):
        """Export evaluations to CSV format for further analysis"""
        import csv
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'evaluation_id',
                'human_proposal_id',
                'ai_proposal_id',
                'human_title',
                'ai_title',
                'evaluation_template',
                'role_description',
                'evaluator_model',
                'preference',
                'timestamp'
            ])
            
            writer.writeheader()
            
            for eval_result in self.evaluations:
                preference = self.extract_overall_preference(
                    eval_result.get('evaluation_response', '')
                )
                
                writer.writerow({
                    'evaluation_id': eval_result.get('evaluation_id'),
                    'human_proposal_id': eval_result.get('human_proposal_id'),
                    'ai_proposal_id': eval_result.get('ai_proposal_id'),
                    'human_title': eval_result.get('human_proposal_title'),
                    'ai_title': eval_result.get('ai_proposal_title'),
                    'evaluation_template': eval_result.get('evaluation_template'),
                    'role_description': eval_result.get('role_description'),
                    'evaluator_model': eval_result.get('evaluator_model'),
                    'preference': preference or 'unclear',
                    'timestamp': eval_result.get('timestamp')
                })
        
        logger.info(f"CSV exported to {output_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results from proposal comparisons"
    )
    parser.add_argument(
        "evaluation_file",
        type=str,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Output path for summary report (markdown)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output path for CSV export"
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary to console"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EvaluationAnalyzer(args.evaluation_file)
    
    # Generate report
    if args.report or args.print_summary:
        report = analyzer.generate_summary_report(output_file=args.report)
        
        if args.print_summary:
            print(report)
    
    # Export to CSV
    if args.csv:
        analyzer.export_to_csv(args.csv)
    
    # Always print some basic stats
    print("\n" + "="*60)
    print(f"Analyzed {len(analyzer.evaluations)} evaluations")
    print("="*60)
    
    # Print preferences
    all_prefs = []
    for eval_result in analyzer.evaluations:
        pref = analyzer.extract_overall_preference(eval_result.get('evaluation_response', ''))
        if pref:
            all_prefs.append(pref)
    
    pref_counts = Counter(all_prefs)
    total_clear = len(all_prefs)
    
    if total_clear > 0:
        print(f"\nPreferences (from {total_clear} clear evaluations):")
        print(f"  Human: {pref_counts['human']} ({pref_counts['human']/total_clear*100:.1f}%)")
        print(f"  AI: {pref_counts['ai']} ({pref_counts['ai']/total_clear*100:.1f}%)")
    else:
        print("\nCould not extract clear preferences from evaluations")
    
    print("\nFor detailed analysis, use --report or --csv options")


if __name__ == "__main__":
    main()

