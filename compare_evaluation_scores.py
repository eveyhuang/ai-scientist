#!/usr/bin/env python3
"""
Compare evaluation scores between two JSON files (e.g., human vs AI proposals)
Automatically detects and matches score fields across different evaluation structures
Modeled after analyze_evaluations.py but with automatic field detection
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from scipy import stats
from difflib import SequenceMatcher
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class EvaluationScoreComparator:
    """Compare evaluation scores between two sets of proposals"""
    
    def __init__(self, file1: str, file2: str, similarity_threshold: float = 0.6):
        """
        Initialize with paths to two evaluation JSON files
        
        Args:
            file1: Path to first evaluation JSON file
            file2: Path to second evaluation JSON file
            similarity_threshold: Minimum similarity (0-1) for automatic field matching
        """
        self.file1 = Path(file1)
        self.file2 = Path(file2)
        self.similarity_threshold = similarity_threshold
        
        # Load data
        self.data1 = self._load_json(file1)
        self.data2 = self._load_json(file2)
        
        # Extract scores (automatically flatten all structures)
        self.scores1 = self._extract_all_scores(self.data1, 'group1')
        self.scores2 = self._extract_all_scores(self.data2, 'group2')
        
        # Metadata
        self.metadata1 = self.data1.get('metadata', {})
        self.metadata2 = self.data2.get('metadata', {})
        
        # Create distinguishable output directory
        self.output_dir = self._create_output_directory()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Automatically match score fields
        self.field_mapping = self._auto_match_fields()
        
    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_output_directory(self) -> Path:
        """Create a distinguishable output directory based on input files"""
        # Extract source information
        source1 = self.scores1['source'].iloc[0] if len(self.scores1) > 0 else 'unknown1'
        source2 = self.scores2['source'].iloc[0] if len(self.scores2) > 0 else 'unknown2'
        
        # Extract template information - try metadata first, then from evaluations
        templates1 = self.metadata1.get('evaluation_templates', None)
        if templates1 is None and self.data1.get('evaluations'):
            # Extract from first evaluation entry
            first_eval = self.data1['evaluations'][0]
            template1 = first_eval.get('evaluation_template', 'unknown')
        else:
            template1 = templates1[0] if templates1 else 'unknown'
        
        templates2 = self.metadata2.get('evaluation_templates', None)
        if templates2 is None and self.data2.get('evaluations'):
            # Extract from first evaluation entry
            first_eval = self.data2['evaluations'][0]
            template2 = first_eval.get('evaluation_template', 'unknown')
        else:
            template2 = templates2[0] if templates2 else 'unknown'
        
        # Shorten template names for cleaner directory names
        template1_short = template1.replace('_', '-')[:15]
        template2_short = template2.replace('_', '-')[:15]
        
        # Create distinguishable directory name
        dir_name = f"{source1}_{template1_short}_vs_{source2}_{template2_short}"
        
        return Path("reports") / dir_name
    
    def _generate_filename_prefix(self) -> str:
        """Generate a distinguishable filename prefix based on input files"""
        # Extract source information
        source1 = self.scores1['source'].iloc[0] if len(self.scores1) > 0 else 'unknown1'
        source2 = self.scores2['source'].iloc[0] if len(self.scores2) > 0 else 'unknown2'
        
        # Extract template information - try metadata first, then from evaluations
        templates1 = self.metadata1.get('evaluation_templates', None)
        if templates1 is None and self.data1.get('evaluations'):
            # Extract from first evaluation entry
            first_eval = self.data1['evaluations'][0]
            template1 = first_eval.get('evaluation_template', 'unknown')
        else:
            template1 = templates1[0] if templates1 else 'unknown'
        
        templates2 = self.metadata2.get('evaluation_templates', None)
        if templates2 is None and self.data2.get('evaluations'):
            # Extract from first evaluation entry
            first_eval = self.data2['evaluations'][0]
            template2 = first_eval.get('evaluation_template', 'unknown')
        else:
            template2 = templates2[0] if templates2 else 'unknown'
        
        # Shorten template names
        template1_short = template1.replace('_', '-')[:15]
        template2_short = template2.replace('_', '-')[:15]
        
        # Create prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"comparison_{source1}-{template1_short}_vs_{source2}-{template2_short}_{timestamp}"
        
        return prefix
    
    def _normalize_field_name(self, name: str) -> str:
        """Normalize field name for matching"""
        # Convert to lowercase, remove punctuation, normalize spaces
        name = name.lower()
        name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
        name = re.sub(r'\s+', '_', name.strip())  # Normalize spaces
        return name
    
    def _similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two field names"""
        norm1 = self._normalize_field_name(name1)
        norm2 = self._normalize_field_name(name2)
        
        # Direct match
        if norm1 == norm2:
            return 1.0
        
        # Fuzzy match
        ratio = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Also check if one contains the other
        if norm1 in norm2 or norm2 in norm1:
            ratio = max(ratio, 0.8)
        
        return ratio
    
    def _extract_all_scores(self, data: Dict[str, Any], source: str) -> pd.DataFrame:
        """
        Automatically extract all scores from evaluation structure
        Handles nested structures (categories -> subcriteria) and flat structures
        """
        rows = []
        
        for eval_result in data.get('evaluations', []):
            proposal_id = eval_result.get('proposal_id', 'unknown')
            proposal_who = eval_result.get('proposal_who', source)
            proposal_model = eval_result.get('proposal_model', None)
            proposal_role = eval_result.get('proposal_role', None)
            
            eval_response = eval_result.get('evaluation_response', {})
            if not isinstance(eval_response, dict):
                continue
            
            evaluation = eval_response.get('evaluation', {})
            if not evaluation:
                continue
            
            # Start with base metadata
            row = {
                'proposal_id': proposal_id,
                'source': proposal_who,
                'template': eval_result.get('evaluation_template', 'unknown')
            }
            
            if proposal_model:
                row['ai_model'] = proposal_model
            if proposal_role:
                row['ai_role'] = proposal_role
            
            # Extract overall score (try different possible keys)
            overall_keys = ['overall_score', 'final_numeric_score', 'overall_rating']
            for key in overall_keys:
                val = evaluation.get(key)
                if val is not None:
                    if isinstance(val, dict):
                        val = val.get('final_numeric_score')
                    if val is not None:
                        row['overall_score'] = float(val)
                        break
            
            # Extract all criteria scores
            # Handle flat structure (comprehensive template)
            criteria_list = evaluation.get('criteria', [])
            for criterion in criteria_list:
                if isinstance(criterion, dict):
                    crit_name = criterion.get('criterion', '')
                    score = criterion.get('score')
                    if crit_name and score is not None:
                        normalized_name = self._normalize_field_name(crit_name)
                        row[normalized_name] = float(score)
            
            # Handle nested structure (human_criteria template)
            criteria_scores = evaluation.get('criteria_scores', [])
            for category_data in criteria_scores:
                if not isinstance(category_data, dict):
                    continue
                
                category = category_data.get('category', '')
                category_avg = category_data.get('category_average')
                
                # Store category average
                if category and category_avg is not None:
                    normalized_category = self._normalize_field_name(category)
                    row[normalized_category] = float(category_avg)
                
                # Store subcriteria
                subcriteria = category_data.get('subcriteria', [])
                for subcrit in subcriteria:
                    if isinstance(subcrit, dict):
                        subcrit_name = subcrit.get('criterion', '')
                        score = subcrit.get('score')
                        if subcrit_name and score is not None:
                            normalized_name = self._normalize_field_name(subcrit_name)
                            row[normalized_name] = float(score)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _auto_match_fields(self) -> Dict[str, Tuple[str, str]]:
        """
        Automatically match score fields between datasets
        Returns dict: {matched_name: (field1, field2)}
        """
        # Get all score columns (exclude metadata columns)
        metadata_cols = {'proposal_id', 'source', 'template', 'ai_model', 'ai_role'}
        fields1 = set(self.scores1.columns) - metadata_cols - {'overall_score'}
        fields2 = set(self.scores2.columns) - metadata_cols - {'overall_score'}
        
        mapping = {}
        
        # Always include overall_score
        if 'overall_score' in self.scores1.columns and 'overall_score' in self.scores2.columns:
            mapping['overall_score'] = ('overall_score', 'overall_score')
        
        # Match fields by similarity
        matched_fields2 = set()
        matched_fields1 = set()
        
        for field1 in sorted(fields1):
            if field1 in matched_fields1:
                continue
            
            best_match = None
            best_similarity = 0
            
            for field2 in sorted(fields2):
                if field2 in matched_fields2:
                    continue
                
                similarity = self._similarity(field1, field2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = field2
            
            # If similarity is above threshold, create mapping
            if best_match and best_similarity >= self.similarity_threshold:
                # Use more readable name for the mapping key
                matched_name = field1 if len(field1) <= len(best_match) else best_match
                mapping[matched_name] = (field1, best_match)
                matched_fields2.add(best_match)
                matched_fields1.add(field1)
        
        # Include unmatched fields with their original names
        for field1 in sorted(fields1):
            if field1 not in matched_fields1:
                mapping[f"{field1}_group1_only"] = (field1, None)
        
        for field2 in sorted(fields2):
            if field2 not in matched_fields2:
                mapping[f"{field2}_group2_only"] = (None, field2)
        
        return mapping
    
    def compare_overall_scores(self) -> Dict[str, Any]:
        """Compare overall scores between two groups"""
        scores1 = self.scores1['overall_score'].dropna()
        scores2 = self.scores2['overall_score'].dropna()
        
        if len(scores1) == 0 or len(scores2) == 0:
            return {'error': 'No overall scores found'}
        
        results = {
            'group1': {
                'mean': float(scores1.mean()),
                'median': float(scores1.median()),
                'std': float(scores1.std()),
                'min': float(scores1.min()),
                'max': float(scores1.max()),
                'count': len(scores1)
            },
            'group2': {
                'mean': float(scores2.mean()),
                'median': float(scores2.median()),
                'std': float(scores2.std()),
                'min': float(scores2.min()),
                'max': float(scores2.max()),
                'count': len(scores2)
            }
        }
        
        # Statistical tests
        if len(scores1) > 1 and len(scores2) > 1:
            # T-test (parametric)
            t_stat, t_p = stats.ttest_ind(scores1, scores2)
            results['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': bool(t_p < 0.05)
            }
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            results['mannwhitney'] = {
                'statistic': float(u_stat),
                'p_value': float(u_p),
                'significant': bool(u_p < 0.05)
            }
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(scores1) - 1) * scores1.std()**2 + 
                                  (len(scores2) - 1) * scores2.std()**2) / 
                                 (len(scores1) + len(scores2) - 2))
            cohens_d = (scores1.mean() - scores2.mean()) / pooled_std if pooled_std > 0 else 0
            results['effect_size'] = {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        
        return results
    
    def compare_fields(self) -> Dict[str, Any]:
        """Compare all matched fields automatically"""
        comparisons = {}
        
        for matched_name, (field1, field2) in self.field_mapping.items():
            # Skip if either field is None (unmatched fields)
            if field1 is None or field2 is None:
                continue
            
            scores1 = self.scores1[field1].dropna() if field1 in self.scores1.columns else pd.Series()
            scores2 = self.scores2[field2].dropna() if field2 in self.scores2.columns else pd.Series()
            
            if len(scores1) == 0 or len(scores2) == 0:
                continue
            
            comparisons[matched_name] = {
                'group1': {
                    'mean': float(scores1.mean()),
                    'median': float(scores1.median()),
                    'std': float(scores1.std()),
                    'count': len(scores1)
                },
                'group2': {
                    'mean': float(scores2.mean()),
                    'median': float(scores2.median()),
                    'std': float(scores2.std()),
                    'count': len(scores2)
                },
                'field1': field1,
                'field2': field2
            }
            
            # Statistical test
            if len(scores1) > 1 and len(scores2) > 1:
                try:
                    u_stat, u_p = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                    comparisons[matched_name]['mannwhitney'] = {
                        'p_value': float(u_p),
                        'significant': bool(u_p < 0.05)
                    }
                except Exception as e:
                    comparisons[matched_name]['mannwhitney'] = {
                        'error': str(e)
                    }
        
        return comparisons
    
    def compare_by_model(self) -> Dict[str, Any]:
        """Compare scores by AI model (if applicable)"""
        results = {}
        
        # Check which dataset has ai_model column
        for idx, scores_df in enumerate([self.scores1, self.scores2], 1):
            if 'ai_model' not in scores_df.columns:
                continue
            
            group_results = {}
            for model in scores_df['ai_model'].unique():
                if pd.isna(model):
                    continue
                model_scores = scores_df[scores_df['ai_model'] == model]['overall_score'].dropna()
                
                if len(model_scores) > 0:
                    group_results[str(model)] = {
                        'mean': float(model_scores.mean()),
                        'median': float(model_scores.median()),
                        'std': float(model_scores.std()),
                        'count': len(model_scores)
                    }
            
            if group_results:
                results[f'group{idx}'] = group_results
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        # Generate distinguishable filename prefix
        prefix = self._generate_filename_prefix()
        
        # 1. Overall score distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scores1 = self.scores1['overall_score'].dropna()
        scores2 = self.scores2['overall_score'].dropna()
        
        group1_label = self.scores1['source'].iloc[0] if len(self.scores1) > 0 else 'Group 1'
        group2_label = self.scores2['source'].iloc[0] if len(self.scores2) > 0 else 'Group 2'
        
        # Histogram
        ax = axes[0, 0]
        ax.hist(scores1, bins=15, alpha=0.6, label=f'{group1_label} (n={len(scores1)})', 
                color='blue', edgecolor='black')
        ax.hist(scores2, bins=15, alpha=0.6, label=f'{group2_label} (n={len(scores2)})', 
                color='red', edgecolor='black')
        ax.axvline(scores1.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'{group1_label} mean: {scores1.mean():.2f}')
        ax.axvline(scores2.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'{group2_label} mean: {scores2.mean():.2f}')
        ax.set_xlabel('Overall Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Overall Score Distribution: {group1_label} vs {group2_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[0, 1]
        data = [scores1, scores2]
        bp = ax.boxplot(data, tick_labels=[group1_label, group2_label], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Overall Score')
        ax.set_title('Overall Score Comparison (Box Plot)')
        ax.grid(True, alpha=0.3)
        
        # Violin plot
        ax = axes[1, 0]
        df_plot = pd.DataFrame({
            'Score': pd.concat([scores1, scores2]),
            'Source': [group1_label] * len(scores1) + [group2_label] * len(scores2)
        })
        sns.violinplot(data=df_plot, x='Source', y='Score', hue='Source', ax=ax, 
                       palette=['lightblue', 'lightcoral'], legend=False)
        ax.set_title('Overall Score Distribution (Violin Plot)')
        ax.grid(True, alpha=0.3)
        
        # Field comparison bar chart
        ax = axes[1, 1]
        comparisons = self.compare_fields()
        # Filter to only matched fields (exclude overall_score for this plot)
        matched_fields = {k: v for k, v in comparisons.items() if k != 'overall_score'}
        
        field_names = []
        group1_means = []
        group2_means = []
        
        for field_name, field_data in sorted(matched_fields.items()):
            field_names.append(field_name.replace('_', ' ').title())
            group1_means.append(field_data['group1']['mean'])
            group2_means.append(field_data['group2']['mean'])
        
        if len(field_names) > 0:
            x = np.arange(len(field_names))
            width = 0.35
            ax.bar(x - width/2, group1_means, width, label=group1_label, color='lightblue', edgecolor='black')
            ax.bar(x + width/2, group2_means, width, label=group2_label, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Field')
            ax.set_ylabel('Mean Score')
            ax.set_title('Mean Scores by Field (Auto-Matched)')
            ax.set_xticks(x)
            ax.set_xticklabels(field_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Field-by-field detailed comparison
        if len(comparisons) > 0:
            # Exclude overall_score from detailed plots
            detailed_fields = {k: v for k, v in comparisons.items() if k != 'overall_score'}
            
            if len(detailed_fields) > 0:
                n_fields = len(detailed_fields)
                n_cols = 3
                n_rows = (n_fields + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
                axes = axes.flatten() if n_fields > 1 else [axes]
                
                for idx, (field_name, field_data) in enumerate(sorted(detailed_fields.items())):
                    ax = axes[idx]
                    
                    field1 = field_data['field1']
                    field2 = field_data['field2']
                    
                    scores1 = self.scores1[field1].dropna()
                    scores2 = self.scores2[field2].dropna()
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        data = [scores1, scores2]
                        bp = ax.boxplot(data, tick_labels=[group1_label, group2_label], patch_artist=True)
                        bp['boxes'][0].set_facecolor('lightblue')
                        bp['boxes'][1].set_facecolor('lightcoral')
                        
                        # Add mean lines
                        ax.axhline(scores1.mean(), color='blue', linestyle='--', alpha=0.7, linewidth=1)
                        ax.axhline(scores2.mean(), color='red', linestyle='--', alpha=0.7, linewidth=1)
                        
                        # Add p-value if available
                        if 'mannwhitney' in field_data and 'p_value' in field_data['mannwhitney']:
                            p_val = field_data['mannwhitney']['p_value']
                            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                            ax.text(0.5, 0.95, f'p={p_val:.3f} {sig}', 
                                   transform=ax.transAxes, ha='center', va='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        ax.set_ylabel('Score')
                        ax.set_title(field_name.replace('_', ' ').title())
                        ax.grid(True, alpha=0.3, axis='y')
                
                # Hide unused subplots
                for idx in range(len(detailed_fields), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{prefix}_field_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self) -> str:
        """Generate comprehensive comparison report"""
        report = []
        report.append("# Evaluation Score Comparison Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # File information
        report.append("## Input Files\n\n")
        # Extract template information for report
        templates1 = self.metadata1.get('evaluation_templates', None)
        if templates1 is None and self.data1.get('evaluations'):
            template1 = self.data1['evaluations'][0].get('evaluation_template', 'Unknown')
        else:
            template1 = templates1[0] if templates1 else 'Unknown'
        
        templates2 = self.metadata2.get('evaluation_templates', None)
        if templates2 is None and self.data2.get('evaluations'):
            template2 = self.data2['evaluations'][0].get('evaluation_template', 'Unknown')
        else:
            template2 = templates2[0] if templates2 else 'Unknown'
        
        report.append(f"**File 1:** `{self.file1.name}`\n")
        report.append(f"- Source: {self.scores1['source'].iloc[0] if len(self.scores1) > 0 else 'Unknown'}\n")
        report.append(f"- Evaluations: {len(self.scores1)}\n")
        report.append(f"- Template: {template1}\n\n")
        
        report.append(f"**File 2:** `{self.file2.name}`\n")
        report.append(f"- Source: {self.scores2['source'].iloc[0] if len(self.scores2) > 0 else 'Unknown'}\n")
        report.append(f"- Evaluations: {len(self.scores2)}\n")
        report.append(f"- Template: {template2}\n\n")
        
        # Show detected fields
        report.append("## Detected Fields and Auto-Matching\n\n")
        report.append("**File 1 Fields:**\n")
        fields1 = sorted([c for c in self.scores1.columns 
                         if c not in {'proposal_id', 'source', 'template', 'ai_model', 'ai_role'}])
        for field in fields1:
            report.append(f"- `{field}`\n")
        
        report.append("\n**File 2 Fields:**\n")
        fields2 = sorted([c for c in self.scores2.columns 
                         if c not in {'proposal_id', 'source', 'template', 'ai_model', 'ai_role'}])
        for field in fields2:
            report.append(f"- `{field}`\n")
        
        report.append("\n**Auto-Matched Fields:**\n")
        for matched_name, (field1, field2) in sorted(self.field_mapping.items()):
            if field1 and field2:
                report.append(f"- `{matched_name}`: file1=`{field1}` ↔ file2=`{field2}`\n")
        
        # Show unmatched fields
        unmatched1 = [k for k, (f1, f2) in self.field_mapping.items() if f1 and not f2]
        unmatched2 = [k for k, (f1, f2) in self.field_mapping.items() if f2 and not f1]
        
        if unmatched1:
            report.append("\n**File 1 Only (no match):**\n")
            for name in unmatched1:
                report.append(f"- `{name}`\n")
        
        if unmatched2:
            report.append("\n**File 2 Only (no match):**\n")
            for name in unmatched2:
                report.append(f"- `{name}`\n")
        
        report.append("\n")
        
        # Overall comparison
        report.append("## Overall Score Comparison\n\n")
        overall = self.compare_overall_scores()
        
        if 'error' not in overall:
            report.append("### Summary Statistics\n\n")
            group1_label = self.scores1['source'].iloc[0] if len(self.scores1) > 0 else 'Group 1'
            group2_label = self.scores2['source'].iloc[0] if len(self.scores2) > 0 else 'Group 2'
            
            report.append(f"**{group1_label}:**\n")
            report.append(f"- Mean: {overall['group1']['mean']:.2f}\n")
            report.append(f"- Median: {overall['group1']['median']:.2f}\n")
            report.append(f"- Std Dev: {overall['group1']['std']:.2f}\n")
            report.append(f"- Range: {overall['group1']['min']:.2f} - {overall['group1']['max']:.2f}\n")
            report.append(f"- Count: {overall['group1']['count']}\n\n")
            
            report.append(f"**{group2_label}:**\n")
            report.append(f"- Mean: {overall['group2']['mean']:.2f}\n")
            report.append(f"- Median: {overall['group2']['median']:.2f}\n")
            report.append(f"- Std Dev: {overall['group2']['std']:.2f}\n")
            report.append(f"- Range: {overall['group2']['min']:.2f} - {overall['group2']['max']:.2f}\n")
            report.append(f"- Count: {overall['group2']['count']}\n\n")
            
            report.append(f"**Difference:** {overall['group1']['mean'] - overall['group2']['mean']:+.2f} points\n\n")
            
            # Statistical tests
            if 't_test' in overall:
                report.append("### Statistical Tests\n\n")
                report.append(f"**Independent Samples T-test:**\n")
                report.append(f"- t-statistic: {overall['t_test']['statistic']:.3f}\n")
                report.append(f"- p-value: {overall['t_test']['p_value']:.4f}\n")
                report.append(f"- Significant: {'Yes' if overall['t_test']['significant'] else 'No'} (α=0.05)\n\n")
                
                report.append(f"**Mann-Whitney U Test (Non-parametric):**\n")
                report.append(f"- U-statistic: {overall['mannwhitney']['statistic']:.3f}\n")
                report.append(f"- p-value: {overall['mannwhitney']['p_value']:.4f}\n")
                report.append(f"- Significant: {'Yes' if overall['mannwhitney']['significant'] else 'No'} (α=0.05)\n\n")
                
                report.append(f"**Effect Size (Cohen's d):**\n")
                report.append(f"- d = {overall['effect_size']['cohens_d']:.3f}\n")
                report.append(f"- Interpretation: {overall['effect_size']['interpretation']}\n\n")
        
        # Field comparison
        report.append("## Field-by-Field Comparison\n\n")
        field_comp = self.compare_fields()
        
        report.append("| Field | Group 1 Mean | Group 2 Mean | Difference | p-value | Significant |\n")
        report.append("|-------|--------------|--------------|------------|---------|-------------|\n")
        
        for field_name, field_data in sorted(field_comp.items()):
            mean1 = field_data['group1']['mean']
            mean2 = field_data['group2']['mean']
            diff = mean1 - mean2
            p_val = field_data.get('mannwhitney', {}).get('p_value', np.nan)
            sig = 'Yes' if field_data.get('mannwhitney', {}).get('significant', False) else 'No'
            
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            report.append(f"| {field_name.replace('_', ' ').title()} | {mean1:.2f} | {mean2:.2f} | {diff:+.2f} | {p_str} | {sig} |\n")
        
        report.append("\n")
        
        # Model comparison (if applicable)
        model_comp = self.compare_by_model()
        if model_comp:
            report.append("## By AI Model Comparison\n\n")
            
            for group_name, models in model_comp.items():
                report.append(f"**{group_name.title()}:**\n\n")
                for model, stats in sorted(models.items()):
                    report.append(f"- **{model}**: Mean = {stats['mean']:.2f}, Count = {stats['count']}\n")
                report.append("\n")
        
        return "".join(report)
    
    def save_results(self):
        """Save all results"""
        # Generate distinguishable filename prefix
        prefix = self._generate_filename_prefix()
        
        # Save report
        report = self.generate_report()
        report_file = self.output_dir / f'{prefix}_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save statistics
        stats_data = {
            'field_mapping': {k: {'field1': v[0], 'field2': v[1]} 
                             for k, v in self.field_mapping.items()},
            'overall_comparison': self.compare_overall_scores(),
            'field_comparison': self.compare_fields(),
            'model_comparison': self.compare_by_model()
        }
        
        stats_file = self.output_dir / f'{prefix}_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {self.output_dir}/")
        logger.info(f"  - {report_file.name}")
        logger.info(f"  - {stats_file.name}")
        logger.info(f"  - {prefix}_overall_comparison.png")
        logger.info(f"  - {prefix}_field_comparison.png")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Compare evaluation scores between two JSON files with automatic field detection"
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to first evaluation JSON file"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Path to second evaluation JSON file"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Minimum similarity (0-1) for automatic field matching (default: 0.6)"
    )
    parser.add_argument(
        "--filter-model-file1",
        type=str,
        default=None,
        help="Filter file1 by AI model (e.g., 'gpt-4', 'gemini-2.5-pro')"
    )
    parser.add_argument(
        "--filter-model-file2",
        type=str,
        default=None,
        help="Filter file2 by AI model (e.g., 'gpt-4', 'gemini-2.5-pro')"
    )
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = EvaluationScoreComparator(args.file1, args.file2, args.similarity_threshold)
    
    # Apply model filters if specified
    if args.filter_model_file1 and 'ai_model' in comparator.scores1.columns:
        original_len = len(comparator.scores1)
        available_models = comparator.scores1['ai_model'].unique()
        comparator.scores1 = comparator.scores1[comparator.scores1['ai_model'] == args.filter_model_file1]
        logger.info(f"Filtered file1 by model '{args.filter_model_file1}': {original_len} → {len(comparator.scores1)} proposals")
        if len(comparator.scores1) == 0:
            logger.error(f"No proposals found with model '{args.filter_model_file1}' in file1")
            logger.error(f"Available models in file1: {list(available_models)}")
            return
    
    if args.filter_model_file2 and 'ai_model' in comparator.scores2.columns:
        original_len = len(comparator.scores2)
        available_models = comparator.scores2['ai_model'].unique()
        comparator.scores2 = comparator.scores2[comparator.scores2['ai_model'] == args.filter_model_file2]
        logger.info(f"Filtered file2 by model '{args.filter_model_file2}': {original_len} → {len(comparator.scores2)} proposals")
        if len(comparator.scores2) == 0:
            logger.error(f"No proposals found with model '{args.filter_model_file2}' in file2")
            logger.error(f"Available models in file2: {list(available_models)}")
            return
    
    # Recompute field mapping after filtering (in case column availability changed)
    if args.filter_model_file1 or args.filter_model_file2:
        comparator.field_mapping = comparator._auto_match_fields()
        logger.info(f"Recomputed field mappings after filtering")
    
    # Print detected mappings
    logger.info("\n=== AUTO-DETECTED FIELD MAPPINGS ===\n")
    for matched_name, (field1, field2) in sorted(comparator.field_mapping.items()):
        if field1 and field2:
            logger.info(f"✓ {matched_name}")
            logger.info(f"    File 1: {field1}")
            logger.info(f"    File 2: {field2}\n")
        elif field1:
            logger.info(f"⚠ {matched_name} (file 1 only, no file 2 match)")
            logger.info(f"    File 1: {field1}\n")
        elif field2:
            logger.info(f"⚠ {matched_name} (file 2 only, no file 1 match)")
            logger.info(f"    File 2: {field2}\n")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    comparator.create_visualizations()
    
    # Save results
    logger.info("\nSaving results...")
    comparator.save_results()
    
    # Print summary
    logger.info("\n=== COMPARISON SUMMARY ===")
    overall = comparator.compare_overall_scores()
    if 'error' not in overall:
        group1_label = comparator.scores1['source'].iloc[0] if len(comparator.scores1) > 0 else 'Group 1'
        group2_label = comparator.scores2['source'].iloc[0] if len(comparator.scores2) > 0 else 'Group 2'
        
        logger.info(f"\n{group1_label} Mean: {overall['group1']['mean']:.2f} ± {overall['group1']['std']:.2f}")
        logger.info(f"{group2_label} Mean: {overall['group2']['mean']:.2f} ± {overall['group2']['std']:.2f}")
        logger.info(f"Difference: {overall['group1']['mean'] - overall['group2']['mean']:+.2f} points")
        
        if 'mannwhitney' in overall:
            logger.info(f"\nMann-Whitney U test p-value: {overall['mannwhitney']['p_value']:.4f}")
            logger.info(f"Significant difference: {'Yes' if overall['mannwhitney']['significant'] else 'No'} (α=0.05)")
    
    logger.info("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()

