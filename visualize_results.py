#!/usr/bin/env python3
"""
Visualize evaluation and similarity analysis results.
Creates comprehensive heatmaps, statistics, and distribution plots.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 10


class ResultVisualizer:
    """Visualize evaluation and similarity analysis results"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def visualize_evaluations(self, evaluation_file: str, custom_subfolder: str = None) -> None:
        """
        Create visualizations for pairwise evaluation results
        
        Args:
            evaluation_file: Path to evaluation JSON file
            custom_subfolder: Optional custom subfolder name (auto-generated if None)
        """
        logger.info(f"Loading evaluation results from: {evaluation_file}")
        
        # Load evaluation data
        with open(evaluation_file, 'r') as f:
            data = json.load(f)
        
        evaluations = data['evaluations']
        eval_type = data['metadata'].get('evaluation_type', 'unknown')
        
        logger.info(f"Loaded {len(evaluations)} evaluations (type: {eval_type})")
        
        # Use the directory of the input file as output directory
        input_path = Path(evaluation_file)
        
        # Update output directory to the same directory as input file
        original_output_dir = self.output_dir
        self.output_dir = input_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating visualizations in: {self.output_dir}")
        
        # Extract unique proposal IDs
        proposal_ids = set()
        for eval_data in evaluations:
            proposal_ids.add(eval_data['proposal_1_id'])
            proposal_ids.add(eval_data['proposal_2_id'])
        
        proposal_ids = sorted(list(proposal_ids))
        n_proposals = len(proposal_ids)
        logger.info(f"Number of unique proposals: {n_proposals}")
        
        # Create index mapping
        id_to_idx = {pid: idx for idx, pid in enumerate(proposal_ids)}
        
        # Extract dimension names
        dimension_names = []
        if evaluations and 'evaluation_response' in evaluations[0]:
            comparison = evaluations[0]['evaluation_response'].get('comparison', {})
            dimensions = comparison.get('dimensions', [])
            dimension_names = [d['dimension'] for d in dimensions]
        
        logger.info(f"Dimensions: {dimension_names}")
        
        # Initialize matrices for each dimension
        matrices = {}
        for dim_name in dimension_names:
            matrices[dim_name] = np.full((n_proposals, n_proposals), np.nan)
        
        # Fill matrices with scores
        for eval_data in evaluations:
            prop_1_id = eval_data['proposal_1_id']
            prop_2_id = eval_data['proposal_2_id']
            
            idx_1 = id_to_idx[prop_1_id]
            idx_2 = id_to_idx[prop_2_id]
            
            if 'evaluation_response' in eval_data:
                comparison = eval_data['evaluation_response'].get('comparison', {})
                dimensions = comparison.get('dimensions', [])
                
                for dim in dimensions:
                    dim_name = dim['dimension']
                    score = dim['score']
                    
                    if dim_name in matrices:
                        matrices[dim_name][idx_1, idx_2] = score
        
        logger.info("Matrices filled with evaluation scores")
        
        # Create combined visualization
        self._create_combined_heatmaps(
            matrices, dimension_names, proposal_ids, 
            'overlap_matrices_combined.png', eval_type
        )
        
        # Create individual detailed heatmaps
        self._create_individual_heatmaps(
            matrices, dimension_names, proposal_ids, 'matrix_', eval_type
        )
        
        # Generate statistics
        self._generate_evaluation_statistics(
            matrices, dimension_names, eval_type
        )
        
        logger.info("✓ Evaluation visualizations complete!")
        
        # Restore original output directory
        self.output_dir = original_output_dir
    
    def visualize_similarity(self, similarity_file: str, custom_subfolder: str = None) -> None:
        """
        Create visualizations for similarity analysis results
        
        Args:
            similarity_file: Path to similarity CSV file
            custom_subfolder: Optional custom subfolder name (auto-generated if None)
        """
        logger.info(f"Loading similarity results from: {similarity_file}")
        
        # Load similarity data
        similarity_df = pd.read_csv(similarity_file)
        logger.info(f"Loaded {len(similarity_df)} similarity comparisons")
        
        # Use the directory of the input file as output directory
        input_path = Path(similarity_file)
        
        # Update output directory to the same directory as input file
        original_output_dir = self.output_dir
        self.output_dir = input_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating visualizations in: {self.output_dir}")
        
        # Extract unique proposal IDs
        sim_proposal_ids = set()
        for _, row in similarity_df.iterrows():
            sim_proposal_ids.add(row['proposal_1_id'])
            sim_proposal_ids.add(row['proposal_2_id'])
        
        sim_proposal_ids = sorted(list(sim_proposal_ids))
        n_sim_proposals = len(sim_proposal_ids)
        logger.info(f"Number of unique proposals: {n_sim_proposals}")
        
        # Create index mapping
        sim_id_to_idx = {pid: idx for idx, pid in enumerate(sim_proposal_ids)}
        
        # Define similarity metrics
        similarity_metrics = [
            'tfidf_similarity',
            'embedding_similarity',
            'keyword_jaccard',
            'topic_similarity'
        ]
        
        metric_names = {
            'tfidf_similarity': 'TF-IDF Cosine Similarity',
            'embedding_similarity': 'Embedding Similarity (OpenAI)',
            'keyword_jaccard': 'Keyword Jaccard Similarity',
            'topic_similarity': 'Topic Similarity (LDA)'
        }
        
        logger.info(f"Similarity metrics: {list(metric_names.values())}")
        
        # Initialize matrices for each similarity metric
        sim_matrices = {}
        for metric in similarity_metrics:
            sim_matrices[metric] = np.full((n_sim_proposals, n_sim_proposals), np.nan)
        
        # Fill matrices with similarity scores
        for _, row in similarity_df.iterrows():
            prop_1_id = row['proposal_1_id']
            prop_2_id = row['proposal_2_id']
            
            idx_1 = sim_id_to_idx[prop_1_id]
            idx_2 = sim_id_to_idx[prop_2_id]
            
            for metric in similarity_metrics:
                score = row[metric]
                sim_matrices[metric][idx_1, idx_2] = score
        
        logger.info("Similarity matrices created")
        
        # Create combined similarity heatmap
        self._create_combined_similarity_heatmaps(
            sim_matrices, similarity_metrics, metric_names, sim_proposal_ids,
            'similarity_matrices_combined.png'
        )
        
        # Create individual detailed heatmaps
        self._create_individual_similarity_heatmaps(
            sim_matrices, similarity_metrics, metric_names, sim_proposal_ids
        )
        
        # Create correlation heatmap
        self._create_correlation_heatmap(
            sim_matrices, similarity_metrics, metric_names
        )
        
        # Create distribution plots
        self._create_distribution_plots(
            sim_matrices, similarity_metrics, metric_names
        )
        
        # Generate statistics
        self._generate_similarity_statistics(
            sim_matrices, similarity_metrics, metric_names, sim_proposal_ids
        )
        
        logger.info("✓ Similarity visualizations complete!")
        
        # Restore original output directory
        self.output_dir = original_output_dir
    
    def _create_combined_heatmaps(self, matrices: Dict, dimension_names: List[str],
                                  proposal_ids: List[str], filename: str,
                                  eval_type: str) -> None:
        """Create combined heatmap with all dimensions"""
        n_dims = len(dimension_names)
        n_cols = 2
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12 * n_rows // 2))
        if n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if n_cols == 1 else axes.flatten()
        
        for idx, dim_name in enumerate(dimension_names):
            ax = axes[idx]
            matrix = matrices[dim_name]
            
            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=4,
                       cbar_kws={'label': 'Overlap Score'}, xticklabels=proposal_ids,
                       yticklabels=proposal_ids, ax=ax, linewidths=0.5, square=True)
            
            ax.set_title(f'{dim_name}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Proposal 2', fontsize=12)
            ax.set_ylabel('Proposal 1', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Hide extra subplots
        for idx in range(len(dimension_names), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Pairwise Evaluation Overlap Scores\n({eval_type})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved combined heatmap: {output_path}")
    
    def _create_individual_heatmaps(self, matrices: Dict, dimension_names: List[str],
                                   proposal_ids: List[str], prefix: str,
                                   eval_type: str) -> None:
        """Create individual detailed heatmaps for each dimension"""
        for dim_name in dimension_names:
            fig, ax = plt.subplots(figsize=(14, 12))
            matrix = matrices[dim_name]
            
            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=4,
                       cbar_kws={'label': 'Overlap Score (0-4)'}, xticklabels=proposal_ids,
                       yticklabels=proposal_ids, ax=ax, linewidths=1, linecolor='white',
                       square=True, annot_kws={'size': 11, 'weight': 'bold'})
            
            ax.set_title(f'{dim_name}\nPairwise Overlap Scores ({eval_type})',
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Proposal 2 (Compared Against)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Proposal 1 (Reference)', fontsize=13, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            plt.tight_layout()
            
            safe_name = dim_name.replace('/', '_').replace(' ', '_').lower()
            output_path = self.output_dir / f"{prefix}{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Saved: {output_path.name}")
    
    def _create_combined_similarity_heatmaps(self, sim_matrices: Dict,
                                            similarity_metrics: List[str],
                                            metric_names: Dict[str, str],
                                            proposal_ids: List[str],
                                            filename: str) -> None:
        """Create combined heatmap for all similarity metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        axes = axes.flatten()
        
        for idx, metric in enumerate(similarity_metrics):
            ax = axes[idx]
            matrix = sim_matrices[metric]
            
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1,
                       cbar_kws={'label': 'Similarity Score (0-1)'}, xticklabels=proposal_ids,
                       yticklabels=proposal_ids, ax=ax, linewidths=0.5, square=True,
                       annot_kws={'size': 8})
            
            ax.set_title(metric_names[metric], fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Proposal 2', fontsize=11)
            ax.set_ylabel('Proposal 1', fontsize=11)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.suptitle('Pairwise Similarity Scores (All Metrics)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved combined similarity heatmap: {output_path}")
    
    def _create_individual_similarity_heatmaps(self, sim_matrices: Dict,
                                              similarity_metrics: List[str],
                                              metric_names: Dict[str, str],
                                              proposal_ids: List[str]) -> None:
        """Create individual detailed heatmaps for each similarity metric"""
        for metric in similarity_metrics:
            fig, ax = plt.subplots(figsize=(14, 12))
            matrix = sim_matrices[metric]
            
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                       cbar_kws={'label': 'Similarity Score (0-1)'}, xticklabels=proposal_ids,
                       yticklabels=proposal_ids, ax=ax, linewidths=1, linecolor='white',
                       square=True, annot_kws={'size': 10, 'weight': 'bold'})
            
            ax.set_title(f'{metric_names[metric]}\nPairwise Similarity Scores',
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Proposal 2 (Compared Against)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Proposal 1 (Reference)', fontsize=13, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            plt.tight_layout()
            
            safe_name = metric.replace('_', '_')
            output_path = self.output_dir / f"matrix_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Saved: {output_path.name}")
    
    def _create_correlation_heatmap(self, sim_matrices: Dict,
                                   similarity_metrics: List[str],
                                   metric_names: Dict[str, str]) -> None:
        """Create correlation heatmap between similarity metrics"""
        # Flatten all matrices and compute correlation
        correlation_data = {}
        for metric in similarity_metrics:
            matrix = sim_matrices[metric]
            correlation_data[metric] = matrix.flatten()
        
        # Create correlation dataframe
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        # Visualize correlation
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True, linewidths=1,
                   xticklabels=[metric_names[m] for m in similarity_metrics],
                   yticklabels=[metric_names[m] for m in similarity_metrics],
                   cbar_kws={'label': 'Correlation Coefficient'},
                   annot_kws={'size': 12, 'weight': 'bold'})
        
        ax.set_title('Correlation Between Similarity Metrics',
                    fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = self.output_dir / 'metrics_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved correlation heatmap: {output_path}")
    
    def _create_distribution_plots(self, sim_matrices: Dict,
                                  similarity_metrics: List[str],
                                  metric_names: Dict[str, str]) -> None:
        """Create distribution plots for each similarity metric"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(similarity_metrics):
            ax = axes[idx]
            matrix = sim_matrices[metric]
            valid_scores = matrix[~np.isnan(matrix)]
            
            # Create histogram
            ax.hist(valid_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(np.mean(valid_scores), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(valid_scores):.3f}')
            ax.axvline(np.median(valid_scores), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(valid_scores):.3f}')
            
            ax.set_title(metric_names[metric], fontsize=13, fontweight='bold')
            ax.set_xlabel('Similarity Score', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Similarity Score Distributions',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'similarity_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved distribution plots: {output_path}")
    
    def _generate_evaluation_statistics(self, matrices: Dict,
                                       dimension_names: List[str],
                                       eval_type: str) -> None:
        """Generate and save evaluation statistics"""
        stats_lines = []
        stats_lines.append("=" * 70)
        stats_lines.append(f"EVALUATION SUMMARY STATISTICS")
        stats_lines.append(f"Evaluation Type: {eval_type}")
        stats_lines.append("=" * 70)
        
        for dim_name in dimension_names:
            matrix = matrices[dim_name]
            valid_scores = matrix[~np.isnan(matrix)]
            
            stats_lines.append(f"\n{dim_name}:")
            stats_lines.append(f"  Mean score: {np.mean(valid_scores):.2f}")
            stats_lines.append(f"  Median score: {np.median(valid_scores):.2f}")
            stats_lines.append(f"  Std dev: {np.std(valid_scores):.2f}")
            stats_lines.append(f"  Min: {np.min(valid_scores):.0f}, Max: {np.max(valid_scores):.0f}")
            stats_lines.append(f"  Score distribution:")
            
            for score in range(5):
                count = np.sum(valid_scores == score)
                pct = count / len(valid_scores) * 100
                stats_lines.append(f"    Score {score}: {count:3d} ({pct:5.1f}%)")
        
        # Save statistics to file
        output_path = self.output_dir / 'evaluation_statistics.txt'
        with open(output_path, 'w') as f:
            f.write('\n'.join(stats_lines))
        
        logger.info(f"✓ Saved statistics: {output_path}")
        
        # Also print to console
        print('\n'.join(stats_lines))
    
    def _generate_similarity_statistics(self, sim_matrices: Dict,
                                       similarity_metrics: List[str],
                                       metric_names: Dict[str, str],
                                       proposal_ids: List[str]) -> None:
        """Generate and save similarity statistics"""
        stats_lines = []
        stats_lines.append("=" * 70)
        stats_lines.append("SIMILARITY METRICS SUMMARY STATISTICS")
        stats_lines.append("=" * 70)
        
        for metric in similarity_metrics:
            matrix = sim_matrices[metric]
            valid_scores = matrix[~np.isnan(matrix)]
            
            stats_lines.append(f"\n{metric_names[metric]}:")
            stats_lines.append(f"  Mean: {np.mean(valid_scores):.4f}")
            stats_lines.append(f"  Median: {np.median(valid_scores):.4f}")
            stats_lines.append(f"  Std dev: {np.std(valid_scores):.4f}")
            stats_lines.append(f"  Min: {np.min(valid_scores):.4f}, Max: {np.max(valid_scores):.4f}")
            stats_lines.append(f"  Range: {np.max(valid_scores) - np.min(valid_scores):.4f}")
            
            # Percentiles
            p25, p50, p75 = np.percentile(valid_scores, [25, 50, 75])
            stats_lines.append(f"  Percentiles: 25th={p25:.4f}, 50th={p50:.4f}, 75th={p75:.4f}")
        
        # High similarity pairs
        stats_lines.append("\n" + "=" * 70)
        stats_lines.append("HIGH SIMILARITY PAIRS (Top 10% for each metric)")
        stats_lines.append("=" * 70)
        
        for metric in similarity_metrics:
            matrix = sim_matrices[metric]
            valid_scores = matrix[~np.isnan(matrix)]
            threshold = np.percentile(valid_scores, 90)  # Top 10%
            
            high_sim_indices = np.argwhere(matrix >= threshold)
            
            stats_lines.append(f"\n{metric_names[metric]} (threshold >= {threshold:.4f}):")
            
            # Sort by score (descending)
            pairs_with_scores = []
            for idx1, idx2 in high_sim_indices:
                prop1 = proposal_ids[idx1]
                prop2 = proposal_ids[idx2]
                score = matrix[idx1, idx2]
                pairs_with_scores.append((prop1, prop2, score))
            
            pairs_with_scores.sort(key=lambda x: x[2], reverse=True)
            
            for prop1, prop2, score in pairs_with_scores[:10]:  # Show top 10
                stats_lines.append(f"  {prop1} vs {prop2}: {score:.4f}")
        
        # Save statistics to file
        output_path = self.output_dir / 'similarity_statistics.txt'
        with open(output_path, 'w') as f:
            f.write('\n'.join(stats_lines))
        
        logger.info(f"✓ Saved statistics: {output_path}")
        
        # Also print to console
        print('\n'.join(stats_lines))
    
    def _generate_eval_subfolder_name(self, evaluations: List[Dict[str, Any]], eval_type: str) -> str:
        """
        Generate a descriptive subfolder name based on evaluation metadata
        
        Args:
            evaluations: List of evaluation results
            eval_type: Evaluation type from metadata
        
        Returns:
            Subfolder name (e.g., "pairwise_ai-ai_genby_gemini_eval_gemini")
        """
        if not evaluations:
            return eval_type
        
        # Extract metadata from first evaluation
        first_eval = evaluations[0]
        
        # Get comparison type from eval_type (e.g., "pairwise_human-human" -> "human-human")
        comparison_type = eval_type.replace('pairwise_', '').replace('single_', '')
        
        # Build subfolder name parts
        parts = [comparison_type]
        
        # Try to infer role and model from proposal IDs
        prop_1_id = first_eval.get('proposal_1_id', '')
        
        # Check if it's AI proposals
        if prop_1_id.startswith('ai_'):
            # Extract role and model from ID (e.g., "ai_generate_ideas_no_role_gemini_01")
            id_parts = prop_1_id.split('_')
            if len(id_parts) >= 3:
                # Role could be multiple parts (e.g., "generate_ideas_no_role")
                # Model is typically second to last part
                if len(id_parts) == 4:  # ai_role_model_num
                    role = id_parts[1]
                    model = id_parts[2]
                elif len(id_parts) == 5:  # ai_role1_role2_model_num
                    role = f"{id_parts[1]}_{id_parts[2]}"
                    model = id_parts[3]
                elif len(id_parts) >= 6:  # ai_role1_role2_role3_model_num
                    role = "_".join(id_parts[1:-2])
                    model = id_parts[-2]
                else:
                    role = id_parts[1] if len(id_parts) > 1 else None
                    model = id_parts[-2] if len(id_parts) > 2 else None
                
                if role and role not in ['single', 'group', 'group_int']:
                    parts.append(role)
                if model:
                    parts.append(f"genby_{model}")
        
        # Get evaluator model from the evaluation
        evaluator = first_eval.get('evaluator_model', '')
        if evaluator:
            eval_short = evaluator.split('-')[0]
            parts.append(f"eval_{eval_short}")
        
        return "_".join(parts)
    
    def _generate_sim_subfolder_name(self, similarity_file: str) -> str:
        """
        Generate a descriptive subfolder name based on similarity filename
        
        Args:
            similarity_file: Path to similarity CSV file
        
        Returns:
            Subfolder name (e.g., "similarity_human-human")
        """
        # Extract filename without extension
        filename = Path(similarity_file).stem
        
        # similarity_human-human_20251030_160213 -> similarity_human-human
        # Remove timestamp part
        parts = filename.split('_')
        if len(parts) >= 2:
            # Keep parts until we hit a timestamp-like part (YYYYMMDD)
            kept_parts = []
            for part in parts:
                if part.isdigit() and len(part) == 8:  # Likely a date
                    break
                kept_parts.append(part)
            return "_".join(kept_parts) if kept_parts else filename
        
        return filename


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Visualize evaluation and similarity analysis results"
    )
    parser.add_argument(
        "--evaluation-file",
        type=str,
        help="Path to evaluation JSON file"
    )
    parser.add_argument(
        "--similarity-file",
        type=str,
        help="Path to similarity CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visuals",
        help="Output directory for visualizations (default: visuals)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process both evaluation and similarity files if both are provided"
    )
    
    args = parser.parse_args()
    
    # Check if at least one file is provided
    if not args.evaluation_file and not args.similarity_file:
        parser.error("At least one of --evaluation-file or --similarity-file must be provided")
    
    # Initialize visualizer
    visualizer = ResultVisualizer(output_dir=args.output_dir)
    
    # Process evaluation file
    if args.evaluation_file:
        eval_path = Path(args.evaluation_file)
        if not eval_path.exists():
            logger.error(f"Evaluation file not found: {eval_path}")
        else:
            logger.info(f"\n{'='*70}")
            logger.info("PROCESSING EVALUATION RESULTS")
            logger.info(f"{'='*70}\n")
            visualizer.visualize_evaluations(str(eval_path))
    
    # Process similarity file
    if args.similarity_file:
        sim_path = Path(args.similarity_file)
        if not sim_path.exists():
            logger.error(f"Similarity file not found: {sim_path}")
        else:
            logger.info(f"\n{'='*70}")
            logger.info("PROCESSING SIMILARITY RESULTS")
            logger.info(f"{'='*70}\n")
            visualizer.visualize_similarity(str(sim_path))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ ALL VISUALIZATIONS COMPLETE!")
    logger.info(f"✓ Output directory: {visualizer.output_dir.absolute()}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()

