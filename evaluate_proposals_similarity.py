#!/usr/bin/env python3
"""
Evaluate textual similarity and topic overlap between research proposals.
Uses embeddings, cosine similarity, and topic modeling to analyze proposal pairs.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import itertools

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Container for similarity analysis results"""
    proposal_1_id: str
    proposal_2_id: str
    proposal_1_title: str
    proposal_2_title: str
    proposal_1_who: str
    proposal_2_who: str
    tfidf_similarity: float
    embedding_similarity: float
    keyword_overlap: List[str]
    topic_overlap: Dict[str, Any]
    timestamp: str


class ProposalSimilarityAnalyzer:
    """Analyze textual similarity and topic overlap between proposals"""
    
    def __init__(self, config_path: str = "config.env", proposals_csv: str = "all_proposals_combined.csv"):
        """Initialize the similarity analyzer"""
        self.proposals_csv = proposals_csv
        self.proposals_df = None
        self._load_proposals_from_csv()
        
        self.results_dir = Path("similarity_analysis")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Load OpenAI API key for embeddings
        self._load_openai_config(config_path)
        
        logger.info("ProposalSimilarityAnalyzer initialized")
    
    def _load_openai_config(self, config_path: str):
        """Load OpenAI API key from config file"""
        import os
        self.openai_key = os.getenv('OPENAI_API_KEY')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        value = value.strip().strip('"').strip("'")
                        if key == 'OPENAI_API_KEY' and not self.openai_key:
                            self.openai_key = value
        
        if self.openai_key:
            openai.api_key = self.openai_key
            logger.info("OpenAI API key loaded for embeddings")
        else:
            logger.warning("OpenAI API key not found. Embedding similarity will be skipped.")
    
    def _load_proposals_from_csv(self):
        """Load all proposals from the combined CSV file"""
        try:
            self.proposals_df = pd.read_csv(self.proposals_csv)
            logger.info(f"Loaded {len(self.proposals_df)} proposals from {self.proposals_csv}")
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
    
    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF based cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Fit and transform the texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing TF-IDF similarity: {e}")
            return 0.0
    
    def get_openai_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get OpenAI embedding for a text
        
        Args:
            text: Text to embed
            model: OpenAI embedding model
        
        Returns:
            Embedding vector
        """
        try:
            client = openai.OpenAI(api_key=self.openai_key)
            
            # Truncate text if too long (max 8191 tokens for embedding models)
            max_length = 30000  # Approximate character limit
            if len(text) > max_length:
                text = text[:max_length]
            
            response = client.embeddings.create(
                input=text,
                model=model
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return None
    
    def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Compute embedding-based cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score (0-1), or None if embeddings unavailable
        """
        if not self.openai_key:
            return None
        
        try:
            # Get embeddings
            emb1 = self.get_openai_embedding(text1)
            emb2 = self.get_openai_embedding(text2)
            
            if emb1 is None or emb2 is None:
                return None
            
            # Compute cosine similarity
            emb1 = np.array(emb1).reshape(1, -1)
            emb2 = np.array(emb2).reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing embedding similarity: {e}")
            return None
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Extract top keywords from text using TF-IDF
        
        Args:
            text: Text to analyze
            top_n: Number of top keywords to return
        
        Returns:
            List of (keyword, score) tuples
        """
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get scores for this document
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def compute_keyword_overlap(self, text1: str, text2: str, top_n: int = 20) -> Dict[str, Any]:
        """
        Compute keyword overlap between two texts
        
        Args:
            text1: First text
            text2: Second text
            top_n: Number of top keywords to consider
        
        Returns:
            Dictionary with overlap statistics
        """
        keywords1 = self.extract_keywords(text1, top_n)
        keywords2 = self.extract_keywords(text2, top_n)
        
        # Extract just the keywords (not scores)
        kw_set1 = set([kw for kw, _ in keywords1])
        kw_set2 = set([kw for kw, _ in keywords2])
        
        # Compute overlap
        overlap = kw_set1.intersection(kw_set2)
        jaccard_similarity = len(overlap) / len(kw_set1.union(kw_set2)) if len(kw_set1.union(kw_set2)) > 0 else 0
        
        return {
            'overlapping_keywords': sorted(list(overlap)),
            'num_overlapping': len(overlap),
            'jaccard_similarity': jaccard_similarity,
            'proposal_1_unique': sorted(list(kw_set1 - kw_set2))[:10],
            'proposal_2_unique': sorted(list(kw_set2 - kw_set1))[:10]
        }
    
    def perform_topic_modeling(self, texts: List[str], n_topics: int = 10) -> Tuple[Any, Any]:
        """
        Perform LDA topic modeling on a collection of texts
        
        Args:
            texts: List of texts to analyze
            n_topics: Number of topics to extract
        
        Returns:
            (lda_model, tfidf_matrix)
        """
        try:
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Fit LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            lda.fit(tfidf_matrix)
            
            return lda, tfidf_matrix
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return None, None
    
    def compute_topic_overlap(self, text1: str, text2: str, n_topics: int = 5) -> Dict[str, Any]:
        """
        Compute topic overlap between two texts using LDA
        
        Args:
            text1: First text
            text2: Second text
            n_topics: Number of topics to extract
        
        Returns:
            Dictionary with topic overlap information
        """
        try:
            # Perform topic modeling on both texts
            lda, tfidf_matrix = self.perform_topic_modeling([text1, text2], n_topics)
            
            if lda is None:
                return {'error': 'Topic modeling failed'}
            
            # Get topic distributions for each text
            topic_dist = lda.transform(tfidf_matrix)
            
            # Compute cosine similarity between topic distributions
            topic_similarity = cosine_similarity(topic_dist[0:1], topic_dist[1:2])[0][0]
            
            # Get top words for each topic
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'top_words': top_words
                })
            
            return {
                'topic_similarity': float(topic_similarity),
                'proposal_1_topic_distribution': topic_dist[0].tolist(),
                'proposal_2_topic_distribution': topic_dist[1].tolist(),
                'topics': topics
            }
        except Exception as e:
            logger.error(f"Error computing topic overlap: {e}")
            return {'error': str(e)}
    
    def analyze_proposal_pair(self, 
                             proposal_1: Dict[str, Any],
                             proposal_2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive similarity analysis on a pair of proposals
        
        Args:
            proposal_1: First proposal
            proposal_2: Second proposal
        
        Returns:
            Dictionary with all similarity metrics
        """
        proposal_1_id = proposal_1.get('proposal_id', 'unknown')
        proposal_2_id = proposal_2.get('proposal_id', 'unknown')
        proposal_1_title = proposal_1.get('title', 'N/A')
        proposal_2_title = proposal_2.get('title', 'N/A')
        
        logger.info(f"Analyzing similarity: '{proposal_1_title}' vs '{proposal_2_title}'")
        
        # Get full texts
        text1 = proposal_1.get('full_draft', '')
        text2 = proposal_2.get('full_draft', '')
        
        # Skip if either text is empty
        if not text1 or not text2:
            logger.warning(f"Skipping pair {proposal_1_id} vs {proposal_2_id}: Empty text")
            return {
                'proposal_1_id': proposal_1_id,
                'proposal_2_id': proposal_2_id,
                'error': 'Empty text in one or both proposals'
            }
        
        # Compute TF-IDF similarity
        tfidf_sim = self.compute_tfidf_similarity(text1, text2)
        logger.info(f"  TF-IDF similarity: {tfidf_sim:.4f}")
        
        # Compute embedding similarity
        embedding_sim = self.compute_embedding_similarity(text1, text2)
        if embedding_sim is not None:
            logger.info(f"  Embedding similarity: {embedding_sim:.4f}")
        
        # Compute keyword overlap
        keyword_overlap = self.compute_keyword_overlap(text1, text2)
        logger.info(f"  Keyword overlap: {keyword_overlap['num_overlapping']} keywords")
        
        # Compute topic overlap
        topic_overlap = self.compute_topic_overlap(text1, text2)
        if 'topic_similarity' in topic_overlap:
            logger.info(f"  Topic similarity: {topic_overlap['topic_similarity']:.4f}")
        
        # Create result dictionary
        result = {
            'proposal_1_id': proposal_1_id,
            'proposal_2_id': proposal_2_id,
            'proposal_1_title': proposal_1_title,
            'proposal_2_title': proposal_2_title,
            'proposal_1_who': proposal_1.get('who', 'unknown'),
            'proposal_2_who': proposal_2.get('who', 'unknown'),
            'proposal_1_role': proposal_1.get('role', 'unknown'),
            'proposal_2_role': proposal_2.get('role', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'similarity_metrics': {
                'tfidf_cosine_similarity': tfidf_sim,
                'embedding_cosine_similarity': embedding_sim,
                'keyword_overlap': keyword_overlap,
                'topic_analysis': topic_overlap
            }
        }
        
        return result
    
    def analyze_all_pairs(self,
                         proposals_1: List[Dict[str, Any]],
                         proposals_2: List[Dict[str, Any]],
                         comparison_type: str = "human-ai") -> List[Dict[str, Any]]:
        """
        Analyze similarity for all pairs of proposals
        
        Args:
            proposals_1: First set of proposals
            proposals_2: Second set of proposals
            comparison_type: Type of comparison ('human-human', 'human-ai', 'ai-ai')
        
        Returns:
            List of all similarity analysis results
        """
        logger.info(f"Analyzing {len(proposals_1)} x {len(proposals_2)} proposal pairs")
        logger.info(f"Comparison type: {comparison_type}")
        
        # Generate all combinations with indices for tracking
        all_pairs = list(itertools.product(enumerate(proposals_1), enumerate(proposals_2)))
        
        # Count non-self comparisons
        total_analyses = sum(
            1 for (_, p1), (_, p2) in all_pairs 
            if p1.get('proposal_id') != p2.get('proposal_id')
        )
        
        logger.info(f"Total analyses (excluding self-comparisons): {total_analyses}")
        
        all_results = []
        current = 0
        
        for (i1, prop_1), (i2, prop_2) in all_pairs:
            # Skip self-comparisons
            if prop_1.get('proposal_id') == prop_2.get('proposal_id'):
                continue
            
            current += 1
            logger.info(f"Processing analysis {current}/{total_analyses}")
            
            try:
                result = self.analyze_proposal_pair(prop_1, prop_2)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing pair {current}/{total_analyses}: {e}")
                error_result = {
                    'proposal_1_id': prop_1.get('proposal_id', 'unknown'),
                    'proposal_2_id': prop_2.get('proposal_id', 'unknown'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(error_result)
        
        logger.info(f"Completed {len(all_results)} similarity analyses")
        return all_results
    
    def save_results(self, 
                    results: List[Dict[str, Any]],
                    comparison_type: str,
                    output_filename: str = None):
        """Save similarity analysis results to a JSON file"""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"similarity_{comparison_type}_{timestamp}.json"
        
        output_path = self.results_dir / output_filename
        
        # Create results dict
        results_dict = {
            "metadata": {
                "comparison_type": comparison_type,
                "total_analyses": len(results),
                "generation_timestamp": datetime.now().isoformat(),
                "methods": {
                    "tfidf": "TF-IDF based cosine similarity",
                    "embeddings": "OpenAI text-embedding-3-small",
                    "keywords": "TF-IDF top-20 keywords with Jaccard similarity",
                    "topics": "Latent Dirichlet Allocation (LDA) with 5 topics"
                }
            },
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} similarity analyses to {output_path}")
        
        # Also save as CSV for easy viewing
        csv_filename = output_filename.replace('.json', '.csv')
        csv_path = self.results_dir / csv_filename
        self._save_results_csv(results, csv_path)
    
    def _save_results_csv(self, results: List[Dict[str, Any]], csv_path: Path):
        """Save simplified results to CSV"""
        csv_data = []
        for result in results:
            if 'error' in result:
                continue
            
            metrics = result.get('similarity_metrics', {})
            csv_data.append({
                'proposal_1_id': result.get('proposal_1_id'),
                'proposal_2_id': result.get('proposal_2_id'),
                'proposal_1_title': result.get('proposal_1_title'),
                'proposal_2_title': result.get('proposal_2_title'),
                'proposal_1_who': result.get('proposal_1_who'),
                'proposal_2_who': result.get('proposal_2_who'),
                'tfidf_similarity': metrics.get('tfidf_cosine_similarity'),
                'embedding_similarity': metrics.get('embedding_cosine_similarity'),
                'keyword_overlap_count': metrics.get('keyword_overlap', {}).get('num_overlapping'),
                'keyword_jaccard': metrics.get('keyword_overlap', {}).get('jaccard_similarity'),
                'topic_similarity': metrics.get('topic_analysis', {}).get('topic_similarity')
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved simplified CSV results to {csv_path}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze textual similarity and topic overlap between research proposals"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="all_proposals_combined.csv",
        help="Path to combined proposals CSV file"
    )
    parser.add_argument(
        "--compare-type",
        type=str,
        choices=["human-human", "ai-ai", "human-ai"],
        default="human-ai",
        help="Type of comparison"
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help="Specific AI model name to filter proposals"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="single_scientist",
        help="Filter AI proposals by template/role"
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=None,
        help="Maximum number of proposals to analyze per group"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for results"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ProposalSimilarityAnalyzer(proposals_csv=args.csv)
    
    # Load proposals based on comparison type
    logger.info(f"Starting similarity analysis: {args.compare_type}")
    
    if args.compare_type == "human-human":
        proposals_1 = analyzer.load_proposals(who="human", max_proposals=args.max_proposals)
        proposals_2 = analyzer.load_proposals(who="human", max_proposals=args.max_proposals)
    elif args.compare_type == "ai-ai":
        proposals_1 = analyzer.load_proposals(
            who="ai",
            role=args.template if args.template != "single_scientist" else None,
            model=args.ai_model,
            max_proposals=args.max_proposals
        )
        proposals_2 = analyzer.load_proposals(
            who="ai",
            role=args.template if args.template != "single_scientist" else None,
            model=args.ai_model,
            max_proposals=args.max_proposals
        )
    else:  # human-ai
        proposals_1 = analyzer.load_proposals(who="human", max_proposals=args.max_proposals)
        proposals_2 = analyzer.load_proposals(
            who="ai",
            role=args.template if args.template != "single_scientist" else None,
            model=args.ai_model,
            max_proposals=args.max_proposals
        )
    
    # Perform similarity analysis
    all_results = analyzer.analyze_all_pairs(
        proposals_1=proposals_1,
        proposals_2=proposals_2,
        comparison_type=args.compare_type
    )
    
    # Save results
    analyzer.save_results(
        results=all_results,
        comparison_type=args.compare_type,
        output_filename=args.output
    )
    
    logger.info(f"Similarity analysis complete!")


if __name__ == "__main__":
    main()

