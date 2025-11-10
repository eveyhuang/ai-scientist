#!/usr/bin/env python3
"""
Evaluate textual similarity and topic overlap between research proposals.
Uses local embeddings (Nomic-Embed), cosine similarity, and topic modeling to analyze proposal pairs.

Features:
- Local embedding generation using Nomic-Embed (supports 8192 tokens, runs locally)
- TF-IDF similarity analysis
- Keyword and topic overlap detection (LDA)
- Caching for efficient repeated analysis
"""

import json
import logging
import os
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
from sentence_transformers import SentenceTransformer

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
    
    def __init__(self, config_path: str = "config.env", proposals_csv: str = "all_proposals_combined_no_role.csv"):
        """Initialize the similarity analyzer"""
        self.proposals_csv = proposals_csv
        self.proposals_df = None
        self._load_proposals_from_csv()
        
        self.results_dir = Path("semantic_similarity")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Load Nomic-Embed model for local embeddings
        logger.info("Loading Nomic-Embed model (this may take a moment on first run)...")
        self.embedding_model = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True
        )
        logger.info("Nomic-Embed model loaded successfully")
        
        # Cache for embeddings to avoid redundant computations
        self.embedding_cache = {}
        
        logger.info("ProposalSimilarityAnalyzer initialized")
    
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
                      max_proposals: int = None,
                      proposal_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load proposals from the combined CSV with optional filters
        
        Args:
            who: Filter by source ('human' or 'ai')
            role: Filter by role ('human', 'single', 'group', 'group_int')
            model: Filter by model name
            max_proposals: Maximum number of proposals to return
            proposal_ids: Filter by specific proposal IDs
        
        Returns:
            List of proposal dictionaries
        """
        if self.proposals_df is None or len(self.proposals_df) == 0:
            logger.warning("No proposals loaded")
            return []
        
        df = self.proposals_df.copy()
        
        # Apply filters
        if proposal_ids:
            logger.info(f"Filtering by {len(proposal_ids)} specific proposal IDs")
            df = df[df['proposal_id'].isin(proposal_ids)]
            logger.info(f"After filtering by IDs: {len(df)} proposals found")
            
            # Debug: show which IDs were found and which weren't
            found_ids = set(df['proposal_id'].tolist())
            requested_ids = set(proposal_ids)
            missing = requested_ids - found_ids
            if missing:
                logger.warning(f"Could not find these {len(missing)} proposal IDs in CSV: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        else:
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
        
        if proposal_ids:
            logger.info(f"Loaded {len(proposals)} proposals from specified IDs")
        else:
            logger.info(f"Loaded {len(proposals)} proposals with filters: who={who}, role={role}, model={model}")
        
        return proposals
    
    def load_proposal_ids_from_json(self, json_file: str) -> List[str]:
        """
        Load proposal IDs from a JSON file
        
        Args:
            json_file: Path to JSON file containing proposal IDs
        
        Returns:
            List of proposal IDs
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Try different possible field names
            if 'selected_proposals' in data:
                proposal_ids = data['selected_proposals']
            elif 'proposal_ids' in data:
                proposal_ids = data['proposal_ids']
            elif 'proposals' in data:
                proposal_ids = data['proposals']
            elif isinstance(data, list):
                proposal_ids = data
            else:
                logger.error(f"Could not find proposal IDs in JSON file. Expected 'selected_proposals', 'proposal_ids', or 'proposals' field")
                return []
            
            logger.info(f"Loaded {len(proposal_ids)} proposal IDs from {json_file}")
            return proposal_ids
        
        except Exception as e:
            logger.error(f"Error loading proposal IDs from JSON: {e}")
            return []
    
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
    
    def get_embedding(self, text: str, proposal_id: str = None) -> List[float]:
        """
        Get local embedding for a text using Nomic-Embed (with caching)
        
        Args:
            text: Text to embed
            proposal_id: Unique ID for caching
        
        Returns:
            Embedding vector
        """
        # Check cache first
        if proposal_id and proposal_id in self.embedding_cache:
            logger.debug(f"Using cached embedding for {proposal_id}")
            return self.embedding_cache[proposal_id]
        
        try:
            # Nomic-Embed supports up to 8192 tokens (~6000 words, ~30000 chars)
            max_length = 30000
            if len(text) > max_length:
                logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
                text = text[:max_length]
            
            # Add task prefix for Nomic-Embed (improves performance)
            prefixed_text = f"search_document: {text}"
            
            # Generate embedding (normalized by default with Nomic-Embed)
            embedding = self.embedding_model.encode(
                prefixed_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Convert to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Cache the result
            if proposal_id:
                self.embedding_cache[proposal_id] = embedding_list
            
            return embedding_list
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def compute_embedding_similarity(self, text1: str, text2: str, proposal_1_id: str = None, proposal_2_id: str = None) -> float:
        """
        Compute embedding-based cosine similarity between two texts using local embeddings
        
        Args:
            text1: First text
            text2: Second text
            proposal_1_id: ID for caching
            proposal_2_id: ID for caching
        
        Returns:
            Cosine similarity score (0-1), or None if embeddings unavailable
        """
        try:
            # Get embeddings (with caching)
            emb1 = self.get_embedding(text1, proposal_1_id)
            emb2 = self.get_embedding(text2, proposal_2_id)
            
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
        
        NOTE: To ensure symmetry, we always use consistent ordering
        (alphabetically sorted texts) so that compare(A,B) == compare(B,A)
        
        Args:
            text1: First text
            text2: Second text
            n_topics: Number of topics to extract
        
        Returns:
            Dictionary with topic overlap information
        """
        try:
            # Sort texts to ensure consistent ordering for symmetric results
            # This ensures that (text1, text2) and (text2, text1) produce the same topics
            texts_sorted = sorted([text1, text2], key=lambda x: hash(x))
            text1_is_first = (texts_sorted[0] == text1)
            
            # Perform topic modeling on both texts (in consistent order)
            lda, tfidf_matrix = self.perform_topic_modeling(texts_sorted, n_topics)
            
            if lda is None:
                return {'error': 'Topic modeling failed'}
            
            # Get topic distributions for each text
            topic_dist = lda.transform(tfidf_matrix)
            
            # Extract distributions based on original order
            if text1_is_first:
                dist1, dist2 = topic_dist[0], topic_dist[1]
            else:
                dist1, dist2 = topic_dist[1], topic_dist[0]
            
            # Compute cosine similarity between topic distributions
            dist1_reshaped = dist1.reshape(1, -1)
            dist2_reshaped = dist2.reshape(1, -1)
            topic_similarity = cosine_similarity(dist1_reshaped, dist2_reshaped)[0][0]
            
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
                'proposal_1_topic_distribution': dist1.tolist(),
                'proposal_2_topic_distribution': dist2.tolist(),
                'topics': topics
            }
        except Exception as e:
            logger.error(f"Error computing topic overlap: {e}")
            return {'error': str(e)}
    
    def _clean_text(self, text: Any) -> str:
        """
        Clean text field, handling NaN and empty values
        
        Args:
            text: Text to clean (may be str, NaN, None, etc.)
        
        Returns:
            Cleaned string (empty string if invalid)
        """
        # Handle NaN values (pandas reads empty CSV cells as NaN)
        if pd.isna(text):
            return ''
        
        # Convert to string if not already
        if not isinstance(text, str):
            return str(text)
        
        # Return stripped text
        return text.strip()
    
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
        
        # Get full texts and clean them (handles NaN values)
        text1 = self._clean_text(proposal_1.get('full_draft', ''))
        text2 = self._clean_text(proposal_2.get('full_draft', ''))
        
        # Skip if either text is empty
        if not text1 or not text2:
            logger.warning(f"Skipping pair {proposal_1_id} vs {proposal_2_id}: Empty text (text1 len={len(text1)}, text2 len={len(text2)})")
            return {
                'proposal_1_id': proposal_1_id,
                'proposal_2_id': proposal_2_id,
                'error': 'Empty text in one or both proposals'
            }
        
        # Compute TF-IDF similarity
        tfidf_sim = self.compute_tfidf_similarity(text1, text2)
        logger.info(f"  TF-IDF similarity: {tfidf_sim:.4f}")
        
        # Compute embedding similarity (with caching)
        embedding_sim = self.compute_embedding_similarity(text1, text2, proposal_1_id, proposal_2_id)
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
        logger.info(f"Embedding cache size: {len(self.embedding_cache)} proposals")
        
        # Generate all combinations
        all_pairs = list(itertools.product(proposals_1, proposals_2))
        
        # Count non-self comparisons
        total_analyses = sum(
            1 for p1, p2 in all_pairs 
            if p1.get('proposal_id') != p2.get('proposal_id')
        )
        
        logger.info(f"Total analyses (excluding self-comparisons): {total_analyses}")
        
        all_results = []
        current = 0
        
        for prop_1, prop_2 in all_pairs:
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
        logger.info(f"Final embedding cache size: {len(self.embedding_cache)} proposals")
        return all_results
    
    def save_results(self, 
                    results: List[Dict[str, Any]],
                    comparison_type: str,
                    output_filename: str = None,
                    role_filter: str = None,
                    model_filter: str = None):
        """Save similarity analysis results to a JSON file in organized subfolders"""
        
        # Check if output_filename is a full path
        if output_filename and (os.path.sep in output_filename or os.path.altsep and os.path.altsep in output_filename):
            # It's a full path, use it directly
            output_path = Path(output_filename)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create subfolder name based on comparison parameters
            subfolder_parts = ["similarity", comparison_type]
            
            if role_filter:
                subfolder_parts.append(role_filter)
            if model_filter:
                model_short = model_filter.split('-')[0]
                subfolder_parts.append(f"genby_{model_short}")
            
            subfolder_name = "_".join(subfolder_parts)
            output_dir = self.results_dir / subfolder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Build descriptive filename
                filename_parts = ["similarity", comparison_type]
                
                if role_filter:
                    filename_parts.append(role_filter)
                if model_filter:
                    model_short = model_filter.split('-')[0]
                    filename_parts.append(f"genby_{model_short}")
                
                filename_parts.append(timestamp)
                output_filename = "_".join(filename_parts) + ".json"
            
            output_path = output_dir / output_filename
        
        # Create results dict with metadata including filters
        results_dict = {
            "metadata": {
                "comparison_type": comparison_type,
                "role_filter": role_filter,
                "model_filter": model_filter,
                "total_analyses": len(results),
                "generation_timestamp": datetime.now().isoformat(),
                "methods": {
                    "tfidf": "TF-IDF based cosine similarity",
                    "embeddings": "Nomic-Embed",
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
        csv_path = output_path.with_suffix('.csv')
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
        default="all_proposals_combined_no_role.csv",
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
        default=None,
        help="Filter AI proposals by template/role (e.g., 'generate_ideas_no_role', 'single', 'group', 'group_int'). Default: None (all templates)"
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
    parser.add_argument(
        "--proposals-json",
        type=str,
        default=None,
        help="Path to JSON file containing specific proposal IDs to analyze (looks for 'selected_proposals' field)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ProposalSimilarityAnalyzer(proposals_csv=args.csv)
    
    # Check if JSON file with proposal IDs is provided
    proposal_ids = None
    json_output_path = None
    if args.proposals_json:
        logger.info(f"Loading proposal IDs from JSON: {args.proposals_json}")
        proposal_ids = analyzer.load_proposal_ids_from_json(args.proposals_json)
        if not proposal_ids:
            logger.error("No proposal IDs loaded from JSON file. Exiting.")
            return
        
        # Determine output path based on input JSON location
        if not args.output:
            input_path = Path(args.proposals_json)
            output_filename = f"similarity_{input_path.stem}.json"
            json_output_path = input_path.parent / output_filename
            logger.info(f"Output will be saved to: {json_output_path}")
    
    # Load proposals based on comparison type
    logger.info(f"Starting similarity analysis: {args.compare_type}")
    
    if proposal_ids:
        # If JSON file provided, load only those specific proposals
        logger.info("Using proposals from JSON file for analysis")
        logger.info(f"Target proposal IDs: {proposal_ids}")
        
        proposals_1 = analyzer.load_proposals(proposal_ids=proposal_ids)
        proposals_2 = analyzer.load_proposals(proposal_ids=proposal_ids)
        
        # Verify that we loaded the correct proposals
        if proposals_1:
            loaded_ids = [p.get('proposal_id') for p in proposals_1]
            logger.info(f"Actually loaded {len(proposals_1)} proposals with IDs: {loaded_ids}")
            
            # Check for mismatches
            missing_ids = set(proposal_ids) - set(loaded_ids)
            if missing_ids:
                logger.warning(f"WARNING: Could not find {len(missing_ids)} proposals in CSV: {missing_ids}")
            
            extra_ids = set(loaded_ids) - set(proposal_ids)
            if extra_ids:
                logger.error(f"ERROR: Loaded {len(extra_ids)} unexpected proposals: {extra_ids}")
                logger.error("This should not happen! Check the filtering logic.")
        
        # Auto-detect comparison type based on loaded proposals
        if proposals_1:
            who_types = set(p.get('who') for p in proposals_1)
            if len(who_types) == 1:
                if 'human' in who_types:
                    detected_type = "human-human"
                else:
                    detected_type = "ai-ai"
            else:
                detected_type = "human-ai"
            
            logger.info(f"Auto-detected comparison type from JSON proposals: {detected_type}")
            comparison_type = detected_type
        else:
            logger.error("No proposals loaded from JSON file!")
            comparison_type = args.compare_type
    else:
        # Use the command-line comparison type
        comparison_type = args.compare_type
        
        if args.compare_type == "human-human":
            proposals_1 = analyzer.load_proposals(who="human", max_proposals=args.max_proposals)
            proposals_2 = analyzer.load_proposals(who="human", max_proposals=args.max_proposals)
        elif args.compare_type == "ai-ai":
            proposals_1 = analyzer.load_proposals(
                who="ai",
                role=args.template,
                model=args.ai_model,
                max_proposals=args.max_proposals
            )
            proposals_2 = analyzer.load_proposals(
                who="ai",
                role=args.template,
                model=args.ai_model,
                max_proposals=args.max_proposals
            )
        else:  # human-ai
            proposals_1 = analyzer.load_proposals(who="human", max_proposals=args.max_proposals)
            proposals_2 = analyzer.load_proposals(
                who="ai",
                role=args.template,
                model=args.ai_model,
                max_proposals=args.max_proposals
            )
    
    # Perform similarity analysis
    all_results = analyzer.analyze_all_pairs(
        proposals_1=proposals_1,
        proposals_2=proposals_2,
        comparison_type=comparison_type
    )
    
    # Save results with role and model info for better filename
    # Use json_output_path if it was set (when --proposals-json was provided)
    output_to_use = str(json_output_path) if json_output_path else args.output
    
    analyzer.save_results(
        results=all_results,
        comparison_type=comparison_type,
        output_filename=output_to_use,
        role_filter=args.template,
        model_filter=args.ai_model
    )
    
    logger.info(f"Similarity analysis complete!")


if __name__ == "__main__":
    main()

