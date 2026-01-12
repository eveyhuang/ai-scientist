#!/usr/bin/env python3
"""
Evaluate textual similarity between different reviewers' reviews for the same proposal.
Uses the same methods as evaluate_proposals_similarity.py for consistency:
- TF-IDF cosine similarity
- Nomic-Embed semantic embeddings
- Keyword overlap (TF-IDF based)
- LDA topic modeling
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

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment analysis
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReviewerSimilarityResult:
    """Container for reviewer similarity analysis results"""
    proposal_id: str
    proposal_title: str
    year: str
    reviewer_1_id: str
    reviewer_2_id: str
    criterion: str  # 'combined' or specific criterion name
    tfidf_similarity: float
    embedding_similarity: float
    keyword_overlap: Dict[str, Any]
    topic_overlap: Dict[str, Any]
    timestamp: str


class ReviewerSimilarityAnalyzer:
    """Analyze textual similarity between different reviewers' reviews"""
    
    # Define justification columns for HUMAN reviews
    HUMAN_JUSTIFICATION_COLUMNS = [
        'scientific_merit_and_innovation_justification',
        'feasibility_justification',
        'data_sources_and_limitations_justification',
        'open_science_compliance_justification',
        'overall_rating_summary'
    ]
    
    # Define justification columns for AI reviews (from all_evaluations_by_ai_merged.csv)
    AI_JUSTIFICATION_COLUMNS = [
        'Scientific_Merit_and_Innovation_justification',
        'Feasibility_justification',
        'Data_Sources_and_Limitations_justification',
        'Open_Science_Compliance_justification',
        'narrative_summary'
    ]
    
    # Mapping between human and AI column names (for cross-comparison)
    COLUMN_MAPPING = {
        'scientific_merit_and_innovation_justification': 'Scientific_Merit_and_Innovation_justification',
        'feasibility_justification': 'Feasibility_justification',
        'data_sources_and_limitations_justification': 'Data_Sources_and_Limitations_justification',
        'open_science_compliance_justification': 'Open_Science_Compliance_justification',
        'overall_rating_summary': 'narrative_summary',
        # Reverse mapping
        'Scientific_Merit_and_Innovation_justification': 'scientific_merit_and_innovation_justification',
        'Feasibility_justification': 'feasibility_justification',
        'Data_Sources_and_Limitations_justification': 'data_sources_and_limitations_justification',
        'Open_Science_Compliance_justification': 'open_science_compliance_justification',
        'narrative_summary': 'overall_rating_summary'
    }
    
    # Friendly names for criteria (works for both formats)
    CRITERION_NAMES = {
        'scientific_merit_and_innovation_justification': 'Scientific Merit & Innovation',
        'Scientific_Merit_and_Innovation_justification': 'Scientific Merit & Innovation',
        'feasibility_justification': 'Feasibility',
        'Feasibility_justification': 'Feasibility',
        'data_sources_and_limitations_justification': 'Data Sources & Limitations',
        'Data_Sources_and_Limitations_justification': 'Data Sources & Limitations',
        'open_science_compliance_justification': 'Open Science Compliance',
        'Open_Science_Compliance_justification': 'Open Science Compliance',
        'overall_rating_summary': 'Overall Rating',
        'narrative_summary': 'Overall Rating',
        'combined': 'All Justifications Combined'
    }
    
    def __init__(self, 
                 reviews_file: str = "qualitative_evaluation/all_human_reviews.xlsx",
                 ai_reviews_file: str = "qualitative_evaluation/all_evaluations_by_ai_merged.csv"):
        """Initialize the reviewer similarity analyzer"""
        self.reviews_file = reviews_file
        self.ai_reviews_file = ai_reviews_file
        self.reviews_df = None
        self.ai_reviews_df = None
        self._load_reviews()
        self._load_ai_reviews()
        
        self.results_dir = Path("semantic_similarity/reviewer_similarity")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TF-IDF vectorizer (same settings as evaluate_proposals_similarity.py)
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
        
        logger.info("ReviewerSimilarityAnalyzer initialized")
    
    def _load_reviews(self):
        """Load human reviews from the Excel file"""
        try:
            self.reviews_df = pd.read_excel(self.reviews_file)
            logger.info(f"Loaded {len(self.reviews_df)} human reviews from {self.reviews_file}")
            
            # Get unique proposals and reviewers
            n_proposals = self.reviews_df.groupby(['year', 'id']).ngroups
            n_reviewers = self.reviews_df['reviewer_id'].nunique()
            logger.info(f"  - Unique proposals: {n_proposals}")
            logger.info(f"  - Unique reviewers: {n_reviewers}")
        except Exception as e:
            logger.error(f"Error loading human reviews: {e}")
            self.reviews_df = pd.DataFrame()
    
    def _load_ai_reviews(self):
        """Load AI reviews from the CSV file"""
        try:
            if self.ai_reviews_file and Path(self.ai_reviews_file).exists():
                self.ai_reviews_df = pd.read_csv(self.ai_reviews_file)
                logger.info(f"Loaded {len(self.ai_reviews_df)} AI reviews from {self.ai_reviews_file}")
                
                # Filter to human_y1 proposals only (for comparison with human reviews)
                if 'source' in self.ai_reviews_df.columns:
                    human_y1_ai = self.ai_reviews_df[self.ai_reviews_df['source'] == 'human_y1']
                    logger.info(f"  - Human Y1 proposals with AI reviews: {len(human_y1_ai)}")
            else:
                logger.info("No AI reviews file provided or file not found")
                self.ai_reviews_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading AI reviews: {e}")
            self.ai_reviews_df = pd.DataFrame()
    
    def _clean_text(self, text: Any) -> str:
        """
        Clean text field, handling NaN and empty values
        (Same as evaluate_proposals_similarity.py)
        """
        if pd.isna(text):
            return ''
        if not isinstance(text, str):
            return str(text)
        return text.strip()
    
    def combine_justifications(self, row: pd.Series, source_type: str = 'human') -> str:
        """
        Combine all justification columns into one text
        
        Args:
            row: DataFrame row
            source_type: 'human' for human reviews, 'ai' for AI reviews
        """
        if source_type == 'ai':
            columns = self.AI_JUSTIFICATION_COLUMNS
        else:
            columns = self.HUMAN_JUSTIFICATION_COLUMNS
        
        texts = []
        for col in columns:
            text = self._clean_text(row.get(col, ''))
            if text:
                texts.append(text)
        return ' '.join(texts)
    
    def get_justification_text(self, row: pd.Series, criterion: str, source_type: str = 'human') -> str:
        """
        Get justification text for a specific criterion, handling column name differences
        
        Args:
            row: DataFrame row
            criterion: Criterion column name (can be in human or AI format)
            source_type: 'human' or 'ai'
        """
        if criterion == 'combined':
            return self.combine_justifications(row, source_type)
        
        # Try direct column access first
        text = self._clean_text(row.get(criterion, ''))
        if text:
            return text
        
        # Try mapped column name
        mapped_col = self.COLUMN_MAPPING.get(criterion)
        if mapped_col:
            text = self._clean_text(row.get(mapped_col, ''))
        
        return text
    
    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF based cosine similarity between two texts
        (Same as evaluate_proposals_similarity.py)
        """
        try:
            if not text1 or not text2:
                return 0.0
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing TF-IDF similarity: {e}")
            return 0.0
    
    def get_embedding(self, text: str, cache_key: str = None) -> List[float]:
        """
        Get local embedding for a text using Nomic-Embed (with caching)
        (Same as evaluate_proposals_similarity.py)
        """
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            max_length = 30000
            if len(text) > max_length:
                logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
                text = text[:max_length]
            
            prefixed_text = f"search_document: {text}"
            embedding = self.embedding_model.encode(
                prefixed_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embedding_list = embedding.tolist()
            
            if cache_key:
                self.embedding_cache[cache_key] = embedding_list
            
            return embedding_list
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def compute_embedding_similarity(self, text1: str, text2: str, 
                                     cache_key1: str = None, cache_key2: str = None) -> float:
        """
        Compute embedding-based cosine similarity between two texts
        (Same as evaluate_proposals_similarity.py)
        """
        try:
            if not text1 or not text2:
                return None
            
            emb1 = self.get_embedding(text1, cache_key1)
            emb2 = self.get_embedding(text2, cache_key2)
            
            if emb1 is None or emb2 is None:
                return None
            
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
        (Same as evaluate_proposals_similarity.py)
        """
        try:
            if not text:
                return []
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def compute_keyword_overlap(self, text1: str, text2: str, top_n: int = 20) -> Dict[str, Any]:
        """
        Compute keyword overlap between two texts
        (Same as evaluate_proposals_similarity.py)
        """
        keywords1 = self.extract_keywords(text1, top_n)
        keywords2 = self.extract_keywords(text2, top_n)
        
        kw_set1 = set([kw for kw, _ in keywords1])
        kw_set2 = set([kw for kw, _ in keywords2])
        
        overlap = kw_set1.intersection(kw_set2)
        union = kw_set1.union(kw_set2)
        jaccard_similarity = len(overlap) / len(union) if len(union) > 0 else 0
        
        return {
            'overlapping_keywords': sorted(list(overlap)),
            'num_overlapping': len(overlap),
            'jaccard_similarity': jaccard_similarity,
            'reviewer_1_unique': sorted(list(kw_set1 - kw_set2))[:10],
            'reviewer_2_unique': sorted(list(kw_set2 - kw_set1))[:10]
        }
    
    def compute_topic_overlap(self, text1: str, text2: str, n_topics: int = 5) -> Dict[str, Any]:
        """
        Compute topic overlap between two texts using LDA
        (Same as evaluate_proposals_similarity.py)
        """
        try:
            if not text1 or not text2:
                return {'error': 'Empty text'}
            
            # Sort texts to ensure consistent ordering for symmetric results
            texts_sorted = sorted([text1, text2], key=lambda x: hash(x))
            text1_is_first = (texts_sorted[0] == text1)
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_sorted)
            
            # Fit LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            lda.fit(tfidf_matrix)
            
            # Get topic distributions
            topic_dist = lda.transform(tfidf_matrix)
            
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
                'reviewer_1_topic_distribution': dist1.tolist(),
                'reviewer_2_topic_distribution': dist2.tolist(),
                'topics': topics
            }
        except Exception as e:
            logger.error(f"Error computing topic overlap: {e}")
            return {'error': str(e)}
    
    def compute_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Compute sentiment scores for a text using TextBlob
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment metrics:
            - polarity: -1 (negative) to +1 (positive)
            - subjectivity: 0 (objective) to 1 (subjective)
            - sentiment_label: 'positive', 'negative', or 'neutral'
        """
        try:
            if not text:
                return {
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'sentiment_label': 'neutral'
                }
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to +1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'polarity': float(polarity),
                'subjectivity': float(subjectivity),
                'sentiment_label': sentiment_label
            }
        except Exception as e:
            logger.error(f"Error computing sentiment: {e}")
            return {
                'polarity': None,
                'subjectivity': None,
                'sentiment_label': None
            }
    
    def compute_sentiment_agreement(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compute sentiment agreement between two texts
        
        Measures whether both reviews have similar sentiment (both positive, both negative)
        or different sentiment (one positive, one negative)
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Dictionary with sentiment comparison metrics:
            - sentiment_1: sentiment of first text
            - sentiment_2: sentiment of second text
            - polarity_difference: absolute difference in polarity (-2 to +2 range, smaller = more aligned)
            - polarity_correlation: whether polarities point same direction
            - sentiment_agreement: 'agree' (same label), 'disagree' (opposite), or 'partial' (one neutral)
            - sentiment_alignment_score: 0-1 score where 1 = perfect alignment
        """
        try:
            sent1 = self.compute_sentiment(text1)
            sent2 = self.compute_sentiment(text2)
            
            pol1 = sent1.get('polarity', 0) or 0
            pol2 = sent2.get('polarity', 0) or 0
            label1 = sent1.get('sentiment_label', 'neutral')
            label2 = sent2.get('sentiment_label', 'neutral')
            
            # Polarity difference (0 = identical sentiment, 2 = completely opposite)
            polarity_difference = abs(pol1 - pol2)
            
            # Sentiment alignment score (0-1, where 1 = perfect alignment)
            # Transform polarity difference from [0, 2] to [1, 0]
            sentiment_alignment_score = 1 - (polarity_difference / 2)
            
            # Check if polarities point in same direction
            if pol1 * pol2 > 0:  # Both positive or both negative
                polarity_correlation = 'same_direction'
            elif pol1 * pol2 < 0:  # Opposite directions
                polarity_correlation = 'opposite_direction'
            else:  # At least one is neutral/zero
                polarity_correlation = 'neutral'
            
            # Determine agreement level
            if label1 == label2:
                sentiment_agreement = 'agree'
            elif (label1 == 'positive' and label2 == 'negative') or \
                 (label1 == 'negative' and label2 == 'positive'):
                sentiment_agreement = 'disagree'
            else:
                sentiment_agreement = 'partial'
            
            return {
                'sentiment_1': sent1,
                'sentiment_2': sent2,
                'polarity_difference': float(polarity_difference),
                'polarity_correlation': polarity_correlation,
                'sentiment_agreement': sentiment_agreement,
                'sentiment_alignment_score': float(sentiment_alignment_score)
            }
        except Exception as e:
            logger.error(f"Error computing sentiment agreement: {e}")
            return {'error': str(e)}
    
    def analyze_reviewer_pair(self,
                              proposal_group: pd.DataFrame,
                              reviewer_1_id: str,
                              reviewer_2_id: str,
                              criterion: str = 'combined') -> Dict[str, Any]:
        """
        Analyze similarity between two reviewers' reviews for a single proposal
        
        Args:
            proposal_group: DataFrame subset for a single proposal
            reviewer_1_id: First reviewer's ID
            reviewer_2_id: Second reviewer's ID
            criterion: Which criterion to analyze ('combined' or specific column name)
        
        Returns:
            Dictionary with similarity metrics
        """
        # Get reviews for each reviewer
        r1_review = proposal_group[proposal_group['reviewer_id'] == reviewer_1_id].iloc[0]
        r2_review = proposal_group[proposal_group['reviewer_id'] == reviewer_2_id].iloc[0]
        
        proposal_id = str(r1_review['id'])
        proposal_title = str(r1_review['title'])
        year = str(r1_review['year'])
        
        # Get texts to compare using unified method
        text1 = self.get_justification_text(r1_review, criterion, 'human')
        text2 = self.get_justification_text(r2_review, criterion, 'human')
        
        # Skip if either text is empty
        if not text1 or not text2:
            logger.warning(f"Skipping {proposal_id} ({reviewer_1_id} vs {reviewer_2_id}): Empty text")
            return {
                'proposal_id': proposal_id,
                'proposal_title': proposal_title,
                'year': year,
                'reviewer_1_id': str(reviewer_1_id),
                'reviewer_2_id': str(reviewer_2_id),
                'criterion': criterion,
                'criterion_name': self.CRITERION_NAMES.get(criterion, criterion),
                'comparison_type': 'human-human',
                'error': 'Empty text in one or both reviews'
            }
        
        # Cache keys for embeddings
        cache_key1 = f"{year}_{proposal_id}_{reviewer_1_id}_{criterion}"
        cache_key2 = f"{year}_{proposal_id}_{reviewer_2_id}_{criterion}"
        
        # Compute all similarity metrics
        tfidf_sim = self.compute_tfidf_similarity(text1, text2)
        embedding_sim = self.compute_embedding_similarity(text1, text2, cache_key1, cache_key2)
        keyword_overlap = self.compute_keyword_overlap(text1, text2)
        topic_overlap = self.compute_topic_overlap(text1, text2)
        
        # Compute sentiment agreement
        sentiment_agreement = self.compute_sentiment_agreement(text1, text2)
        
        result = {
            'proposal_id': proposal_id,
            'proposal_title': proposal_title,
            'year': year,
            'reviewer_1_id': str(reviewer_1_id),
            'reviewer_2_id': str(reviewer_2_id),
            'criterion': criterion,
            'criterion_name': self.CRITERION_NAMES.get(criterion, criterion),
            'comparison_type': 'human-human',
            'timestamp': datetime.now().isoformat(),
            'similarity_metrics': {
                'tfidf_cosine_similarity': tfidf_sim,
                'embedding_cosine_similarity': embedding_sim,
                'keyword_overlap': keyword_overlap,
                'topic_analysis': topic_overlap,
                'sentiment_analysis': sentiment_agreement
            }
        }
        
        return result
    
    def analyze_all_reviewer_pairs(self,
                                   criterion: str = 'combined',
                                   year_filter: str = None) -> List[Dict[str, Any]]:
        """
        Analyze similarity for all pairs of reviewers for each proposal
        
        Args:
            criterion: Which criterion to analyze ('combined' or specific column name)
            year_filter: Optional year to filter proposals
        
        Returns:
            List of all similarity analysis results
        """
        df = self.reviews_df.copy()
        
        if year_filter:
            df = df[df['year'] == year_filter]
            logger.info(f"Filtered to year {year_filter}: {len(df)} reviews")
        
        # Group by proposal
        grouped = df.groupby(['year', 'id', 'title'])
        n_proposals = grouped.ngroups
        logger.info(f"Analyzing {n_proposals} proposals")
        logger.info(f"Criterion: {self.CRITERION_NAMES.get(criterion, criterion)}")
        
        all_results = []
        proposal_count = 0
        
        for (year, prop_id, title), group in grouped:
            proposal_count += 1
            reviewers = group['reviewer_id'].unique()
            
            if len(reviewers) < 2:
                logger.warning(f"Skipping proposal {prop_id}: Only {len(reviewers)} reviewer(s)")
                continue
            
            # Generate all pairs of reviewers
            reviewer_pairs = list(itertools.combinations(reviewers, 2))
            logger.info(f"[{proposal_count}/{n_proposals}] Proposal {prop_id}: {len(reviewer_pairs)} reviewer pairs")
            
            for r1, r2 in reviewer_pairs:
                try:
                    result = self.analyze_reviewer_pair(group, r1, r2, criterion)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {prop_id} ({r1} vs {r2}): {e}")
                    error_result = {
                        'proposal_id': str(prop_id),
                        'year': str(year),
                        'reviewer_1_id': str(r1),
                        'reviewer_2_id': str(r2),
                        'criterion': criterion,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(error_result)
        
        logger.info(f"Completed {len(all_results)} reviewer pair analyses")
        return all_results
    
    def analyze_all_criteria(self, year_filter: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze similarity for all criteria (combined + individual)
        
        Returns:
            Dictionary mapping criterion names to their results
        """
        all_results = {}
        
        # Analyze combined justifications
        logger.info("=== Analyzing combined justifications ===")
        all_results['combined'] = self.analyze_all_reviewer_pairs('combined', year_filter)
        
        # Analyze each individual criterion
        for criterion in self.HUMAN_JUSTIFICATION_COLUMNS:
            logger.info(f"=== Analyzing {self.CRITERION_NAMES[criterion]} ===")
            all_results[criterion] = self.analyze_all_reviewer_pairs(criterion, year_filter)
        
        return all_results
    
    def analyze_human_ai_pair(self,
                              human_review: pd.Series,
                              ai_review: pd.Series,
                              criterion: str = 'combined') -> Dict[str, Any]:
        """
        Analyze similarity between a human review and an AI review for the same proposal
        
        Args:
            human_review: Human review row
            ai_review: AI review row
            criterion: Which criterion to analyze ('combined' or specific column name)
        
        Returns:
            Dictionary with similarity metrics
        """
        # Normalize proposal ID (handle float to int conversion)
        raw_id = human_review.get('id', ai_review.get('proposal_id', 'unknown'))
        try:
            proposal_id = str(int(float(raw_id))) if pd.notna(raw_id) else 'unknown'
        except (ValueError, TypeError):
            proposal_id = str(raw_id)
        
        proposal_title = str(human_review.get('title', ai_review.get('proposal_title', 'N/A')))
        year = str(human_review.get('year', 'N/A'))
        human_reviewer_id = str(human_review.get('reviewer_id', 'human'))
        ai_source = str(ai_review.get('source', 'ai'))
        
        # Get texts to compare
        text1 = self.get_justification_text(human_review, criterion, 'human')
        text2 = self.get_justification_text(ai_review, criterion, 'ai')
        
        # Skip if either text is empty
        if not text1 or not text2:
            logger.warning(f"Skipping {proposal_id} (human {human_reviewer_id} vs AI): Empty text")
            return {
                'proposal_id': proposal_id,
                'proposal_title': proposal_title,
                'year': year,
                'human_reviewer_id': human_reviewer_id,
                'ai_source': ai_source,
                'criterion': criterion,
                'criterion_name': self.CRITERION_NAMES.get(criterion, criterion),
                'comparison_type': 'human-ai',
                'error': 'Empty text in one or both reviews'
            }
        
        # Cache keys for embeddings
        cache_key1 = f"human_{year}_{proposal_id}_{human_reviewer_id}_{criterion}"
        cache_key2 = f"ai_{proposal_id}_{criterion}"
        
        # Compute all similarity metrics
        tfidf_sim = self.compute_tfidf_similarity(text1, text2)
        embedding_sim = self.compute_embedding_similarity(text1, text2, cache_key1, cache_key2)
        keyword_overlap = self.compute_keyword_overlap(text1, text2)
        topic_overlap = self.compute_topic_overlap(text1, text2)
        
        # Compute sentiment agreement
        sentiment_agreement = self.compute_sentiment_agreement(text1, text2)
        
        result = {
            'proposal_id': proposal_id,
            'proposal_title': proposal_title,
            'year': year,
            'human_reviewer_id': human_reviewer_id,
            'ai_source': ai_source,
            'criterion': criterion,
            'criterion_name': self.CRITERION_NAMES.get(criterion, criterion),
            'comparison_type': 'human-ai',
            'timestamp': datetime.now().isoformat(),
            'similarity_metrics': {
                'tfidf_cosine_similarity': tfidf_sim,
                'embedding_cosine_similarity': embedding_sim,
                'keyword_overlap': keyword_overlap,
                'topic_analysis': topic_overlap,
                'sentiment_analysis': sentiment_agreement
            }
        }
        
        return result
    
    def analyze_all_human_ai_pairs(self,
                                   criterion: str = 'combined',
                                   year_filter: str = None,
                                   ai_source_filter: str = 'human_y1') -> List[Dict[str, Any]]:
        """
        Analyze similarity between all human reviews and corresponding AI reviews
        
        Args:
            criterion: Which criterion to analyze ('combined' or specific column name)
            year_filter: Optional year to filter human reviews
            ai_source_filter: Filter AI reviews by source (default: 'human_y1')
        
        Returns:
            List of all similarity analysis results
        """
        if self.ai_reviews_df is None or len(self.ai_reviews_df) == 0:
            logger.error("No AI reviews loaded. Cannot perform human-AI comparison.")
            return []
        
        human_df = self.reviews_df.copy()
        ai_df = self.ai_reviews_df.copy()
        
        # Filter human reviews by year
        if year_filter:
            human_df = human_df[human_df['year'] == year_filter]
            logger.info(f"Filtered human reviews to year {year_filter}: {len(human_df)} reviews")
        
        # Filter AI reviews by source
        if ai_source_filter and 'source' in ai_df.columns:
            ai_df = ai_df[ai_df['source'] == ai_source_filter]
            logger.info(f"Filtered AI reviews to source '{ai_source_filter}': {len(ai_df)} reviews")
        
        # Create mapping from proposal_id to AI review
        # For human_y1 proposals, the proposal_id in AI reviews is just the numeric ID
        ai_review_map = {}
        for _, ai_row in ai_df.iterrows():
            pid = ai_row.get('proposal_id')
            if pd.notna(pid):
                # Normalize to string without decimal (e.g., '1' not '1.0')
                try:
                    pid_normalized = str(int(float(pid)))
                except (ValueError, TypeError):
                    pid_normalized = str(pid).strip()
                ai_review_map[pid_normalized] = ai_row
        
        logger.info(f"AI reviews available for {len(ai_review_map)} proposals")
        logger.info(f"Criterion: {self.CRITERION_NAMES.get(criterion, criterion)}")
        
        all_results = []
        comparison_count = 0
        
        # For each human review, find corresponding AI review
        for _, human_row in human_df.iterrows():
            human_id = human_row.get('id', '')
            # Normalize human ID to string without decimal (e.g., '1' not '1.0')
            if pd.notna(human_id):
                try:
                    human_proposal_id = str(int(float(human_id)))
                except (ValueError, TypeError):
                    human_proposal_id = str(human_id).strip()
            else:
                continue  # Skip rows with missing ID
            
            # Find matching AI review
            ai_row = ai_review_map.get(human_proposal_id)
            
            if ai_row is None:
                logger.debug(f"No AI review found for proposal {human_proposal_id}")
                continue
            
            comparison_count += 1
            
            try:
                result = self.analyze_human_ai_pair(human_row, ai_row, criterion)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing human-AI pair for proposal {human_proposal_id}: {e}")
                error_result = {
                    'proposal_id': human_proposal_id,
                    'human_reviewer_id': str(human_row.get('reviewer_id', 'unknown')),
                    'ai_source': ai_source_filter,
                    'criterion': criterion,
                    'comparison_type': 'human-ai',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(error_result)
        
        logger.info(f"Completed {len(all_results)} human-AI comparisons")
        return all_results
    
    def analyze_all_human_ai_criteria(self, 
                                      year_filter: str = None,
                                      ai_source_filter: str = 'human_y1') -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze human-AI similarity for all criteria
        
        Returns:
            Dictionary mapping criterion names to their results
        """
        all_results = {}
        
        # Analyze combined justifications
        logger.info("=== Analyzing combined justifications (Human vs AI) ===")
        all_results['combined'] = self.analyze_all_human_ai_pairs('combined', year_filter, ai_source_filter)
        
        # Analyze each individual criterion (use human column names)
        for criterion in self.HUMAN_JUSTIFICATION_COLUMNS:
            logger.info(f"=== Analyzing {self.CRITERION_NAMES[criterion]} (Human vs AI) ===")
            all_results[criterion] = self.analyze_all_human_ai_pairs(criterion, year_filter, ai_source_filter)
        
        return all_results
    
    def save_results(self,
                     results: List[Dict[str, Any]],
                     criterion: str,
                     output_filename: str = None,
                     comparison_type: str = None):
        """Save similarity analysis results to JSON and CSV in appropriate subfolder"""
        
        # Auto-detect comparison_type from results if not provided
        if comparison_type is None:
            if results and len(results) > 0:
                comparison_type = results[0].get('comparison_type', 'human-human')
            else:
                comparison_type = 'human-human'
        
        # Create subfolder based on comparison type
        subfolder = self.results_dir / comparison_type
        subfolder.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            criterion_safe = criterion.replace('_justification', '').replace(' ', '_')
            output_filename = f"similarity_{comparison_type}_{criterion_safe}_{timestamp}"
        
        output_path = subfolder / f"{output_filename}.json"
        
        # Create results dict with metadata
        results_dict = {
            "metadata": {
                "analysis_type": "reviewer_similarity",
                "comparison_type": comparison_type,
                "criterion": criterion,
                "criterion_name": self.CRITERION_NAMES.get(criterion, criterion),
                "total_comparisons": len(results),
                "generation_timestamp": datetime.now().isoformat(),
                "methods": {
                    "tfidf": "TF-IDF based cosine similarity",
                    "embeddings": "Nomic-Embed (nomic-ai/nomic-embed-text-v1.5)",
                    "keywords": "TF-IDF top-20 keywords with Jaccard similarity",
                    "topics": "Latent Dirichlet Allocation (LDA) with 5 topics"
                }
            },
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
        
        # Also save as CSV
        csv_path = output_path.with_suffix('.csv')
        self._save_results_csv(results, csv_path)
        
        return output_path
    
    def _save_results_csv(self, results: List[Dict[str, Any]], csv_path: Path):
        """Save simplified results to CSV (same format as evaluate_proposals_similarity.py)"""
        csv_data = []
        for result in results:
            if 'error' in result:
                continue
            
            metrics = result.get('similarity_metrics', {})
            comparison_type = result.get('comparison_type', 'human-human')
            sentiment = metrics.get('sentiment_analysis', {})
            
            row = {
                'year': result.get('year'),
                'proposal_id': result.get('proposal_id'),
                'proposal_title': result.get('proposal_title'),
                'comparison_type': comparison_type,
                'criterion': result.get('criterion_name'),
                'tfidf_similarity': metrics.get('tfidf_cosine_similarity'),
                'embedding_similarity': metrics.get('embedding_cosine_similarity'),
                'keyword_overlap_count': metrics.get('keyword_overlap', {}).get('num_overlapping'),
                'keyword_jaccard': metrics.get('keyword_overlap', {}).get('jaccard_similarity'),
                'topic_similarity': metrics.get('topic_analysis', {}).get('topic_similarity'),
                # Sentiment metrics
                'sentiment_alignment': sentiment.get('sentiment_alignment_score'),
                'sentiment_agreement': sentiment.get('sentiment_agreement'),
                'polarity_1': sentiment.get('sentiment_1', {}).get('polarity'),
                'polarity_2': sentiment.get('sentiment_2', {}).get('polarity'),
                'sentiment_label_1': sentiment.get('sentiment_1', {}).get('sentiment_label'),
                'sentiment_label_2': sentiment.get('sentiment_2', {}).get('sentiment_label'),
                'polarity_correlation': sentiment.get('polarity_correlation')
            }
            
            # Add reviewer columns based on comparison type
            if comparison_type == 'human-ai':
                row['human_reviewer_id'] = result.get('human_reviewer_id')
                row['ai_source'] = result.get('ai_source')
            else:
                row['reviewer_1_id'] = result.get('reviewer_1_id')
                row['reviewer_2_id'] = result.get('reviewer_2_id')
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved simplified CSV results to {csv_path}")
    
    def generate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        # Extract metrics
        tfidf_sims = []
        embedding_sims = []
        keyword_jaccards = []
        topic_sims = []
        sentiment_alignments = []
        sentiment_agreements = {'agree': 0, 'disagree': 0, 'partial': 0}
        polarity_correlations = {'same_direction': 0, 'opposite_direction': 0, 'neutral': 0}
        
        for result in results:
            if 'error' in result:
                continue
            metrics = result.get('similarity_metrics', {})
            
            if metrics.get('tfidf_cosine_similarity') is not None:
                tfidf_sims.append(metrics['tfidf_cosine_similarity'])
            if metrics.get('embedding_cosine_similarity') is not None:
                embedding_sims.append(metrics['embedding_cosine_similarity'])
            if metrics.get('keyword_overlap', {}).get('jaccard_similarity') is not None:
                keyword_jaccards.append(metrics['keyword_overlap']['jaccard_similarity'])
            if metrics.get('topic_analysis', {}).get('topic_similarity') is not None:
                topic_sims.append(metrics['topic_analysis']['topic_similarity'])
            
            # Sentiment metrics
            sentiment = metrics.get('sentiment_analysis', {})
            if sentiment.get('sentiment_alignment_score') is not None:
                sentiment_alignments.append(sentiment['sentiment_alignment_score'])
            if sentiment.get('sentiment_agreement') in sentiment_agreements:
                sentiment_agreements[sentiment['sentiment_agreement']] += 1
            if sentiment.get('polarity_correlation') in polarity_correlations:
                polarity_correlations[sentiment['polarity_correlation']] += 1
        
        def compute_stats(values):
            if not values:
                return {'mean': None, 'std': None, 'min': None, 'max': None, 'median': None}
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'n': len(arr)
            }
        
        total_sentiment = sum(sentiment_agreements.values())
        
        return {
            'tfidf_similarity': compute_stats(tfidf_sims),
            'embedding_similarity': compute_stats(embedding_sims),
            'keyword_jaccard': compute_stats(keyword_jaccards),
            'topic_similarity': compute_stats(topic_sims),
            'sentiment_alignment': compute_stats(sentiment_alignments),
            'sentiment_agreement_counts': sentiment_agreements,
            'sentiment_agreement_pct': {
                k: (v / total_sentiment * 100) if total_sentiment > 0 else 0 
                for k, v in sentiment_agreements.items()
            },
            'polarity_correlation_counts': polarity_correlations
        }
    
    def create_visualization(self,
                            results: List[Dict[str, Any]],
                            criterion: str,
                            comparison_type: str,
                            output_dir: Path) -> Path:
        """
        Create and save visualization for a single criterion's results
        
        Args:
            results: List of similarity results
            criterion: Criterion name
            comparison_type: 'human-human' or 'human-ai'
            output_dir: Directory to save the visualization
        
        Returns:
            Path to saved visualization
        """
        # Extract metrics from results
        metrics_data = {
            'tfidf_similarity': [],
            'embedding_similarity': [],
            'keyword_jaccard': [],
            'topic_similarity': [],
            'sentiment_alignment': []
        }
        sentiment_agreement_counts = {'agree': 0, 'disagree': 0, 'partial': 0}
        
        for result in results:
            if 'error' in result:
                continue
            sim_metrics = result.get('similarity_metrics', {})
            
            if sim_metrics.get('tfidf_cosine_similarity') is not None:
                metrics_data['tfidf_similarity'].append(sim_metrics['tfidf_cosine_similarity'])
            if sim_metrics.get('embedding_cosine_similarity') is not None:
                metrics_data['embedding_similarity'].append(sim_metrics['embedding_cosine_similarity'])
            if sim_metrics.get('keyword_overlap', {}).get('jaccard_similarity') is not None:
                metrics_data['keyword_jaccard'].append(sim_metrics['keyword_overlap']['jaccard_similarity'])
            if sim_metrics.get('topic_analysis', {}).get('topic_similarity') is not None:
                metrics_data['topic_similarity'].append(sim_metrics['topic_analysis']['topic_similarity'])
            
            # Sentiment metrics
            sentiment = sim_metrics.get('sentiment_analysis', {})
            if sentiment.get('sentiment_alignment_score') is not None:
                metrics_data['sentiment_alignment'].append(sentiment['sentiment_alignment_score'])
            if sentiment.get('sentiment_agreement') in sentiment_agreement_counts:
                sentiment_agreement_counts[sentiment['sentiment_agreement']] += 1
        
        # Metric display names
        metric_names = {
            'tfidf_similarity': 'TF-IDF Cosine\nSimilarity',
            'embedding_similarity': 'Embedding\nSimilarity',
            'keyword_jaccard': 'Keyword Overlap\n(Jaccard)',
            'topic_similarity': 'Topic Similarity\n(LDA)',
            'sentiment_alignment': 'Sentiment\nAlignment'
        }
        
        # Colors based on comparison type
        if comparison_type == 'human-ai':
            box_color = '#E74C3C'  # Red for human-AI
            title_prefix = 'Human vs AI'
        else:
            box_color = '#3498DB'  # Blue for human-human
            title_prefix = 'Human vs Human'
        
        # Create figure with subplots - 5 box plots + 1 bar chart for sentiment agreement
        fig, axes = plt.subplots(1, 6, figsize=(20, 5))
        
        criterion_name = self.CRITERION_NAMES.get(criterion, criterion)
        
        # Plot box plots for first 5 metrics
        for idx, (metric_key, metric_label) in enumerate(metric_names.items()):
            ax = axes[idx]
            values = metrics_data[metric_key]
            
            if len(values) > 0:
                # Create box plot
                bp = ax.boxplot([values], patch_artist=True, showmeans=True, meanline=True)
                
                # Style the box plot
                bp['boxes'][0].set_facecolor(box_color)
                bp['boxes'][0].set_alpha(0.7)
                bp['medians'][0].set_color('black')
                bp['medians'][0].set_linewidth(2)
                bp['means'][0].set_color('green')
                bp['means'][0].set_linewidth(2)
                
                # Add statistics annotation
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.text(0.95, 0.95, f'n={len(values)}\nμ={mean_val:.3f}\nσ={std_val:.3f}',
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
            
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(metric_label, fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.grid(axis='y', alpha=0.3)
        
        # Plot sentiment agreement bar chart in last subplot
        ax = axes[5]
        agreement_labels = ['Agree', 'Partial', 'Disagree']
        agreement_values = [
            sentiment_agreement_counts['agree'],
            sentiment_agreement_counts['partial'],
            sentiment_agreement_counts['disagree']
        ]
        agreement_colors = ['#27AE60', '#F39C12', '#E74C3C']  # Green, Orange, Red
        
        bars = ax.bar(agreement_labels, agreement_values, color=agreement_colors, alpha=0.7)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Sentiment\nAgreement', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, agreement_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(val), ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'{title_prefix} Review Similarity: {criterion_name}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        criterion_safe = criterion.replace('_justification', '').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"viz_{comparison_type}_{criterion_safe}_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        return output_path
    
    def create_combined_visualization(self,
                                      all_results: Dict[str, List[Dict[str, Any]]],
                                      comparison_type: str,
                                      output_dir: Path) -> Path:
        """
        Create a combined visualization comparing all criteria
        
        Args:
            all_results: Dictionary mapping criterion names to their results
            comparison_type: 'human-human' or 'human-ai'
            output_dir: Directory to save the visualization
        
        Returns:
            Path to saved visualization
        """
        # Metric display names
        metric_names = {
            'tfidf_similarity': 'TF-IDF Similarity',
            'embedding_similarity': 'Embedding Similarity',
            'keyword_jaccard': 'Keyword Jaccard',
            'topic_similarity': 'Topic Similarity',
            'sentiment_alignment': 'Sentiment Alignment'
        }
        
        # Colors based on comparison type
        if comparison_type == 'human-ai':
            box_color = '#E74C3C'
            title_prefix = 'Human vs AI'
        else:
            box_color = '#3498DB'
            title_prefix = 'Human vs Human'
        
        # Prepare data for each metric across all criteria
        criteria_list = list(all_results.keys())
        criteria_labels = [self.CRITERION_NAMES.get(c, c).replace(' & ', '\n& ') for c in criteria_list]
        
        # Create figure - one row per metric
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
        
        for metric_idx, (metric_key, metric_label) in enumerate(metric_names.items()):
            ax = axes[metric_idx]
            
            # Collect data for each criterion
            data_to_plot = []
            valid_labels = []
            
            for criterion in criteria_list:
                results = all_results[criterion]
                values = []
                
                for result in results:
                    if 'error' in result:
                        continue
                    sim_metrics = result.get('similarity_metrics', {})
                    
                    if metric_key == 'tfidf_similarity':
                        val = sim_metrics.get('tfidf_cosine_similarity')
                    elif metric_key == 'embedding_similarity':
                        val = sim_metrics.get('embedding_cosine_similarity')
                    elif metric_key == 'keyword_jaccard':
                        val = sim_metrics.get('keyword_overlap', {}).get('jaccard_similarity')
                    elif metric_key == 'topic_similarity':
                        val = sim_metrics.get('topic_analysis', {}).get('topic_similarity')
                    elif metric_key == 'sentiment_alignment':
                        val = sim_metrics.get('sentiment_analysis', {}).get('sentiment_alignment_score')
                    else:
                        val = None
                    
                    if val is not None:
                        values.append(val)
                
                if len(values) > 0:
                    data_to_plot.append(values)
                    criterion_label = self.CRITERION_NAMES.get(criterion, criterion)
                    valid_labels.append(f'{criterion_label}\n(n={len(values)})')
            
            if len(data_to_plot) > 0:
                # Create box plot
                bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True, meanline=True)
                
                # Style boxes
                for box in bp['boxes']:
                    box.set_facecolor(box_color)
                    box.set_alpha(0.7)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
                for mean in bp['means']:
                    mean.set_color('green')
                    mean.set_linewidth(2)
                
                ax.set_xticklabels(valid_labels, fontsize=9, rotation=0)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
            
            ax.set_ylabel('Similarity Score', fontsize=11, fontweight='bold')
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{title_prefix} Review Similarity: All Criteria Comparison',
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"viz_{comparison_type}_all_criteria_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved combined visualization to {output_path}")
        return output_path
    
    def create_per_proposal_heatmap(self,
                                    results: List[Dict[str, Any]],
                                    criterion: str,
                                    comparison_type: str,
                                    output_dir: Path) -> Path:
        """
        Create a heatmap showing similarity per proposal
        
        Args:
            results: List of similarity results
            criterion: Criterion name
            comparison_type: 'human-human' or 'human-ai'
            output_dir: Directory to save the visualization
        
        Returns:
            Path to saved visualization
        """
        # Extract data per proposal
        proposal_data = {}
        
        for result in results:
            if 'error' in result:
                continue
            
            proposal_id = result.get('proposal_id', 'unknown')
            sim_metrics = result.get('similarity_metrics', {})
            
            if proposal_id not in proposal_data:
                proposal_data[proposal_id] = {
                    'title': result.get('proposal_title', ''),
                    'tfidf': [],
                    'embedding': [],
                    'keyword': [],
                    'topic': []
                }
            
            if sim_metrics.get('tfidf_cosine_similarity') is not None:
                proposal_data[proposal_id]['tfidf'].append(sim_metrics['tfidf_cosine_similarity'])
            if sim_metrics.get('embedding_cosine_similarity') is not None:
                proposal_data[proposal_id]['embedding'].append(sim_metrics['embedding_cosine_similarity'])
            if sim_metrics.get('keyword_overlap', {}).get('jaccard_similarity') is not None:
                proposal_data[proposal_id]['keyword'].append(sim_metrics['keyword_overlap']['jaccard_similarity'])
            if sim_metrics.get('topic_analysis', {}).get('topic_similarity') is not None:
                proposal_data[proposal_id]['topic'].append(sim_metrics['topic_analysis']['topic_similarity'])
        
        if len(proposal_data) == 0:
            logger.warning("No data for heatmap")
            return None
        
        # Create DataFrame for heatmap
        heatmap_data = []
        proposal_labels = []
        
        for proposal_id, data in sorted(proposal_data.items()):
            row = {
                'TF-IDF': np.mean(data['tfidf']) if data['tfidf'] else np.nan,
                'Embedding': np.mean(data['embedding']) if data['embedding'] else np.nan,
                'Keyword': np.mean(data['keyword']) if data['keyword'] else np.nan,
                'Topic': np.mean(data['topic']) if data['topic'] else np.nan
            }
            heatmap_data.append(row)
            # Truncate title for display
            title = data['title'][:30] + '...' if len(data['title']) > 30 else data['title']
            proposal_labels.append(f"{proposal_id}: {title}")
        
        df_heatmap = pd.DataFrame(heatmap_data, index=proposal_labels)
        
        # Create figure
        fig_height = max(6, len(proposal_labels) * 0.4)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        
        # Color based on comparison type
        cmap = 'Reds' if comparison_type == 'human-ai' else 'Blues'
        
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap=cmap,
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Mean Similarity'})
        
        criterion_name = self.CRITERION_NAMES.get(criterion, criterion)
        title_prefix = 'Human vs AI' if comparison_type == 'human-ai' else 'Human vs Human'
        ax.set_title(f'{title_prefix}: Mean Similarity per Proposal\n{criterion_name}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Similarity Metric', fontsize=11)
        ax.set_ylabel('Proposal', fontsize=11)
        
        plt.tight_layout()
        
        # Save figure
        criterion_safe = criterion.replace('_justification', '').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"heatmap_{comparison_type}_{criterion_safe}_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap to {output_path}")
        return output_path


def compare_human_human_vs_human_ai(
    hh_results: List[Dict[str, Any]],
    hai_results: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Statistically compare human-human vs human-AI similarity results.
    
    Uses:
    - Mann-Whitney U test for continuous metrics (embedding similarity, sentiment alignment)
    - Chi-square test for categorical metrics (sentiment agreement)
    - Effect size calculations (Cohen's d, Cramér's V)
    
    Args:
        hh_results: Human-human comparison results
        hai_results: Human-AI comparison results
        output_dir: Directory to save comparison results
    
    Returns:
        Dictionary with statistical comparison results
    """
    from scipy import stats
    
    # Extract continuous metrics
    def extract_metrics(results):
        metrics = {
            'tfidf_similarity': [],
            'embedding_similarity': [],
            'keyword_jaccard': [],
            'topic_similarity': [],
            'sentiment_alignment': []
        }
        sentiment_counts = {'agree': 0, 'partial': 0, 'disagree': 0}
        
        for result in results:
            if 'error' in result:
                continue
            sim = result.get('similarity_metrics', {})
            
            if sim.get('tfidf_cosine_similarity') is not None:
                metrics['tfidf_similarity'].append(sim['tfidf_cosine_similarity'])
            if sim.get('embedding_cosine_similarity') is not None:
                metrics['embedding_similarity'].append(sim['embedding_cosine_similarity'])
            if sim.get('keyword_overlap', {}).get('jaccard_similarity') is not None:
                metrics['keyword_jaccard'].append(sim['keyword_overlap']['jaccard_similarity'])
            if sim.get('topic_analysis', {}).get('topic_similarity') is not None:
                metrics['topic_similarity'].append(sim['topic_analysis']['topic_similarity'])
            
            sentiment = sim.get('sentiment_analysis', {})
            if sentiment.get('sentiment_alignment_score') is not None:
                metrics['sentiment_alignment'].append(sentiment['sentiment_alignment_score'])
            if sentiment.get('sentiment_agreement') in sentiment_counts:
                sentiment_counts[sentiment['sentiment_agreement']] += 1
        
        return metrics, sentiment_counts
    
    hh_metrics, hh_sentiment = extract_metrics(hh_results)
    hai_metrics, hai_sentiment = extract_metrics(hai_results)
    
    comparison_results = {
        'n_human_human': len([r for r in hh_results if 'error' not in r]),
        'n_human_ai': len([r for r in hai_results if 'error' not in r]),
        'continuous_metrics': {},
        'categorical_metrics': {}
    }
    
    # --- Statistical tests for continuous metrics ---
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return None
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    metric_names = {
        'tfidf_similarity': 'TF-IDF Similarity',
        'embedding_similarity': 'Embedding Similarity',
        'keyword_jaccard': 'Keyword Jaccard',
        'topic_similarity': 'Topic Similarity',
        'sentiment_alignment': 'Sentiment Alignment'
    }
    
    for metric_key, metric_name in metric_names.items():
        hh_vals = hh_metrics[metric_key]
        hai_vals = hai_metrics[metric_key]
        
        if len(hh_vals) < 2 or len(hai_vals) < 2:
            comparison_results['continuous_metrics'][metric_key] = {
                'error': 'Insufficient data for comparison'
            }
            continue
        
        # Mann-Whitney U test (non-parametric, doesn't assume normality)
        statistic, p_value = stats.mannwhitneyu(hh_vals, hai_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        effect_size = cohens_d(hh_vals, hai_vals)
        
        # Interpret effect size
        if effect_size is not None:
            abs_d = abs(effect_size)
            if abs_d < 0.2:
                effect_interpretation = 'negligible'
            elif abs_d < 0.5:
                effect_interpretation = 'small'
            elif abs_d < 0.8:
                effect_interpretation = 'medium'
            else:
                effect_interpretation = 'large'
        else:
            effect_interpretation = 'N/A'
        
        comparison_results['continuous_metrics'][metric_key] = {
            'metric_name': metric_name,
            'human_human': {
                'n': len(hh_vals),
                'mean': float(np.mean(hh_vals)),
                'std': float(np.std(hh_vals)),
                'median': float(np.median(hh_vals))
            },
            'human_ai': {
                'n': len(hai_vals),
                'mean': float(np.mean(hai_vals)),
                'std': float(np.std(hai_vals)),
                'median': float(np.median(hai_vals))
            },
            'mann_whitney_u': {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01
            },
            'effect_size': {
                'cohens_d': float(effect_size) if effect_size else None,
                'interpretation': effect_interpretation
            }
        }
    
    # --- Chi-square test for sentiment agreement ---
    # Create contingency table
    observed = np.array([
        [hh_sentiment['agree'], hh_sentiment['partial'], hh_sentiment['disagree']],
        [hai_sentiment['agree'], hai_sentiment['partial'], hai_sentiment['disagree']]
    ])
    
    # Only run chi-square if we have enough data
    if observed.sum() > 0 and (observed > 0).sum() >= 4:
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            
            # Cramér's V effect size
            n = observed.sum()
            min_dim = min(observed.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
            
            # Interpret Cramér's V
            if cramers_v < 0.1:
                v_interpretation = 'negligible'
            elif cramers_v < 0.3:
                v_interpretation = 'small'
            elif cramers_v < 0.5:
                v_interpretation = 'medium'
            else:
                v_interpretation = 'large'
            
            comparison_results['categorical_metrics']['sentiment_agreement'] = {
                'human_human_counts': dict(hh_sentiment),
                'human_ai_counts': dict(hai_sentiment),
                'chi_square': {
                    'statistic': float(chi2),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(dof),
                    'significant_at_0.05': p_value < 0.05,
                    'significant_at_0.01': p_value < 0.01
                },
                'effect_size': {
                    'cramers_v': float(cramers_v),
                    'interpretation': v_interpretation
                }
            }
        except Exception as e:
            comparison_results['categorical_metrics']['sentiment_agreement'] = {
                'error': str(e),
                'human_human_counts': dict(hh_sentiment),
                'human_ai_counts': dict(hai_sentiment)
            }
    else:
        comparison_results['categorical_metrics']['sentiment_agreement'] = {
            'error': 'Insufficient data for chi-square test',
            'human_human_counts': dict(hh_sentiment),
            'human_ai_counts': dict(hai_sentiment)
        }
    
    # --- Create comparison visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Colors
    hh_color = '#3498DB'  # Blue
    hai_color = '#E74C3C'  # Red
    
    # Plot continuous metrics as paired box plots
    continuous_keys = ['embedding_similarity', 'sentiment_alignment', 'tfidf_similarity', 
                       'keyword_jaccard', 'topic_similarity']
    
    for idx, metric_key in enumerate(continuous_keys):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        hh_vals = hh_metrics[metric_key]
        hai_vals = hai_metrics[metric_key]
        
        if len(hh_vals) > 0 and len(hai_vals) > 0:
            bp = ax.boxplot([hh_vals, hai_vals], patch_artist=True, 
                           tick_labels=['Human-Human', 'Human-AI'])
            bp['boxes'][0].set_facecolor(hh_color)
            bp['boxes'][1].set_facecolor(hai_color)
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            # Add significance annotation
            result = comparison_results['continuous_metrics'].get(metric_key, {})
            if 'mann_whitney_u' in result:
                p = result['mann_whitney_u']['p_value']
                sig_text = ''
                if p < 0.001:
                    sig_text = '***'
                elif p < 0.01:
                    sig_text = '**'
                elif p < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                # Add p-value and significance
                y_max = max(max(hh_vals), max(hai_vals))
                ax.text(1.5, y_max * 1.05, f'p={p:.4f} ({sig_text})', 
                       ha='center', fontsize=9)
                
                # Add effect size
                d = result['effect_size'].get('cohens_d')
                if d is not None:
                    ax.text(1.5, y_max * 0.95, f"d={d:.2f} ({result['effect_size']['interpretation']})",
                           ha='center', fontsize=8, style='italic')
        
        ax.set_title(metric_names[metric_key], fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    # Plot sentiment agreement as grouped bar chart
    ax = axes[1, 2]
    x = np.arange(3)
    width = 0.35
    
    hh_vals = [hh_sentiment['agree'], hh_sentiment['partial'], hh_sentiment['disagree']]
    hai_vals = [hai_sentiment['agree'], hai_sentiment['partial'], hai_sentiment['disagree']]
    
    bars1 = ax.bar(x - width/2, hh_vals, width, label='Human-Human', color=hh_color, alpha=0.7)
    bars2 = ax.bar(x + width/2, hai_vals, width, label='Human-AI', color=hai_color, alpha=0.7)
    
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Agreement', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Agree', 'Partial', 'Disagree'])
    ax.legend()
    
    # Add chi-square result
    cat_result = comparison_results['categorical_metrics'].get('sentiment_agreement', {})
    if 'chi_square' in cat_result:
        p = cat_result['chi_square']['p_value']
        sig_text = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        v = cat_result['effect_size'].get('cramers_v', 0)
        ax.text(0.95, 0.95, f"χ²: p={p:.4f} ({sig_text})\nV={v:.2f}",
               transform=ax.transAxes, ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Human-Human vs Human-AI Review Similarity Comparison\n(* p<0.05, ** p<0.01, *** p<0.001, ns=not significant)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"comparison_hh_vs_hai_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison visualization to {fig_path}")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save comparison results to JSON
    json_path = output_dir / f"statistical_comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(comparison_results), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistical comparison to {json_path}")
    
    return comparison_results


def print_comparison_summary(comparison_results: Dict[str, Any]):
    """Print a formatted summary of the statistical comparison"""
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON: Human-Human vs Human-AI")
    print("="*70)
    print(f"Sample sizes: Human-Human n={comparison_results['n_human_human']}, "
          f"Human-AI n={comparison_results['n_human_ai']}")
    print()
    
    print("CONTINUOUS METRICS (Mann-Whitney U Test):")
    print("-"*70)
    for metric_key, result in comparison_results['continuous_metrics'].items():
        if 'error' in result:
            print(f"  {metric_key}: {result['error']}")
            continue
        
        hh = result['human_human']
        hai = result['human_ai']
        mw = result['mann_whitney_u']
        es = result['effect_size']
        
        sig = "***" if mw['p_value'] < 0.001 else "**" if mw['p_value'] < 0.01 else "*" if mw['p_value'] < 0.05 else "ns"
        
        print(f"\n  {result['metric_name']}:")
        print(f"    Human-Human: mean={hh['mean']:.4f} ± {hh['std']:.4f} (n={hh['n']})")
        print(f"    Human-AI:    mean={hai['mean']:.4f} ± {hai['std']:.4f} (n={hai['n']})")
        print(f"    Difference:  {hh['mean'] - hai['mean']:+.4f}")
        print(f"    Mann-Whitney U: p={mw['p_value']:.4f} ({sig})")
        print(f"    Effect size: Cohen's d={es['cohens_d']:.3f} ({es['interpretation']})" if es['cohens_d'] else "")
    
    print("\n" + "-"*70)
    print("CATEGORICAL METRICS (Chi-Square Test):")
    print("-"*70)
    
    sent_result = comparison_results['categorical_metrics'].get('sentiment_agreement', {})
    if 'error' in sent_result:
        print(f"  Sentiment Agreement: {sent_result['error']}")
    elif 'chi_square' in sent_result:
        hh_counts = sent_result['human_human_counts']
        hai_counts = sent_result['human_ai_counts']
        chi = sent_result['chi_square']
        es = sent_result['effect_size']
        
        sig = "***" if chi['p_value'] < 0.001 else "**" if chi['p_value'] < 0.01 else "*" if chi['p_value'] < 0.05 else "ns"
        
        print(f"\n  Sentiment Agreement:")
        print(f"    Human-Human: Agree={hh_counts['agree']}, Partial={hh_counts['partial']}, Disagree={hh_counts['disagree']}")
        print(f"    Human-AI:    Agree={hai_counts['agree']}, Partial={hai_counts['partial']}, Disagree={hai_counts['disagree']}")
        print(f"    Chi-Square: χ²={chi['statistic']:.3f}, p={chi['p_value']:.4f} ({sig})")
        print(f"    Effect size: Cramér's V={es['cramers_v']:.3f} ({es['interpretation']})")
    
    print("\n" + "="*70)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001, ns=not significant")
    print("Effect size: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)")
    print("="*70)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze textual similarity between different reviewers' reviews"
    )
    parser.add_argument(
        "--reviews-file",
        type=str,
        default="qualitative_evaluation/all_human_reviews.xlsx",
        help="Path to human reviews Excel file"
    )
    parser.add_argument(
        "--ai-reviews-file",
        type=str,
        default="qualitative_evaluation/all_evaluations_by_ai_merged.csv",
        help="Path to AI reviews CSV file (for human-AI comparison)"
    )
    parser.add_argument(
        "--compare-type",
        type=str,
        default="human-human",
        choices=['human-human', 'human-ai'],
        help="Type of comparison: 'human-human' (between human reviewers) or 'human-ai' (human vs AI reviews)"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="combined",
        choices=['combined', 'scientific_merit_and_innovation_justification',
                 'feasibility_justification', 'data_sources_and_limitations_justification',
                 'open_science_compliance_justification', 'overall_rating_summary', 'all'],
        help="Which criterion to analyze ('combined' for all justifications, 'all' for each criterion separately)"
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Filter by year (optional)"
    )
    parser.add_argument(
        "--ai-source",
        type=str,
        default="human_y1",
        help="Filter AI reviews by source (default: 'human_y1')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (without extension)"
    )
    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="Run both human-human and human-AI comparisons, then statistically compare results"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running analysis even if saved results exist (use with --compare-both)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ReviewerSimilarityAnalyzer(
        reviews_file=args.reviews_file,
        ai_reviews_file=args.ai_reviews_file
    )
    
    # Special mode: Run both comparisons and statistically compare
    if args.compare_both:
        logger.info("=== Statistical Comparison: Human-Human vs Human-AI ===")
        
        # Get criterion to analyze
        criterion = args.criterion if args.criterion != 'all' else 'combined'
        criterion_safe = criterion.replace('_justification', '').replace(' ', '_')
        logger.info(f"Analyzing criterion: {analyzer.CRITERION_NAMES.get(criterion, criterion)}")
        
        # Try to load existing results from saved JSON files (unless --force-rerun)
        hh_dir = analyzer.results_dir / 'human-human'
        hai_dir = analyzer.results_dir / 'human-ai'
        
        hh_results = None
        hai_results = None
        
        if args.force_rerun:
            logger.info("--force-rerun specified, will regenerate all results")
        else:
            # Look for existing human-human results
            if hh_dir.exists():
                hh_files = sorted(hh_dir.glob(f"similarity_human-human_{criterion_safe}_*.json"), 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                if hh_files:
                    logger.info(f"Loading existing human-human results from: {hh_files[0].name}")
                    with open(hh_files[0], 'r', encoding='utf-8') as f:
                        hh_data = json.load(f)
                        hh_results = hh_data.get('results', [])
                    logger.info(f"  Loaded {len(hh_results)} human-human comparisons")
            
            # Look for existing human-AI results
            if hai_dir.exists():
                hai_files = sorted(hai_dir.glob(f"similarity_human-ai_{criterion_safe}_*.json"),
                                 key=lambda x: x.stat().st_mtime, reverse=True)
                if hai_files:
                    logger.info(f"Loading existing human-AI results from: {hai_files[0].name}")
                    with open(hai_files[0], 'r', encoding='utf-8') as f:
                        hai_data = json.load(f)
                        hai_results = hai_data.get('results', [])
                    logger.info(f"  Loaded {len(hai_results)} human-AI comparisons")
        
        # If results not found, run the analysis
        if hh_results is None:
            logger.info("\n--- Running Human-Human Comparison (no existing results found) ---")
            hh_results = analyzer.analyze_all_reviewer_pairs(
                criterion=criterion,
                year_filter=args.year
            )
            analyzer.save_results(hh_results, criterion, comparison_type='human-human')
            logger.info(f"Human-Human: {len(hh_results)} comparisons")
        
        if hai_results is None:
            logger.info("\n--- Running Human-AI Comparison (no existing results found) ---")
            hai_results = analyzer.analyze_all_human_ai_pairs(
                criterion=criterion,
                year_filter=args.year,
                ai_source_filter=args.ai_source
            )
            analyzer.save_results(hai_results, criterion, comparison_type='human-ai')
            logger.info(f"Human-AI: {len(hai_results)} comparisons")
        
        # Run statistical comparison
        logger.info("\n--- Statistical Comparison ---")
        comparison_results = compare_human_human_vs_human_ai(
            hh_results, 
            hai_results, 
            analyzer.results_dir
        )
        
        # Print summary
        print_comparison_summary(comparison_results)
        
        logger.info("\nComparison analysis complete!")
        return
    
    if args.compare_type == 'human-ai':
        # Human vs AI comparison
        logger.info("=== Human vs AI Review Comparison ===")
        comparison_type = 'human-ai'
        output_dir = analyzer.results_dir / comparison_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.criterion == 'all':
            # Analyze all criteria
            logger.info("Analyzing all criteria (Human vs AI)...")
            all_results = analyzer.analyze_all_human_ai_criteria(
                year_filter=args.year,
                ai_source_filter=args.ai_source
            )
            
            # Save each criterion's results and create visualizations
            for criterion, results in all_results.items():
                analyzer.save_results(
                    results, 
                    criterion, 
                    output_filename=None,
                    comparison_type=comparison_type
                )
                
                # Create visualization for this criterion
                analyzer.create_visualization(results, criterion, comparison_type, output_dir)
                analyzer.create_per_proposal_heatmap(results, criterion, comparison_type, output_dir)
                
                # Print summary
                print(f"\n=== Summary (Human vs AI): {analyzer.CRITERION_NAMES.get(criterion, criterion)} ===")
                summary = analyzer.generate_summary_statistics(results)
                for metric, stats in summary.items():
                    if isinstance(stats, dict) and 'mean' in stats and stats['mean'] is not None:
                        print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
                    elif metric == 'sentiment_agreement_pct':
                        print(f"  {metric}: agree={stats['agree']:.1f}%, partial={stats['partial']:.1f}%, disagree={stats['disagree']:.1f}%")
            
            # Create combined visualization comparing all criteria
            analyzer.create_combined_visualization(all_results, comparison_type, output_dir)
            
        else:
            # Analyze single criterion
            logger.info(f"Analyzing criterion (Human vs AI): {args.criterion}")
            results = analyzer.analyze_all_human_ai_pairs(
                criterion=args.criterion,
                year_filter=args.year,
                ai_source_filter=args.ai_source
            )
            
            # Save results
            analyzer.save_results(
                results, 
                args.criterion, 
                output_filename=args.output,
                comparison_type=comparison_type
            )
            
            # Create visualizations
            analyzer.create_visualization(results, args.criterion, comparison_type, output_dir)
            analyzer.create_per_proposal_heatmap(results, args.criterion, comparison_type, output_dir)
            
            # Print summary
            print(f"\n=== Summary Statistics (Human vs AI) ===")
            summary = analyzer.generate_summary_statistics(results)
            for metric, stats in summary.items():
                if isinstance(stats, dict) and 'mean' in stats and stats['mean'] is not None:
                    print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['n']}")
                elif metric == 'sentiment_agreement_pct':
                    print(f"  {metric}: agree={stats['agree']:.1f}%, partial={stats['partial']:.1f}%, disagree={stats['disagree']:.1f}%")
    
    else:
        # Human vs Human comparison (original behavior)
        logger.info("=== Human vs Human Review Comparison ===")
        comparison_type = 'human-human'
        output_dir = analyzer.results_dir / comparison_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.criterion == 'all':
            # Analyze all criteria
            logger.info("Analyzing all criteria...")
            all_results = analyzer.analyze_all_criteria(year_filter=args.year)
            
            # Save each criterion's results and create visualizations
            for criterion, results in all_results.items():
                analyzer.save_results(
                    results, 
                    criterion,
                    output_filename=None,
                    comparison_type=comparison_type
                )
                
                # Create visualization for this criterion
                analyzer.create_visualization(results, criterion, comparison_type, output_dir)
                analyzer.create_per_proposal_heatmap(results, criterion, comparison_type, output_dir)
                
                # Print summary
                print(f"\n=== Summary: {analyzer.CRITERION_NAMES.get(criterion, criterion)} ===")
                summary = analyzer.generate_summary_statistics(results)
                for metric, stats in summary.items():
                    if isinstance(stats, dict) and 'mean' in stats and stats['mean'] is not None:
                        print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
                    elif metric == 'sentiment_agreement_pct':
                        print(f"  {metric}: agree={stats['agree']:.1f}%, partial={stats['partial']:.1f}%, disagree={stats['disagree']:.1f}%")
            
            # Create combined visualization comparing all criteria
            analyzer.create_combined_visualization(all_results, comparison_type, output_dir)
            
        else:
            # Analyze single criterion
            logger.info(f"Analyzing criterion: {args.criterion}")
            results = analyzer.analyze_all_reviewer_pairs(
                criterion=args.criterion,
                year_filter=args.year
            )
            
            # Save results
            analyzer.save_results(
                results, 
                args.criterion, 
                output_filename=args.output,
                comparison_type=comparison_type
            )
            
            # Create visualizations
            analyzer.create_visualization(results, args.criterion, comparison_type, output_dir)
            analyzer.create_per_proposal_heatmap(results, args.criterion, comparison_type, output_dir)
            
            # Print summary
            print(f"\n=== Summary Statistics ===")
            summary = analyzer.generate_summary_statistics(results)
            for metric, stats in summary.items():
                if isinstance(stats, dict) and 'mean' in stats and stats['mean'] is not None:
                    print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['n']}")
                elif metric == 'sentiment_agreement_pct':
                    print(f"  {metric}: agree={stats['agree']:.1f}%, partial={stats['partial']:.1f}%, disagree={stats['disagree']:.1f}%")
    
    logger.info("Reviewer similarity analysis complete!")


if __name__ == "__main__":
    main()
