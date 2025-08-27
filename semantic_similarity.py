#!/usr/bin/env python3
"""
Semantic Similarity Module for Media AI Pipeline
Provides cosine similarity calculations for semantic content analysis
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticSimilarityEngine:
    def __init__(self):
        """Initialize the semantic similarity engine"""
        config.validate_config()
        openai.api_key = config.OPENAI_API_KEY
        
        # Initialize TF-IDF vectorizer for fallback text similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,
            lowercase=True
        )
        
        # Cache for embeddings to avoid repeated API calls
        self.embedding_cache = {}
        
        # Semantic categories for content classification
        self.semantic_categories = {
            'code': ['programming', 'coding', 'development', 'syntax', 'function', 'variable', 'class'],
            'error': ['error', 'exception', 'bug', 'failed', 'crash', 'issue', 'problem'],
            'ui': ['interface', 'button', 'menu', 'dialog', 'window', 'form', 'input'],
            'browser': ['website', 'web', 'url', 'browser', 'chrome', 'firefox', 'safari'],
            'terminal': ['command', 'terminal', 'shell', 'bash', 'console', 'cli'],
            'design': ['mockup', 'prototype', 'layout', 'design', 'wireframe', 'sketch'],
            'email': ['email', 'message', 'inbox', 'compose', 'mail', 'gmail'],
            'document': ['document', 'text', 'writing', 'note', 'report', 'article']
        }
        
        logger.info("Semantic Similarity Engine initialized")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text content"""
        if not text or not text.strip():
            return np.zeros(1536)  # OpenAI ada-002 embedding dimension
        
        # Check cache first
        text_key = text.strip().lower()
        if text_key in self.embedding_cache:
            return self.embedding_cache[text_key]
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding
            self.embedding_cache[text_key] = embedding
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to get OpenAI embedding: {e}")
            # Fallback to TF-IDF if OpenAI fails
            return self._get_tfidf_embedding(text)
    
    def _get_tfidf_embedding(self, text: str) -> np.ndarray:
        """Fallback TF-IDF embedding when OpenAI is unavailable"""
        try:
            # Fit and transform single text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            return tfidf_matrix.toarray()[0]
        except Exception as e:
            logger.warning(f"TF-IDF embedding failed: {e}")
            return np.random.random(100)  # Random fallback
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text strings"""
        try:
            embedding1 = self.get_text_embedding(text1)
            embedding2 = self.get_text_embedding(text2)
            
            # Reshape for sklearn cosine_similarity
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def find_similar_content(self, query_text: str, content_items: List[Dict], 
                           similarity_threshold: float = 0.7, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Find content items similar to query text"""
        if not content_items:
            return []
        
        query_embedding = self.get_text_embedding(query_text)
        similarities = []
        
        for item in content_items:
            # Extract text content from item
            item_text = self._extract_searchable_text(item)
            if not item_text:
                continue
            
            item_embedding = self.get_text_embedding(item_text)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                item_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                similarities.append((item, float(similarity)))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _extract_searchable_text(self, item: Dict) -> str:
        """Extract searchable text from a media analysis item"""
        text_parts = []
        
        # Extract from various fields
        if 'analysis' in item:
            analysis = item['analysis']
            if 'summary' in analysis:
                text_parts.append(analysis['summary'])
            if 'content_description' in analysis:
                text_parts.append(analysis['content_description'])
            if 'detected_text' in analysis:
                text_parts.append(analysis['detected_text'])
            if 'ocr_text' in analysis:
                text_parts.append(analysis['ocr_text'])
            if 'transcript' in analysis:
                text_parts.append(analysis['transcript'])
        
        # Extract metadata
        if 'metadata' in item:
            metadata = item['metadata']
            if 'filename' in metadata:
                text_parts.append(metadata['filename'])
            if 'tags' in metadata:
                text_parts.extend(metadata['tags'])
        
        return ' '.join(text_parts)
    
    def categorize_content(self, text: str) -> Dict[str, float]:
        """Categorize content based on semantic similarity to category keywords"""
        if not text:
            return {}
        
        text_embedding = self.get_text_embedding(text)
        category_scores = {}
        
        for category, keywords in self.semantic_categories.items():
            # Create category embedding from keywords
            category_text = ' '.join(keywords)
            category_embedding = self.get_text_embedding(category_text)
            
            # Calculate similarity
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                category_embedding.reshape(1, -1)
            )[0][0]
            
            category_scores[category] = float(similarity)
        
        return category_scores
    
    def find_semantic_clusters(self, content_items: List[Dict], 
                             similarity_threshold: float = 0.8) -> List[List[Dict]]:
        """Group content items into semantic clusters"""
        if not content_items:
            return []
        
        # Get embeddings for all items
        embeddings = []
        valid_items = []
        
        for item in content_items:
            text = self._extract_searchable_text(item)
            if text:
                embedding = self.get_text_embedding(text)
                embeddings.append(embedding)
                valid_items.append(item)
        
        if not embeddings:
            return []
        
        # Calculate similarity matrix
        embeddings_matrix = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Simple clustering based on similarity threshold
        clusters = []
        used_indices = set()
        
        for i in range(len(valid_items)):
            if i in used_indices:
                continue
            
            cluster = [valid_items[i]]
            used_indices.add(i)
            
            for j in range(i + 1, len(valid_items)):
                if j in used_indices:
                    continue
                
                if similarity_matrix[i][j] >= similarity_threshold:
                    cluster.append(valid_items[j])
                    used_indices.add(j)
            
            if len(cluster) > 1:  # Only keep clusters with multiple items
                clusters.append(cluster)
        
        return clusters
    
    def get_content_summary(self, content_items: List[Dict]) -> Dict[str, Any]:
        """Generate semantic summary of content items"""
        if not content_items:
            return {}
        
        # Extract all text content
        all_text = []
        categories = {}
        
        for item in content_items:
            text = self._extract_searchable_text(item)
            if text:
                all_text.append(text)
                
                # Categorize individual item
                item_categories = self.categorize_content(text)
                for category, score in item_categories.items():
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(score)
        
        # Calculate average category scores
        avg_categories = {}
        for category, scores in categories.items():
            avg_categories[category] = sum(scores) / len(scores) if scores else 0.0
        
        # Find dominant category
        dominant_category = max(avg_categories, key=avg_categories.get) if avg_categories else None
        
        # Generate overall summary embedding
        combined_text = ' '.join(all_text[:10])  # Limit to avoid token limits
        summary_embedding = self.get_text_embedding(combined_text)
        
        return {
            'total_items': len(content_items),
            'category_scores': avg_categories,
            'dominant_category': dominant_category,
            'summary_embedding': summary_embedding.tolist(),
            'combined_text_sample': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text
        }
    
    def compare_content_batches(self, batch1: List[Dict], batch2: List[Dict]) -> Dict[str, float]:
        """Compare semantic similarity between two batches of content"""
        summary1 = self.get_content_summary(batch1)
        summary2 = self.get_content_summary(batch2)
        
        if not summary1 or not summary2:
            return {'overall_similarity': 0.0}
        
        # Compare summary embeddings
        emb1 = np.array(summary1['summary_embedding'])
        emb2 = np.array(summary2['summary_embedding'])
        
        overall_similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]
        
        # Compare category distributions
        cat1 = summary1.get('category_scores', {})
        cat2 = summary2.get('category_scores', {})
        
        all_categories = set(cat1.keys()) | set(cat2.keys())
        category_similarities = {}
        
        for category in all_categories:
            score1 = cat1.get(category, 0.0)
            score2 = cat2.get(category, 0.0)
            # Use difference as inverse similarity
            category_similarities[category] = 1.0 - abs(score1 - score2)
        
        return {
            'overall_similarity': float(overall_similarity),
            'category_similarities': category_similarities,
            'batch1_dominant': summary1.get('dominant_category'),
            'batch2_dominant': summary2.get('dominant_category')
        }
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")

def main():
    """Example usage"""
    engine = SemanticSimilarityEngine()
    
    # Example texts for similarity testing
    texts = [
        "Python code with function definition and error handling",
        "JavaScript function showing async await pattern",
        "Terminal command running pytest with coverage",
        "Browser showing Google search results page",
        "Email inbox with unread messages highlighted"
    ]
    
    print("🧠 Semantic Similarity Examples:")
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts[i+1:], i+1):
            similarity = engine.calculate_semantic_similarity(text1, text2)
            print(f"  {i+1}↔{j+1}: {similarity:.3f} | '{text1[:30]}...' ↔ '{text2[:30]}...'")
    
    # Example categorization
    print("\n📊 Content Categorization:")
    test_text = "VS Code editor showing Python function with syntax highlighting and debugging breakpoint"
    categories = engine.categorize_content(test_text)
    for category, score in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {score:.3f}")

if __name__ == "__main__":
    main()
