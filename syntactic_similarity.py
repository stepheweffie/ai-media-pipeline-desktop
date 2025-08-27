#!/usr/bin/env python3
"""
Syntactic Similarity Module for Media AI Pipeline
Provides cosine similarity calculations for syntactic/structural content analysis
"""
import logging
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import ast
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntacticSimilarityEngine:
    def __init__(self):
        """Initialize the syntactic similarity engine"""
        # Code pattern extractors
        self.code_patterns = {
            'python': {
                'keywords': ['def', 'class', 'if', 'for', 'while', 'try', 'except', 'import', 'from'],
                'operators': ['=', '==', '!=', '+', '-', '*', '/', '%', '&', '|'],
                'structures': ['(', ')', '[', ']', '{', '}', ':', ';']
            },
            'javascript': {
                'keywords': ['function', 'var', 'let', 'const', 'if', 'for', 'while', 'try', 'catch'],
                'operators': ['=', '==', '===', '!=', '!==', '+', '-', '*', '/', '%'],
                'structures': ['(', ')', '[', ']', '{', '}', ';']
            },
            'general': {
                'keywords': ['function', 'class', 'method', 'variable', 'if', 'else', 'loop'],
                'operators': ['=', '==', '+', '-', '*', '/'],
                'structures': ['(', ')', '[', ']', '{', '}']
            }
        }
        
        # UI element patterns
        self.ui_patterns = {
            'elements': ['button', 'menu', 'dialog', 'window', 'input', 'form', 'text', 'image'],
            'actions': ['click', 'select', 'type', 'scroll', 'drag', 'drop', 'hover'],
            'states': ['active', 'disabled', 'hidden', 'selected', 'focused', 'loading']
        }
        
        # Layout patterns
        self.layout_patterns = {
            'positions': ['top', 'bottom', 'left', 'right', 'center', 'corner'],
            'arrangements': ['grid', 'list', 'table', 'sidebar', 'header', 'footer'],
            'spacing': ['margin', 'padding', 'gap', 'space', 'aligned']
        }
        
        logger.info("Syntactic Similarity Engine initialized")
    
    def extract_code_features(self, code_text: str, language: str = 'general') -> Dict[str, int]:
        """Extract syntactic features from code text"""
        if not code_text:
            return {}
        
        features = {}
        patterns = self.code_patterns.get(language, self.code_patterns['general'])
        
        # Convert to lowercase for case-insensitive matching
        code_lower = code_text.lower()
        
        # Count keywords
        for keyword in patterns['keywords']:
            # Use word boundaries to avoid partial matches
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', code_lower))
            if count > 0:
                features[f'keyword_{keyword}'] = count
        
        # Count operators
        for operator in patterns['operators']:
            count = code_text.count(operator)
            if count > 0:
                features[f'operator_{operator}'] = count
        
        # Count structural elements
        for structure in patterns['structures']:
            count = code_text.count(structure)
            if count > 0:
                features[f'structure_{structure}'] = count
        
        # Extract additional patterns
        features.update(self._extract_code_patterns(code_text))
        
        return features
    
    def _extract_code_patterns(self, code_text: str) -> Dict[str, int]:
        """Extract additional code patterns"""
        patterns = {}
        
        # Indentation levels (approximate)
        lines = code_text.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.append(leading_spaces // 4)  # Assume 4-space indentation
        
        if indent_levels:
            patterns['max_indent_level'] = max(indent_levels)
            patterns['avg_indent_level'] = sum(indent_levels) / len(indent_levels)
        
        # Function/method definitions
        patterns['function_defs'] = len(re.findall(r'\bdef\s+\w+\s*\(', code_text, re.IGNORECASE))
        patterns['class_defs'] = len(re.findall(r'\bclass\s+\w+', code_text, re.IGNORECASE))
        
        # Comments
        patterns['line_comments'] = len(re.findall(r'#.*$', code_text, re.MULTILINE))
        patterns['block_comments'] = len(re.findall(r'/\*.*?\*/', code_text, re.DOTALL))
        
        # String literals
        patterns['single_quotes'] = code_text.count("'")
        patterns['double_quotes'] = code_text.count('"')
        
        # Control flow complexity (approximate)
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally']
        patterns['control_flow_count'] = sum(
            len(re.findall(r'\b' + keyword + r'\b', code_text, re.IGNORECASE))
            for keyword in control_keywords
        )
        
        return patterns
    
    def extract_ui_features(self, ui_text: str) -> Dict[str, int]:
        """Extract UI/visual features from text descriptions"""
        if not ui_text:
            return {}
        
        features = {}
        ui_lower = ui_text.lower()
        
        # Count UI elements
        for element in self.ui_patterns['elements']:
            count = len(re.findall(r'\b' + re.escape(element) + r'\b', ui_lower))
            if count > 0:
                features[f'ui_element_{element}'] = count
        
        # Count actions
        for action in self.ui_patterns['actions']:
            count = len(re.findall(r'\b' + re.escape(action) + r'\b', ui_lower))
            if count > 0:
                features[f'ui_action_{action}'] = count
        
        # Count states
        for state in self.ui_patterns['states']:
            count = len(re.findall(r'\b' + re.escape(state) + r'\b', ui_lower))
            if count > 0:
                features[f'ui_state_{state}'] = count
        
        # Layout features
        for position in self.layout_patterns['positions']:
            count = len(re.findall(r'\b' + re.escape(position) + r'\b', ui_lower))
            if count > 0:
                features[f'layout_position_{position}'] = count
        
        for arrangement in self.layout_patterns['arrangements']:
            count = len(re.findall(r'\b' + re.escape(arrangement) + r'\b', ui_lower))
            if count > 0:
                features[f'layout_arrangement_{arrangement}'] = count
        
        # Color patterns
        color_patterns = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'orange', 'purple']
        for color in color_patterns:
            count = len(re.findall(r'\b' + color + r'\b', ui_lower))
            if count > 0:
                features[f'color_{color}'] = count
        
        return features
    
    def extract_structural_features(self, content: str, content_type: str = 'general') -> np.ndarray:
        """Extract structural features and return as vector"""
        features = {}
        
        if content_type == 'code':
            # Try to detect language
            language = self._detect_code_language(content)
            features.update(self.extract_code_features(content, language))
        
        elif content_type == 'ui':
            features.update(self.extract_ui_features(content))
        
        else:
            # General structural analysis
            features.update(self.extract_code_features(content))
            features.update(self.extract_ui_features(content))
        
        # Convert to feature vector
        return self._features_to_vector(features)
    
    def _detect_code_language(self, code: str) -> str:
        """Simple language detection based on patterns"""
        code_lower = code.lower()
        
        # Python indicators
        python_indicators = ['def ', 'import ', 'from ', 'print(', '__', 'elif']
        python_score = sum(1 for indicator in python_indicators if indicator in code_lower)
        
        # JavaScript indicators
        js_indicators = ['function', 'var ', 'let ', 'const ', 'console.log', '=>']
        js_score = sum(1 for indicator in js_indicators if indicator in code_lower)
        
        if python_score > js_score:
            return 'python'
        elif js_score > python_score:
            return 'javascript'
        else:
            return 'general'
    
    def _features_to_vector(self, features: Dict[str, int], max_features: int = 100) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""
        if not features:
            return np.zeros(max_features)
        
        # Create a consistent feature ordering
        feature_names = sorted(features.keys())
        
        # Create vector with top features
        vector_size = min(len(feature_names), max_features)
        vector = np.zeros(vector_size)
        
        for i, feature_name in enumerate(feature_names[:vector_size]):
            vector[i] = features[feature_name]
        
        # Normalize to unit vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def calculate_syntactic_similarity(self, content1: str, content2: str, 
                                     content_type: str = 'general') -> float:
        """Calculate cosine similarity based on syntactic features"""
        try:
            vector1 = self.extract_structural_features(content1, content_type)
            vector2 = self.extract_structural_features(content2, content_type)
            
            # Ensure vectors are same length
            min_len = min(len(vector1), len(vector2))
            if min_len == 0:
                return 0.0
            
            vector1 = vector1[:min_len]
            vector2 = vector2[:min_len]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                vector1.reshape(1, -1),
                vector2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating syntactic similarity: {e}")
            return 0.0
    
    def find_syntactically_similar_content(self, query_content: str, content_items: List[Dict],
                                         content_type: str = 'general',
                                         similarity_threshold: float = 0.6,
                                         top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Find content items with similar syntactic structure"""
        if not content_items:
            return []
        
        query_vector = self.extract_structural_features(query_content, content_type)
        similarities = []
        
        for item in content_items:
            # Extract content from item
            item_content = self._extract_structural_content(item)
            if not item_content:
                continue
            
            item_vector = self.extract_structural_features(item_content, content_type)
            
            # Ensure vectors are same length
            min_len = min(len(query_vector), len(item_vector))
            if min_len == 0:
                continue
            
            query_vec = query_vector[:min_len]
            item_vec = item_vector[:min_len]
            
            similarity = cosine_similarity(
                query_vec.reshape(1, -1),
                item_vec.reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                similarities.append((item, float(similarity)))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _extract_structural_content(self, item: Dict) -> str:
        """Extract structural content from media analysis item"""
        content_parts = []
        
        if 'analysis' in item:
            analysis = item['analysis']
            
            # Prioritize OCR/detected text for structural analysis
            if 'detected_text' in analysis:
                content_parts.append(analysis['detected_text'])
            if 'ocr_text' in analysis:
                content_parts.append(analysis['ocr_text'])
            
            # Include other text content
            if 'content_description' in analysis:
                content_parts.append(analysis['content_description'])
            if 'summary' in analysis:
                content_parts.append(analysis['summary'])
        
        return ' '.join(content_parts)
    
    def analyze_code_complexity(self, code_content: str) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        if not code_content:
            return {}
        
        features = self.extract_code_features(code_content)
        
        # Calculate complexity metrics
        cyclomatic_complexity = features.get('control_flow_count', 0)
        nesting_depth = features.get('max_indent_level', 0)
        function_count = features.get('function_defs', 0)
        class_count = features.get('class_defs', 0)
        
        # Lines of code
        lines = [line.strip() for line in code_content.split('\n') if line.strip()]
        loc = len(lines)
        
        return {
            'lines_of_code': loc,
            'cyclomatic_complexity': cyclomatic_complexity,
            'nesting_depth': nesting_depth,
            'function_count': function_count,
            'class_count': class_count,
            'comment_ratio': (features.get('line_comments', 0) + features.get('block_comments', 0)) / max(loc, 1),
            'complexity_score': (cyclomatic_complexity + nesting_depth) / max(loc, 1)
        }
    
    def compare_ui_structures(self, ui_content1: str, ui_content2: str) -> Dict[str, Any]:
        """Compare UI structures and return detailed analysis"""
        features1 = self.extract_ui_features(ui_content1)
        features2 = self.extract_ui_features(ui_content2)
        
        # Find common and different elements
        all_features = set(features1.keys()) | set(features2.keys())
        common_features = set(features1.keys()) & set(features2.keys())
        
        similarity_score = self.calculate_syntactic_similarity(ui_content1, ui_content2, 'ui')
        
        return {
            'similarity_score': similarity_score,
            'common_elements': len(common_features),
            'total_unique_elements': len(all_features),
            'structural_overlap': len(common_features) / len(all_features) if all_features else 0,
            'features_content1': features1,
            'features_content2': features2,
            'common_feature_names': list(common_features)
        }
    
    def find_structural_patterns(self, content_items: List[Dict]) -> Dict[str, Any]:
        """Find common structural patterns across content items"""
        if not content_items:
            return {}
        
        all_features = []
        valid_items = []
        
        # Extract features from all items
        for item in content_items:
            content = self._extract_structural_content(item)
            if content:
                features = {}
                features.update(self.extract_code_features(content))
                features.update(self.extract_ui_features(content))
                
                if features:  # Only include items with features
                    all_features.append(features)
                    valid_items.append(item)
        
        if not all_features:
            return {}
        
        # Find most common features
        feature_counts = {}
        for features in all_features:
            for feature, count in features.items():
                if feature not in feature_counts:
                    feature_counts[feature] = []
                feature_counts[feature].append(count)
        
        # Calculate statistics for each feature
        feature_stats = {}
        for feature, counts in feature_counts.items():
            if len(counts) > 1:  # Only features appearing in multiple items
                feature_stats[feature] = {
                    'frequency': len(counts) / len(all_features),
                    'avg_count': sum(counts) / len(counts),
                    'max_count': max(counts),
                    'appears_in_items': len(counts)
                }
        
        # Find dominant patterns
        dominant_features = sorted(
            feature_stats.items(),
            key=lambda x: x[1]['frequency'] * x[1]['avg_count'],
            reverse=True
        )[:10]
        
        return {
            'total_items_analyzed': len(valid_items),
            'unique_features_found': len(feature_stats),
            'dominant_patterns': dominant_features,
            'pattern_summary': {
                name: stats for name, stats in dominant_features
            }
        }

def main():
    """Example usage"""
    engine = SyntacticSimilarityEngine()
    
    # Example code snippets
    code1 = """
def calculate_similarity(text1, text2):
    vector1 = get_vector(text1)
    vector2 = get_vector(text2)
    return cosine_similarity(vector1, vector2)
"""
    
    code2 = """
function calculateDistance(str1, str2) {
    const vec1 = getVector(str1);
    const vec2 = getVector(str2);
    return euclideanDistance(vec1, vec2);
}
"""
    
    code3 = """
def process_data(data):
    for item in data:
        if item.valid:
            result = transform(item)
            save(result)
"""
    
    print("🔧 Syntactic Similarity Examples:")
    
    # Compare code structures
    sim1 = engine.calculate_syntactic_similarity(code1, code2, 'code')
    sim2 = engine.calculate_syntactic_similarity(code1, code3, 'code')
    sim3 = engine.calculate_syntactic_similarity(code2, code3, 'code')
    
    print(f"  Python vs JavaScript (similar structure): {sim1:.3f}")
    print(f"  Python similarity func vs loop func: {sim2:.3f}")
    print(f"  JavaScript vs Python loop: {sim3:.3f}")
    
    # Code complexity analysis
    print("\n📊 Code Complexity Analysis:")
    complexity = engine.analyze_code_complexity(code1)
    for metric, value in complexity.items():
        print(f"  {metric}: {value}")
    
    # UI structure comparison
    print("\n🖥️  UI Structure Comparison:")
    ui1 = "button in top right corner with blue background"
    ui2 = "red button positioned at bottom left of window"
    ui_comparison = engine.compare_ui_structures(ui1, ui2)
    print(f"  UI Similarity: {ui_comparison['similarity_score']:.3f}")
    print(f"  Common Elements: {ui_comparison['common_elements']}")

if __name__ == "__main__":
    main()
