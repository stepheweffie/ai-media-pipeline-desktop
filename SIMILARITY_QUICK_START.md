# 🚀 Similarity Features Quick Start Guide

The Media AI Pipeline now includes advanced cosine similarity capabilities for both semantic and syntactic analysis. Here's how to use the new bells and whistles:

## ✨ New Features Added

### 1. **Semantic Similarity Engine** (`semantic_similarity.py`)
- Uses OpenAI embeddings for meaning-based similarity
- Fallback to TF-IDF when OpenAI is unavailable
- Content categorization (code, error, UI, browser, etc.)
- Semantic clustering and batch comparison

### 2. **Syntactic Similarity Engine** (`syntactic_similarity.py`)
- Structural pattern analysis for code and UI elements
- Code complexity metrics (cyclomatic complexity, nesting depth)
- Programming language detection
- UI element and layout pattern recognition

### 3. **Enhanced AI Query Interface** (`ai_query_interface.py`)
- Intelligent search method selection (similarity-first)
- Hybrid search combining semantic + syntactic
- Post-ranking of traditional results using similarity
- New query methods: `semantic_search()`, `syntactic_search()`, `hybrid_search()`

### 4. **Similarity Visualization** (`similarity_visualization.py`)
- Comprehensive reporting and explanations
- Score interpretation for humans
- Data export capabilities

## 🔧 Quick Usage Examples

### Interactive Mode
```bash
python media_pipeline.py interactive
```

Then try these queries:
- "Find screenshots with Python code"
- "Show me terminal error messages"
- "Look for UI design mockups"
- "Search for browser debugging sessions"

### Programmatic Usage
```python
from ai_query_interface import AIQueryInterface

# Initialize with similarity engines
interface = AIQueryInterface()

# Semantic search
results = interface.semantic_search(
    "Python function definition", 
    media_type="screenshots", 
    top_k=10
)

# Syntactic search for code patterns
results = interface.syntactic_search(
    "def foo(x): return x*2", 
    content_type="code",
    top_k=5
)

# Hybrid search (best of both worlds)
results = interface.hybrid_search(
    "VSCode debugging breakpoint", 
    top_k=10
)
```

### Generate Similarity Reports
```python
from similarity_visualization import SimilarityVisualizer

viz = SimilarityVisualizer()
report = viz.generate_similarity_report(results, "Python coding session")
print(report)

# Explain individual scores
explanation = viz.explain_similarity_score(0.85, "semantic")
print(explanation)
```

## 🎯 How It Works

1. **Query Processing**: The system detects if your query would benefit from similarity search
2. **Method Selection**: Chooses semantic (meaning), syntactic (structure), or hybrid approach
3. **Smart Fallback**: If similarity search fails, falls back to traditional search with similarity ranking
4. **Result Enhancement**: All results include similarity scores and explanations

### Search Method Priority:
1. **Similarity Search First**: For queries like "find code with functions"
2. **Traditional + Ranking**: When similarity search unavailable, traditional results are re-ranked by similarity
3. **Local Fallback**: Works without Airtable using local file analysis

## ⚙️ Configuration

### Required Dependencies (already installed)
- `scikit-learn>=1.3.0` - For cosine similarity calculations
- `matplotlib>=3.7.0` - For visualizations
- `seaborn>=0.12.0` - Enhanced plotting
- `pandas>=2.0.0` - Data manipulation
- `scipy>=1.11.0` - Scientific computing

### Optional Enhancements
Uncomment in `requirements.txt` for even better embeddings:
```
sentence-transformers>=2.2.0  # Alternative to OpenAI embeddings
transformers>=4.21.0          # Hugging Face transformers
```

## 🎨 Similarity Scores Explained

- **0.9-1.0**: Extremely similar (near duplicates)
- **0.8-0.9**: Highly similar (strong matches)
- **0.7-0.8**: Moderately similar (good matches)
- **0.6-0.7**: Somewhat similar (loose matches)
- **0.0-0.6**: Low similarity (different content)

## 🔍 Advanced Features

### Content Categorization
```python
# Automatically categorize screenshots
categories = interface.categorize_media_semantically()
print(f"Found {categories['semantic_clusters']} content clusters")
```

### Structural Analysis
```python
# Find coding patterns
patterns = interface.analyze_structural_patterns()
print(f"Dominant patterns: {patterns['dominant_patterns']}")
```

### Find Similar Items
```python
# Find items similar to a reference
similar = interface.find_similar_to_item(
    reference_item, 
    similarity_type='hybrid',
    top_k=5
)
```

## 💡 Pro Tips

1. **Code Queries**: Use keywords like "function", "class", "error" for better syntactic matching
2. **Semantic Queries**: Use descriptive terms like "debugging session" or "login form design"
3. **Hybrid Approach**: Let the system choose by using natural language queries
4. **Performance**: Similarity search works best with 10-50 pre-analyzed items

## 🚨 Troubleshooting

### No OpenAI API Key
- Semantic similarity falls back to TF-IDF vectors
- Syntactic similarity works independently
- Both provide meaningful results

### No Media Files
- The demo works with mock data
- Add screenshots/recordings to your configured directories
- Run analysis first: `python media_pipeline.py analyze`

### Performance Issues
- Similarity search is optimized for reasonable datasets
- Large collections are automatically limited for performance
- Consider using content filters for better targeting

---

## 🎉 Ready to Use!

Your Media AI Pipeline now has sophisticated similarity analysis capabilities. The system intelligently chooses the best search method and provides rich, explained results.

Try it now:
```bash
python media_pipeline.py interactive
```

Then ask: *"Find screenshots with Python debugging code"*
