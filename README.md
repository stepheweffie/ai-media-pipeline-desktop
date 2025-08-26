# Media AI Pipeline 🤖📸🎥

An advanced AI-powered pipeline for analyzing and interacting with your desktop screen recordings and screenshots. This system uses OpenAI's Vision API to understand your media content and provides natural language query capabilities.

## Overview

The Media AI Pipeline transforms your collection of screenshots and screen recordings into a searchable, intelligent knowledge base. It automatically:

- 📸 **Analyzes screenshots** using computer vision to extract content, code, UI elements, and context
- 🎥 **Processes screen recordings** by extracting keyframes and analyzing video content
- 🗄️ **Stores analysis results** in Airtable for persistent, structured data management
- 🔍 **Enables natural language queries** like "Find screenshots with Python code" or "Show me recent error messages"
- 🏷️ **Auto-categorizes content** by type (code, design, documentation, etc.)

## Features

### Core Capabilities
- **Screenshot Analysis**: Extract text, detect code, identify applications, categorize content
- **Video Analysis**: Process screen recordings with keyframe extraction and content understanding
- **Natural Language Interface**: Ask questions about your media in plain English
- **Intelligent Search**: Find content based on visual elements, text, context, or technical details
- **Persistent Storage**: Optional Airtable integration for organized data management
- **Content Categorization**: Automatic tagging and classification of media content

### Supported Content Detection
- Programming languages (Python, JavaScript, etc.)
- Applications (VSCode, Chrome, Terminal, etc.)
- Error messages and debugging sessions
- Documentation and design mockups
- Email and browser content
- Terminal commands and outputs

## Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (required)
- **FFmpeg** (for video processing)
- **Airtable API Key** (optional, for data persistence)

### System Dependencies

**macOS:**
```bash
# Install FFmpeg
brew install ffmpeg

# Or via MacPorts
sudo port install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Installation

1. **Clone or download the project:**
```bash
cd ~/Desktop
git clone <repository-url> media_ai_pipeline
# Or download and extract the ZIP file
```

2. **Install Python dependencies:**
```bash
cd media_ai_pipeline
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp .env.example .env

# Edit with your API keys
nano .env
```

4. **Configure your `.env` file:**
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for Airtable integration)
AIRTABLE_API_KEY=your_airtable_api_key
MEDIA_AIRTABLE_BASE_ID=your_airtable_base_id

# Optional configuration
VISION_MODEL=gpt-4o
TEXT_MODEL=gpt-4
MAX_TOKENS=4096
```

## Quick Start

### 1. Check System Status
```bash
python media_pipeline.py status
```

### 2. Analyze Your Media
```bash
# Analyze recent screenshots and recordings
python media_pipeline.py analyze --screenshots 5 --recordings 2

# Or analyze everything (can be expensive!)
python media_pipeline.py analyze
```

### 3. Start Querying!
```bash
# Interactive mode
python media_pipeline.py interactive

# Or direct queries
python media_pipeline.py query "Find screenshots with Python code"
python media_pipeline.py query "Show me recent error messages"
python media_pipeline.py query "What terminal commands did I run today?"
```

## Usage Examples

### Natural Language Queries

```bash
# Content-based searches
"Find screenshots with code"
"Show me browser recordings"
"Look for error messages"
"Find terminal sessions with git commands"

# Time-based queries
"What did I capture today?"
"Show me yesterday's screenshots"
"Recent recordings with errors"

# Application-specific
"Find VSCode screenshots with Python"
"Show me Chrome browser sessions"
"Terminal recordings from this week"

# Technical searches  
"Screenshots containing API calls"
"Find database error messages"
"Show me design mockups"
```

### Command Line Interface

```bash
# Get statistics
python media_pipeline.py stats

# Search with keywords
python media_pipeline.py search "python code" --type screenshots

# Analyze recent media only
python media_pipeline.py analyze --recent 24

# Clean up old data
python media_pipeline.py cleanup --days 30
```

### Python API Usage

```python
from media_pipeline import MediaAIPipeline

# Initialize pipeline
pipeline = MediaAIPipeline()

# Analyze specific files
results = pipeline.analyze_all_media(screenshot_limit=10)

# Query with natural language
response = pipeline.query_media("Find screenshots with error messages")

# Search programmatically
results = pipeline.search_media("python code", media_type="screenshots")
```

## Directory Structure

```
media_ai_pipeline/
├── config.py                 # Configuration management
├── screenshot_analyzer.py    # Screenshot analysis with Vision API
├── video_analyzer.py         # Video processing and analysis
├── airtable_media_manager.py # Airtable integration
├── ai_query_interface.py     # Natural language query processing
├── media_pipeline.py         # Main orchestrator and CLI
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── README.md               # This file
└── AIRTABLE_SCHEMA.md      # Airtable setup guide
```

## Configuration Options

### Core Settings
- `VISION_MODEL`: OpenAI model for image analysis (default: gpt-4o)
- `TEXT_MODEL`: OpenAI model for text processing (default: gpt-4)
- `MAX_TOKENS`: Maximum tokens per API request (default: 4096)

### File Processing
- `MAX_FILE_SIZE_MB`: Maximum file size for processing (default: 50MB)
- `SUPPORTED_IMAGE_FORMATS`: Image file extensions (default: .png, .jpg, .jpeg)
- `SUPPORTED_VIDEO_FORMATS`: Video file extensions (default: .mp4, .mov)

### Content Categories
The system automatically categorizes content into:
- `code` - Programming/development content
- `documentation` - Docs, wikis, guides
- `design` - UI/UX, mockups, graphics
- `email` - Email applications and content
- `browser` - Web browsers and websites
- `terminal` - Command line interfaces
- `ide` - Integrated development environments
- `meeting` - Video calls, presentations
- `error` - Error messages and debugging
- `configuration` - Settings, config files

## Airtable Integration (Optional)

For persistent data storage and advanced querying:

1. **Create a new Airtable base** called "Media AI Pipeline"

2. **Set up tables** following the schema in `AIRTABLE_SCHEMA.md`:
   - Screenshots table
   - Recordings table  
   - Analysis table (optional)
   - Tags table (optional)

3. **Get your API credentials**:
   - Generate a Personal Access Token in Airtable
   - Copy your Base ID from the API documentation

4. **Update your `.env` file**:
```bash
AIRTABLE_API_KEY=your_personal_access_token
MEDIA_AIRTABLE_BASE_ID=your_base_id
```

## Performance Considerations

### API Costs
- **Screenshot analysis**: ~$0.01-0.05 per image (depending on model and detail level)
- **Video analysis**: ~$0.05-0.25 per video (depending on length and keyframes)
- **Text queries**: ~$0.002-0.01 per query

### Optimization Tips
- Use `--limit` flags to process fewer files during testing
- Analyze recent files only with `--recent` flag
- Consider using GPT-3.5-turbo for lower costs (update `TEXT_MODEL` in config)
- Video processing is more intensive than screenshots

### Storage Requirements
- Local processing requires minimal storage
- Airtable integration stores analysis results permanently
- Raw media files remain in your original directories

## Troubleshooting

### Common Issues

**"No module named 'openai'"**
```bash
pip install -r requirements.txt
```

**"ffmpeg not found"**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

**"Airtable not configured"**
- The system works without Airtable (local-only mode)
- Add Airtable credentials to `.env` for full features

**"File too large" errors**
- Adjust `MAX_FILE_SIZE_MB` in `config.py`
- Or skip large files with size limits

**OpenAI API rate limits**
- Add delays between requests if hitting rate limits
- Consider upgrading your OpenAI plan
- Use GPT-3.5-turbo for higher rate limits

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation
```bash
# Test configuration
python -c "import config; config.validate_config(); print('Config OK')"

# Test individual components
python screenshot_analyzer.py
python video_analyzer.py
```

## Advanced Usage

### Custom Categories
Edit `CONTENT_CATEGORIES` in `config.py` to add your own categories:
```python
CONTENT_CATEGORIES = [
    'code', 'documentation', 'design',
    'custom_category_1', 'custom_category_2'
]
```

### Batch Processing
```python
# Process files in batches to manage API costs
for batch_start in range(0, total_files, 10):
    results = pipeline.analyze_all_media(
        screenshot_limit=10, 
        video_limit=5
    )
```

### Integration with Other Tools
```python
# Export analysis results
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Import to other systems
for result in results:
    # Send to your custom database, API, etc.
    pass
```

## Contributing

This is a foundational system that can be extended in many ways:

- **Add new content detectors** for specific use cases
- **Integrate with other AI models** (Claude, Gemini, etc.)
- **Add export formats** (CSV, JSON, XML)
- **Create web interface** for easier querying
- **Add notification systems** for real-time analysis
- **Implement vector databases** for semantic search

## License

This project is provided as-is for educational and personal use. Please respect OpenAI's usage policies and your Airtable plan limits.

---

## Need Help?

1. Check the `status` command for system health
2. Review the example queries in interactive mode (`help` command)
3. Validate your configuration and API keys
4. Check the logs for detailed error information
5. Start with small batches to test functionality

**Happy analyzing!** 🚀
