# Media AI Pipeline 🤖📸🎥🎓

An advanced AI-powered pipeline for analyzing and interacting with your desktop screen recordings, screenshots, and lecture videos. This system uses OpenAI's Vision API for visual analysis and Whisper for audio transcription, providing comprehensive natural language query capabilities across all your media content.

## Overview

The Media AI Pipeline transforms your collection of screenshots, screen recordings, and lecture videos into a searchable, intelligent knowledge base. It automatically:

- 📸 **Analyzes screenshots** using computer vision to extract content, code, UI elements, and context
- 🎥 **Processes screen recordings** by extracting keyframes and analyzing video content
- 🎓 **Transcribes lecture videos** with speaker diarization and semantic search capabilities
- 🗣️ **Identifies speakers** in audio content and maintains speaker attribution
- 🔍 **Enables natural language queries** like "Find screenshots with Python code" or "What did the professor say about neural networks?"
- 🗄️ **Stores analysis results** in Airtable for persistent, structured data management
- 🔊 **Vector database integration** for semantic search across transcribed content
- 🏷️ **Auto-categorizes content** by type (code, design, documentation, lectures, etc.)

## Features

### Core Capabilities
- **Screenshot Analysis**: Extract text, detect code, identify applications, categorize content
- **Video Analysis**: Process screen recordings with keyframe extraction and content understanding
- **Lecture Transcription**: Full audio transcription with OpenAI Whisper integration
- **Speaker Diarization**: Identify and separate different speakers in lecture content
- **Semantic Search**: Vector-based search across all transcribed content using ChromaDB
- **Natural Language Interface**: Ask questions about your media in plain English
- **Intelligent Search**: Find content based on visual elements, text, context, or technical details
- **Persistent Storage**: Optional Airtable integration for organized data management
- **Content Categorization**: Automatic tagging and classification of media content
- **Bulk Processing**: Handle large batches of content efficiently with progress tracking

### Supported Content Detection
- Programming languages (Python, JavaScript, etc.)
- Applications (VSCode, Chrome, Terminal, etc.)
- Error messages and debugging sessions
- Documentation and design mockups
- Email and browser content
- Terminal commands and outputs
- Lecture content with speaker identification
- Educational discussions and presentations
- Audio content transcription and analysis

## Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (required for visual analysis and embeddings)
- **FFmpeg** (for video and audio processing)
- **Airtable API Key** (optional, for data persistence)
- **Hugging Face Token** (optional, for advanced speaker diarization)

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
git clone https://github.com/stepheweffie/ai-media-pipeline-desktop.git
# Or download and extract the ZIP file
```

2. **Install Python dependencies:**
```bash
cd media_ai_pipeline
pip install -r requirements.txt

# For advanced speaker diarization (optional):
pip install pyannote.audio torch torchaudio
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

# Optional (for advanced speaker diarization)
HUGGING_FACE_TOKEN=your_huggingface_token

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

### 3. Process Lecture Content
```bash
# Check lecture processing setup
python lecture_cli.py stats

# Process a single lecture video
python lecture_cli.py process --file /path/to/lecture.mp4

# Process all lectures in a directory
python lecture_cli.py process --directory ~/lectures --limit 5
```

### 4. Start Querying!
```bash
# Interactive mode for screenshots/recordings
python media_pipeline.py interactive

# Direct queries for visual content
python media_pipeline.py query "Find screenshots with Python code"
python media_pipeline.py query "Show me recent error messages"
python media_pipeline.py query "What terminal commands did I run today?"

# Search lecture transcripts
python lecture_cli.py search "machine learning algorithms"
python lecture_cli.py search "neural networks" --speaker "Speaker_1"
```

## Usage Examples

### Natural Language Queries

```bash
# Visual content searches
"Find screenshots with code"
"Show me browser recordings"
"Look for error messages"
"Find terminal sessions with git commands"

# Lecture content searches
"What did the professor say about neural networks?"
"Find discussions about machine learning algorithms"
"Show me explanations of data structures"
"Search for mentions of Python programming"

# Time-based queries
"What did I capture today?"
"Show me yesterday's screenshots"
"Recent recordings with errors"
"Latest lecture transcripts"

# Application-specific
"Find VSCode screenshots with Python"
"Show me Chrome browser sessions"
"Terminal recordings from this week"

# Technical searches  
"Screenshots containing API calls"
"Find database error messages"
"Show me design mockups"
"Lecture segments about optimization"
```

### Command Line Interface

```bash
# Media pipeline commands
python media_pipeline.py stats
python media_pipeline.py search "python code" --type screenshots
python media_pipeline.py analyze --recent 24
python media_pipeline.py cleanup --days 30

# Lecture processing commands
python lecture_cli.py stats
python lecture_cli.py process --directory ~/lectures --limit 10
python lecture_cli.py search "algorithms" --limit 20
python lecture_cli.py export transcripts.txt

# Utility commands
python setup_airtable.py  # Set up Airtable database
python run_tests.py       # Run comprehensive tests
```

### Python API Usage

```python
from media_pipeline import MediaAIPipeline
from lecture_processor import LectureProcessor

# Initialize pipelines
pipeline = MediaAIPipeline()
lecture_processor = LectureProcessor()

# Analyze visual media
results = pipeline.analyze_all_media(screenshot_limit=10)
response = pipeline.query_media("Find screenshots with error messages")

# Process lecture content
lecture_result = lecture_processor.process_video("/path/to/lecture.mp4")
search_results = lecture_processor.search_lectures("machine learning")

# Get comprehensive statistics
stats = lecture_processor.get_statistics()
```

## Directory Structure

```
media_ai_pipeline/
├── config.py                    # Configuration management
├── screenshot_analyzer.py       # Screenshot analysis with Vision API
├── video_analyzer.py            # Video processing and analysis
├── lecture_processor.py         # Lecture transcription with Whisper
├── lecture_cli.py               # Command-line interface for lectures
├── airtable_media_manager.py    # Airtable integration
├── ai_query_interface.py        # Natural language query processing
├── media_pipeline.py            # Main orchestrator and CLI
├── setup_airtable.py            # Airtable database setup utility
├── run_tests.py                 # Comprehensive testing framework
├── requirements.txt             # Python dependencies
├── lecture_vectors/             # ChromaDB vector database (auto-created)
│   ├── chroma.sqlite3          # Vector storage
│   └── ...
├── __pycache__/                # Python cache files
├── .env.example                # Environment variables template
├── .env                        # Your environment configuration
├── LICENSE                     # MIT License
├── README.md                   # This file
├── LECTURE_SETUP.md            # Detailed lecture processing guide
├── TESTING.md                  # Testing documentation
├── test_*.py                   # Test files
├── d6tflow_*.py                # d6tflow pipeline components
├── D6TFLOW_GUIDE.md            # d6tflow usage documentation
├── D6TFLOW_IMPLEMENTATION_SUMMARY.md # d6tflow technical overview
├── README_D6TFLOW.md           # d6tflow-specific README
└── d6tflow_data/               # d6tflow cache directory (auto-created)
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

## Lecture Processing System 🎓

The lecture processing system is a powerful addition that transforms your educational content into a searchable knowledge base.

### Features
- **Audio Transcription**: Using OpenAI's Whisper for high-accuracy speech-to-text
- **Speaker Diarization**: Identify and separate different speakers (professors, students, etc.)
- **Semantic Search**: Find content based on meaning, not just keywords
- **Vector Database**: ChromaDB integration for persistent, fast search
- **Batch Processing**: Handle multiple lecture videos efficiently
- **Export Options**: Generate text transcripts and structured data

### Quick Start for Lectures

1. **Verify Setup**:
```bash
python lecture_cli.py stats
```

2. **Process Your First Lecture**:
```bash
# Single file
python lecture_cli.py process --file ~/Downloads/lecture1.mp4

# Directory of lectures (limit to 5 for testing)
python lecture_cli.py process --directory ~/lectures --limit 5
```

3. **Search Transcripts**:
```bash
# Basic search
python lecture_cli.py search "machine learning"

# Advanced search with filters
python lecture_cli.py search "algorithms" --speaker "Professor" --limit 10
```

4. **Export Transcripts**:
```bash
# Export all transcripts as text
python lecture_cli.py export all_transcripts.txt

# Export as JSON for further processing
python lecture_cli.py export transcripts.json --format json
```

### Processing Large Lecture Collections

For processing 10+ hours of lecture content:

```bash
# Create dedicated directory
mkdir ~/lectures

# Start with recent lectures (most relevant)
python lecture_cli.py process --directory ~/lectures --limit 10

# Monitor progress and verify results
python lecture_cli.py stats

# Process everything once verified
python lecture_cli.py process --directory ~/lectures --yes
```

### Advanced Speaker Diarization

For better speaker separation, install advanced components:

```bash
# Install pyannote.audio
pip install pyannote.audio torch torchaudio

# Get Hugging Face token from https://huggingface.co/settings/tokens
# Add to .env file:
echo "HUGGING_FACE_TOKEN=your_token_here" >> .env

# Use advanced processing
python lecture_cli.py process --directory ~/lectures --advanced
```

### Expected Results

After processing your lecture collection, you'll have:
- **Searchable Database**: Semantic search across all content
- **Speaker Attribution**: Know who said what and when
- **Timestamp References**: Jump to exact moments in videos
- **Exportable Transcripts**: Full text with speaker labels
- **Cost-Effective Processing**: Only embedding costs (~$15-30 for 24 hours)

### Troubleshooting Lectures

**"Model download failed"**
- Whisper models are large (1-3GB). Ensure good internet connection
- Models download automatically on first use

**"Out of memory"**
- Try smaller model: Edit `lecture_processor.py`, change to `"base"` or `"small"`
- Process videos in smaller batches

**"Audio extraction failed"**
- Ensure FFmpeg is installed and video files are valid
- Check video file isn't corrupted

---

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

## d6tflow Integration 🚀

The Media AI Pipeline now includes a robust d6tflow implementation, transforming it from simple scripts into a sophisticated workflow engine with dependency management, caching, error handling, and scalable processing.

### Key Benefits
- **Automatic dependency management** - Tasks only run when needed
- **Intelligent caching** - Results are cached to avoid re-computation
- **Error resilience** - Failed tasks don't break the entire pipeline
- **Incremental processing** - Only process what has changed
- **Parallel execution** - Run independent tasks concurrently

### Quick Start with d6tflow

```bash
# Check system status
python d6tflow_cli.py status

# Test with screenshots (recommended first step)
python d6tflow_cli.py run --screenshots --limit 3

# Run full pipeline with custom limits
python d6tflow_cli.py run --full \
  --screenshot-limit 10 \
  --video-limit 5 \
  --lecture-limit 2

# Query your media
python d6tflow_cli.py query "screenshots with Python code"
```

For complete documentation on the d6tflow implementation, see [D6TFLOW_GUIDE.md](D6TFLOW_GUIDE.md) and [D6TFLOW_IMPLEMENTATION_SUMMARY.md](D6TFLOW_IMPLEMENTATION_SUMMARY.md).

## Performance Considerations

### API Costs
- **Screenshot analysis**: ~$0.01-0.05 per image (depending on model and detail level)
- **Video analysis**: ~$0.05-0.25 per video (depending on length and keyframes)
- **Lecture transcription**: Free (Whisper runs locally)
- **Lecture embeddings**: ~$0.10-0.30 per hour of content (one-time cost)
- **Text queries**: ~$0.002-0.01 per query

### Processing Times
- **Screenshots**: ~2-5 seconds per image
- **Videos**: ~30-60 seconds per video (depending on length)
- **Lecture transcription**: ~0.25-0.5x real-time (e.g., 1 hour video = 15-30 min processing)
- **Speaker diarization**: Adds 20-50% to transcription time

### Storage Requirements
- **Visual analysis**: Minimal local storage
- **Vector database**: ~50-100MB per hour of lecture content
- **Whisper models**: 1-3GB (downloaded once)
- **Airtable integration**: Stores analysis results permanently
- **Raw media files**: Remain in your original directories

### Optimization Tips
- Use `--limit` flags to process fewer files during testing
- Analyze recent files only with `--recent` flag
- For lectures: Start with smaller Whisper models (base/small) for testing
- Process lectures in batches during off-hours (CPU intensive)
- Consider using GPT-3.5-turbo for lower costs (update `TEXT_MODEL` in config)
- SSD storage recommended for better performance

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

**"Whisper model loading failed"**
```bash
# Ensure sufficient disk space (models are 1-3GB)
# First run downloads the model automatically
# Try smaller model if out of memory: edit lecture_processor.py
# Change: self.model_size = "base"  # instead of "large-v3"
```

**"ChromaDB initialization failed"**
```bash
# Check permissions in lecture_vectors/ directory
# Try removing and recreating: rm -rf lecture_vectors/
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

# Test lecture processing
python lecture_cli.py stats

# Run comprehensive tests
python run_tests.py
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please respect OpenAI's usage policies and your Airtable plan limits when using this software.

---

## Need Help?

1. Check the `status` command for system health
2. Review the example queries in interactive mode (`help` command)
3. Validate your configuration and API keys
4. Check the logs for detailed error information
5. Start with small batches to test functionality

**Happy analyzing!** 🚀
