# Lecture Transcription & Vectorization Setup

This guide will help you set up and use the lecture processing capabilities for your 24 hours of lecture videos.

## Quick Start

1. **Install dependencies:**
```bash
cd ~/Desktop/media_ai_pipeline
pip install -r requirements.txt
```

2. **Test your setup:**
```bash
python lecture_cli.py stats
```

3. **Process your first video:**
```bash
python lecture_cli.py process --file /path/to/lecture.mp4
```

4. **Search the transcripts:**
```bash
python lecture_cli.py search "machine learning"
```

## Setup Options

### Option 1: Basic Setup (Recommended to start)
- Uses OpenAI Whisper for transcription
- Simple speaker diarization based on pause detection
- Uses ChromaDB for vector storage
- Works entirely locally (except for OpenAI embeddings)

### Option 2: Advanced Setup (Better diarization)
- Requires additional setup for pyannote.audio
- Much better speaker separation
- Requires Hugging Face token

## Processing Your 24 Hours of Lectures

### Step 1: Organize Your Videos
```bash
# Create a dedicated directory
mkdir ~/lectures
# Move all your lecture videos there
```

### Step 2: Process in Batches
```bash
# Start with a test batch (5 most recent videos)
python lecture_cli.py process --directory ~/lectures --limit 5

# Once confirmed working, process everything
python lecture_cli.py process --directory ~/lectures --yes
```

### Step 3: Monitor Progress
The system will log progress and provide estimates. For 24 hours of content:
- Processing time: ~2-6 hours (depending on your hardware)
- Storage needed: ~1-2 GB for vector database
- API costs: ~$15-30 for embeddings

## Usage Examples

### Processing Commands
```bash
# Process a single file
python lecture_cli.py process --file lecture1.mp4

# Process all videos in a directory
python lecture_cli.py process --directory ./lectures

# Process only recent videos (limit to 10)
python lecture_cli.py process --directory ./lectures --limit 10

# Use advanced diarization (requires pyannote setup)
python lecture_cli.py process --directory ./lectures --advanced
```

### Search Commands
```bash
# Basic semantic search
python lecture_cli.py search "neural networks"

# Search with filters
python lecture_cli.py search "optimization" --speaker "Speaker_1"
python lecture_cli.py search "algorithms" --video "lecture_5"

# Get more results
python lecture_cli.py search "machine learning" --limit 20
```

### Export Commands
```bash
# Export all transcripts as text
python lecture_cli.py export all_transcripts.txt

# Export as JSON for further processing
python lecture_cli.py export transcripts.json --format json
```

### Statistics
```bash
# View database statistics
python lecture_cli.py stats
```

## Expected Results

After processing your 24 hours of lectures, you'll have:

1. **Searchable Database**: Semantic search across all lecture content
2. **Speaker Attribution**: Identify who said what and when
3. **Timestamp References**: Jump to exact moments in videos
4. **Exportable Transcripts**: Full text transcripts with speaker diarization
5. **Vector Embeddings**: Semantic similarity search capabilities

## Cost Breakdown

### Processing Costs (one-time):
- **Whisper**: Free (local processing)
- **OpenAI Embeddings**: ~$15-30 for 24 hours of content
- **Storage**: Minimal (1-2 GB locally)

### Search Costs (ongoing):
- **ChromaDB**: Free (local vector database)
- **Semantic Search**: Free after initial embedding

## Advanced Diarization Setup (Optional)

For better speaker separation:

1. **Install pyannote.audio:**
```bash
pip install pyannote.audio torch torchaudio
```

2. **Get Hugging Face token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token
   - Accept the pyannote model license

3. **Add to your .env file:**
```bash
HUGGING_FACE_TOKEN=your_token_here
```

4. **Use advanced processing:**
```bash
python lecture_cli.py process --directory ~/lectures --advanced
```

## Troubleshooting

### Common Issues:

**"FFmpeg not found"**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

**"OpenAI API key not found"**
- Check your `.env` file has `OPENAI_API_KEY=your_key_here`

**"Model loading failed"**
- Whisper models are large (~1-3GB). Ensure sufficient disk space
- First run will download the model

**"Out of memory"**
- Try smaller Whisper model: edit `lecture_processor.py`, change `model_size = "base"`
- Process videos in smaller batches

### Performance Tips:

1. **Start Small**: Test with 1-2 videos first
2. **Use SSD**: Faster storage significantly improves processing speed  
3. **Adequate RAM**: 8GB+ recommended for large-v3 model
4. **Batch Processing**: Process during off-hours, it's CPU intensive

## File Structure After Setup

```
media_ai_pipeline/
├── lecture_processor.py          # Core processing logic
├── lecture_cli.py               # Command-line interface
├── lecture_vectors/             # Vector database (created automatically)
│   ├── chroma.sqlite3          # ChromaDB storage
│   └── ...
├── requirements.txt            # Updated dependencies
└── LECTURE_SETUP.md           # This guide
```

## Next Steps

1. **Process a test video** to verify everything works
2. **Start with recent lectures** (most relevant content first)
3. **Build your search queries** as you process more content
4. **Export transcripts** for backup or further analysis

## Integration with Existing Pipeline

This lecture processor integrates seamlessly with your existing media AI pipeline:
- Same configuration (uses your existing `.env`)
- Same OpenAI API key
- Complementary to screenshot/video analysis
- Can be used alongside existing features

Happy transcribing! 🎓📝
