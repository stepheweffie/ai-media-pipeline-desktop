"""
Configuration module for Media AI Pipeline
Integrates with desktop screen recordings and screenshots
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Airtable Configuration (optional - for storing analysis results)
AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
AIRTABLE_BASE_ID = os.getenv('MEDIA_AIRTABLE_BASE_ID')  # New base for media files

# File Paths
DESKTOP_PATH = "/Users/savantlab/Desktop"
PNG_PATH = "/Users/savantlab/png"  # Screenshots directory
MP4_PATH = "/Users/savantlab/mp4"  # Screen recordings directory

# Analysis Configuration
VISION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-4"
MAX_TOKENS = 4096

# Media Analysis Settings
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov']
MAX_FILE_SIZE_MB = 50

# Airtable Schema for Media Files
MEDIA_TABLES = {
    'SCREENSHOTS': 'Screenshots',
    'RECORDINGS': 'Screen Recordings',
    'ANALYSIS': 'Media Analysis',
    'TAGS': 'Tags'
}

# Analysis Categories
CONTENT_CATEGORIES = [
    'code',
    'documentation', 
    'design',
    'email',
    'browser',
    'terminal',
    'ide',
    'meeting',
    'presentation',
    'error',
    'configuration',
    'other'
]

# OCR and Vision Settings
EXTRACT_TEXT = True
EXTRACT_UI_ELEMENTS = True
DETECT_CODE = True
DETECT_ERRORS = True

def validate_config():
    """Validate that required configuration is present"""
    missing = []
    if not OPENAI_API_KEY:
        missing.append('OPENAI_API_KEY')
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Check if directories exist
    if not os.path.exists(PNG_PATH):
        print(f"Warning: PNG directory not found at {PNG_PATH}")
    
    if not os.path.exists(MP4_PATH):
        print(f"Warning: MP4 directory not found at {MP4_PATH}")
    
    return True
