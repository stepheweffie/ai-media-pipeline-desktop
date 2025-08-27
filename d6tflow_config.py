"""
d6tflow Configuration for Media AI Pipeline
Sets up paths, parameters, and global configuration for the workflow pipeline.
"""
import os
import d6tflow
import config
from pathlib import Path

# =============================================================================
# d6tflow Configuration
# =============================================================================

# Set up d6tflow paths
WORKFLOW_BASE_DIR = Path(__file__).parent
WORKFLOW_DATA_DIR = WORKFLOW_BASE_DIR / "d6tflow_data"
WORKFLOW_OUTPUT_DIR = WORKFLOW_BASE_DIR / "workflow_outputs"

# Create directories if they don't exist
WORKFLOW_DATA_DIR.mkdir(exist_ok=True)
WORKFLOW_OUTPUT_DIR.mkdir(exist_ok=True)

# Configure d6tflow settings
d6tflow.settings.dir_data = str(WORKFLOW_DATA_DIR)
d6tflow.settings.log_level = 'INFO'

# Enable task caching for better performance
d6tflow.settings.check_dependencies = True

# =============================================================================
# Pipeline Configuration
# =============================================================================

class PipelineConfig:
    """Centralized configuration for the media AI pipeline workflows"""
    
    # Source directories
    PNG_SOURCE_DIR = config.PNG_PATH
    MP4_SOURCE_DIR = config.MP4_PATH
    
    # Output directories
    WORKFLOW_BASE_DIR = WORKFLOW_BASE_DIR
    DATA_DIR = WORKFLOW_DATA_DIR
    OUTPUT_DIR = WORKFLOW_OUTPUT_DIR
    
    # Analysis limits (can be overridden by task parameters)
    DEFAULT_SCREENSHOT_LIMIT = 10
    DEFAULT_VIDEO_LIMIT = 5
    DEFAULT_LECTURE_LIMIT = 3
    
    # File processing settings
    MAX_FILE_SIZE_MB = config.MAX_FILE_SIZE_MB
    SUPPORTED_IMAGE_FORMATS = config.SUPPORTED_IMAGE_FORMATS
    SUPPORTED_VIDEO_FORMATS = config.SUPPORTED_VIDEO_FORMATS
    
    # AI Model settings
    VISION_MODEL = config.VISION_MODEL
    TEXT_MODEL = config.TEXT_MODEL
    MAX_TOKENS = config.MAX_TOKENS
    
    # API Keys and external services
    OPENAI_API_KEY = config.OPENAI_API_KEY
    AIRTABLE_API_KEY = config.AIRTABLE_API_KEY
    AIRTABLE_BASE_ID = config.AIRTABLE_BASE_ID
    HUGGING_FACE_TOKEN = config.HUGGING_FACE_TOKEN
    
    # Content categories
    CONTENT_CATEGORIES = config.CONTENT_CATEGORIES
    
    # Workflow parameters
    BATCH_SIZE_SCREENSHOTS = 5  # Process screenshots in batches
    BATCH_SIZE_VIDEOS = 2       # Process videos in batches
    ENABLE_AIRTABLE = bool(AIRTABLE_API_KEY and AIRTABLE_BASE_ID)
    
    # Vector database settings
    VECTOR_DB_PATH = str(WORKFLOW_BASE_DIR / "lecture_vectors")
    
    # Parallel processing settings
    MAX_WORKERS = 3  # Number of parallel workers for CPU-intensive tasks

# =============================================================================
# Task Parameter Templates
# =============================================================================

class TaskParams:
    """Parameter templates for different workflow types"""
    
    @staticmethod
    def screenshot_analysis(limit=None, recent_hours=None):
        """Parameters for screenshot analysis tasks"""
        return {
            'limit': limit or PipelineConfig.DEFAULT_SCREENSHOT_LIMIT,
            'recent_hours': recent_hours,
            'source_dir': PipelineConfig.PNG_SOURCE_DIR,
            'batch_size': PipelineConfig.BATCH_SIZE_SCREENSHOTS,
            'enable_airtable': PipelineConfig.ENABLE_AIRTABLE
        }
    
    @staticmethod
    def video_analysis(limit=None, recent_hours=None):
        """Parameters for video analysis tasks"""
        return {
            'limit': limit or PipelineConfig.DEFAULT_VIDEO_LIMIT,
            'recent_hours': recent_hours,
            'source_dir': PipelineConfig.MP4_SOURCE_DIR,
            'batch_size': PipelineConfig.BATCH_SIZE_VIDEOS,
            'enable_airtable': PipelineConfig.ENABLE_AIRTABLE
        }
    
    @staticmethod
    def lecture_processing(limit=None, directory=None, advanced_diarization=False):
        """Parameters for lecture processing tasks"""
        return {
            'limit': limit or PipelineConfig.DEFAULT_LECTURE_LIMIT,
            'source_directory': directory,
            'advanced_diarization': advanced_diarization,
            'vector_db_path': PipelineConfig.VECTOR_DB_PATH,
            'model_size': 'large-v3'  # Whisper model size
        }
    
    @staticmethod
    def full_pipeline(screenshot_limit=None, video_limit=None, lecture_limit=None):
        """Parameters for running the complete pipeline"""
        return {
            'screenshot_limit': screenshot_limit or PipelineConfig.DEFAULT_SCREENSHOT_LIMIT,
            'video_limit': video_limit or PipelineConfig.DEFAULT_VIDEO_LIMIT,
            'lecture_limit': lecture_limit or PipelineConfig.DEFAULT_LECTURE_LIMIT,
            'enable_airtable': PipelineConfig.ENABLE_AIRTABLE,
            'parallel_processing': True
        }

# =============================================================================
# Utility Functions
# =============================================================================

def validate_workflow_config():
    """Validate that the workflow configuration is correct"""
    errors = []
    
    # Check API keys
    if not PipelineConfig.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required")
    
    # Check source directories
    if not os.path.exists(PipelineConfig.PNG_SOURCE_DIR):
        errors.append(f"PNG source directory not found: {PipelineConfig.PNG_SOURCE_DIR}")
    
    if not os.path.exists(PipelineConfig.MP4_SOURCE_DIR):
        errors.append(f"MP4 source directory not found: {PipelineConfig.MP4_SOURCE_DIR}")
    
    # Check write permissions for output directories
    try:
        test_file = PipelineConfig.DATA_DIR / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to data directory {PipelineConfig.DATA_DIR}: {e}")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

def get_task_status_summary():
    """Get a summary of current task statuses in the workflow"""
    try:
        # This would show completed/pending tasks
        # Implementation depends on specific d6tflow task structure
        return {
            'data_dir': str(PipelineConfig.DATA_DIR),
            'output_dir': str(PipelineConfig.OUTPUT_DIR),
            'airtable_enabled': PipelineConfig.ENABLE_AIRTABLE,
            'source_dirs': {
                'png': PipelineConfig.PNG_SOURCE_DIR,
                'mp4': PipelineConfig.MP4_SOURCE_DIR
            }
        }
    except Exception as e:
        return {'error': str(e)}

# =============================================================================
# Initialize Configuration
# =============================================================================

# Validate configuration on import
try:
    validate_workflow_config()
    print("✅ d6tflow configuration validated successfully")
except ValueError as e:
    print(f"⚠️  Configuration warning: {e}")

# Note: d6tflow configuration is handled through task parameters
# The PipelineConfig class provides default values and utilities
