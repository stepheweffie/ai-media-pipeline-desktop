"""
Airtable Media Manager for Media AI Pipeline
Stores and retrieves media analysis results in Airtable
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from airtable import Airtable
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirtableMediaManager:
    def __init__(self):
        """Initialize the Airtable media manager"""
        if not config.AIRTABLE_API_KEY or not config.AIRTABLE_BASE_ID:
            logger.warning("Airtable credentials not configured - running in local-only mode")
            self.enabled = False
            return
            
        try:
            self.enabled = True
            self.screenshots_table = Airtable(config.AIRTABLE_BASE_ID, config.MEDIA_TABLES['SCREENSHOTS'], api_key=config.AIRTABLE_API_KEY)
            self.recordings_table = Airtable(config.AIRTABLE_BASE_ID, config.MEDIA_TABLES['RECORDINGS'], api_key=config.AIRTABLE_API_KEY)
            self.analysis_table = Airtable(config.AIRTABLE_BASE_ID, config.MEDIA_TABLES['ANALYSIS'], api_key=config.AIRTABLE_API_KEY)
            self.tags_table = Airtable(config.AIRTABLE_BASE_ID, config.MEDIA_TABLES['TAGS'], api_key=config.AIRTABLE_API_KEY)
            logger.info("Successfully connected to Airtable media database")
        except Exception as e:
            logger.error(f"Failed to initialize Airtable connections: {e}")
            self.enabled = False
    
    def save_screenshot_analysis(self, analysis_result: Dict) -> Optional[str]:
        """Save screenshot analysis to Airtable"""
        if not self.enabled:
            logger.info("Airtable not enabled - skipping save")
            return None
            
        try:
            metadata = analysis_result.get('metadata', {})
            analysis = analysis_result.get('analysis', {})
            
            # Prepare data for Screenshots table
            screenshot_data = {
                'Filename': metadata.get('filename', ''),
                'File Path': metadata.get('filepath', ''),
                'File Size MB': metadata.get('file_size_mb', 0),
                'Created At': metadata.get('created_at', ''),
                'Modified At': metadata.get('modified_at', ''),
                'Analyzed At': metadata.get('analyzed_at', ''),
                
                # Analysis fields
                'Content Type': analysis.get('content_type', ''),
                'Primary Application': analysis.get('primary_application', ''),
                'Summary': analysis.get('summary', ''),
                'Extracted Text': analysis.get('extracted_text', ''),
                'Code Detected': analysis.get('code_detected', False),
                'Programming Language': analysis.get('programming_language', ''),
                'Error Detected': analysis.get('error_detected', False),
                'Error Details': analysis.get('error_details', ''),
                'Technical Details': analysis.get('technical_details', ''),
                'Workflow Step': analysis.get('workflow_step', ''),
                
                # JSON fields for complex data
                'UI Elements': json.dumps(analysis.get('ui_elements', [])),
                'Context Tags': json.dumps(analysis.get('context_tags', [])),
                'Searchable Keywords': json.dumps(analysis.get('searchable_keywords', [])),
                'Full Analysis JSON': json.dumps(analysis_result)
            }
            
            # Remove None values
            screenshot_data = {k: v for k, v in screenshot_data.items() if v is not None}
            
            # Check if record already exists
            existing_records = self.screenshots_table.get_all(
                formula=f"{{Filename}} = '{metadata.get('filename', '')}'"
            )
            
            if existing_records:
                # Update existing record
                record_id = existing_records[0]['id']
                screenshot_data['Last Updated'] = datetime.now().isoformat()
                self.screenshots_table.update(record_id, screenshot_data)
                logger.info(f"Updated screenshot record for {metadata.get('filename', '')}")
                return record_id
            else:
                # Create new record
                result = self.screenshots_table.insert(screenshot_data)
                logger.info(f"Created new screenshot record for {metadata.get('filename', '')}")
                return result['id']
                
        except Exception as e:
            logger.error(f"Error saving screenshot analysis to Airtable: {e}")
            return None
    
    def save_video_analysis(self, analysis_result: Dict) -> Optional[str]:
        """Save video analysis to Airtable"""
        if not self.enabled:
            logger.info("Airtable not enabled - skipping save")
            return None
            
        try:
            metadata = analysis_result.get('metadata', {})
            analysis = analysis_result.get('analysis', {})
            keyframes = analysis_result.get('keyframes', [])
            
            # Prepare data for Recordings table
            recording_data = {
                'Filename': metadata.get('filename', ''),
                'File Path': metadata.get('filepath', ''),
                'File Size MB': metadata.get('file_size_mb', 0),
                'Duration Seconds': metadata.get('duration_seconds', 0),
                'Duration Formatted': metadata.get('duration_formatted', ''),
                'Width': metadata.get('width', 0),
                'Height': metadata.get('height', 0),
                'FPS': metadata.get('fps', 0),
                'Codec': metadata.get('codec', ''),
                'Bitrate': metadata.get('bitrate', 0),
                'Created At': metadata.get('created_at', ''),
                'Modified At': metadata.get('modified_at', ''),
                'Analyzed At': metadata.get('analyzed_at', ''),
                
                # Analysis fields
                'Content Type': analysis.get('content_type', ''),
                'Primary Application': analysis.get('primary_application', ''),
                'Summary': analysis.get('summary', ''),
                'Code Detected': analysis.get('code_detected', False),
                'Errors Detected': analysis.get('errors_detected', False),
                'Error Count': analysis.get('error_count', 0),
                'Keyframe Count': analysis.get('keyframe_count', 0),
                
                # JSON fields for complex data
                'Context Tags': json.dumps(analysis.get('context_tags', [])),
                'Technical Details': json.dumps(analysis.get('technical_details', [])),
                'Workflow Steps': json.dumps(analysis.get('workflow_steps', [])),
                'Searchable Keywords': json.dumps(analysis.get('searchable_keywords', [])),
                'Full Analysis JSON': json.dumps(analysis_result),
                'Keyframes JSON': json.dumps(keyframes),
                
                # Stats
                'Total Tokens Used': analysis_result.get('total_tokens_used', 0)
            }
            
            # Remove None values
            recording_data = {k: v for k, v in recording_data.items() if v is not None}
            
            # Check if record already exists
            existing_records = self.recordings_table.get_all(
                formula=f"{{Filename}} = '{metadata.get('filename', '')}'"
            )
            
            if existing_records:
                # Update existing record
                record_id = existing_records[0]['id']
                recording_data['Last Updated'] = datetime.now().isoformat()
                self.recordings_table.update(record_id, recording_data)
                logger.info(f"Updated recording record for {metadata.get('filename', '')}")
                return record_id
            else:
                # Create new record
                result = self.recordings_table.insert(recording_data)
                logger.info(f"Created new recording record for {metadata.get('filename', '')}")
                return result['id']
                
        except Exception as e:
            logger.error(f"Error saving video analysis to Airtable: {e}")
            return None
    
    def search_media(self, query: str, media_type: str = 'all', limit: int = 10) -> List[Dict]:
        """Search media records by content"""
        if not self.enabled:
            logger.warning("Airtable not enabled - cannot search")
            return []
        
        try:
            results = []
            
            # Build search formula for Airtable
            # Note: Airtable search is limited, so we'll do client-side filtering
            
            if media_type in ['all', 'screenshots']:
                screenshot_records = self.screenshots_table.get_all()
                for record in screenshot_records[:50]:  # Limit to avoid API limits
                    fields = record['fields']
                    
                    # Simple text search across relevant fields
                    searchable_text = ' '.join([
                        fields.get('Summary', ''),
                        fields.get('Extracted Text', ''),
                        fields.get('Technical Details', ''),
                        fields.get('Primary Application', ''),
                        fields.get('Content Type', ''),
                        ' '.join(json.loads(fields.get('Searchable Keywords', '[]'))),
                        ' '.join(json.loads(fields.get('Context Tags', '[]')))
                    ]).lower()
                    
                    if query.lower() in searchable_text:
                        record['fields']['media_type'] = 'screenshot'
                        record['fields']['relevance_score'] = searchable_text.count(query.lower())
                        results.append(record)
            
            if media_type in ['all', 'recordings']:
                recording_records = self.recordings_table.get_all()
                for record in recording_records[:50]:  # Limit to avoid API limits
                    fields = record['fields']
                    
                    # Simple text search across relevant fields
                    searchable_text = ' '.join([
                        fields.get('Summary', ''),
                        fields.get('Primary Application', ''),
                        fields.get('Content Type', ''),
                        ' '.join(json.loads(fields.get('Searchable Keywords', '[]'))),
                        ' '.join(json.loads(fields.get('Context Tags', '[]'))),
                        ' '.join(json.loads(fields.get('Technical Details', '[]'))),
                        ' '.join(json.loads(fields.get('Workflow Steps', '[]')))
                    ]).lower()
                    
                    if query.lower() in searchable_text:
                        record['fields']['media_type'] = 'recording'
                        record['fields']['relevance_score'] = searchable_text.count(query.lower())
                        results.append(record)
            
            # Sort by relevance and limit results
            results.sort(key=lambda x: x['fields']['relevance_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching media in Airtable: {e}")
            return []
    
    def get_media_by_content_type(self, content_type: str) -> List[Dict]:
        """Get all media files of a specific content type"""
        if not self.enabled:
            return []
        
        try:
            results = []
            
            # Search screenshots
            screenshot_formula = f"{{Content Type}} = '{content_type}'"
            screenshot_records = self.screenshots_table.get_all(formula=screenshot_formula)
            for record in screenshot_records:
                record['fields']['media_type'] = 'screenshot'
                results.append(record)
            
            # Search recordings
            recording_formula = f"{{Content Type}} = '{content_type}'"
            recording_records = self.recordings_table.get_all(formula=recording_formula)
            for record in recording_records:
                record['fields']['media_type'] = 'recording'
                results.append(record)
            
            # Sort by creation time (newest first)
            results.sort(key=lambda x: x['fields'].get('Created At', ''), reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error getting media by content type: {e}")
            return []
    
    def get_all_media(self, media_type: str = 'all') -> List[Dict]:
        """Get all media records for similarity search"""
        if not self.enabled:
            return []
        
        try:
            results = []
            
            if media_type in ['all', 'screenshots']:
                screenshot_records = self.screenshots_table.get_all()
                for record in screenshot_records:
                    # Convert Airtable record to standard format
                    fields = record['fields']
                    standard_record = {
                        'metadata': {
                            'filename': fields.get('Filename', ''),
                            'filepath': fields.get('File Path', ''),
                            'created': fields.get('Created At', ''),
                            'file_size_mb': fields.get('File Size MB', 0)
                        },
                        'analysis': {
                            'summary': fields.get('Summary', ''),
                            'content_description': fields.get('Summary', ''),
                            'detected_text': fields.get('Extracted Text', ''),
                            'ocr_text': fields.get('Extracted Text', ''),
                            'content_type': fields.get('Content Type', ''),
                            'primary_application': fields.get('Primary Application', '')
                        },
                        'id': record['id'],
                        'media_type': 'screenshot'
                    }
                    results.append(standard_record)
            
            if media_type in ['all', 'recordings']:
                recording_records = self.recordings_table.get_all()
                for record in recording_records:
                    fields = record['fields']
                    standard_record = {
                        'metadata': {
                            'filename': fields.get('Filename', ''),
                            'filepath': fields.get('File Path', ''),
                            'created': fields.get('Created At', ''),
                            'duration_seconds': fields.get('Duration Seconds', 0),
                            'file_size_mb': fields.get('File Size MB', 0)
                        },
                        'analysis': {
                            'summary': fields.get('Summary', ''),
                            'content_description': fields.get('Summary', ''),
                            'content_type': fields.get('Content Type', ''),
                            'primary_application': fields.get('Primary Application', '')
                        },
                        'id': record['id'],
                        'media_type': 'recording'
                    }
                    results.append(standard_record)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting all media: {e}")
            return []
    
    def get_media_stats(self) -> Dict:
        """Get statistics about stored media"""
        if not self.enabled:
            return {'error': 'Airtable not enabled'}
        
        try:
            stats = {
                'screenshots': {
                    'total': 0,
                    'content_types': {},
                    'applications': {},
                    'with_code': 0,
                    'with_errors': 0
                },
                'recordings': {
                    'total': 0,
                    'content_types': {},
                    'applications': {},
                    'total_duration_minutes': 0,
                    'with_code': 0,
                    'with_errors': 0
                }
            }
            
            # Screenshot stats
            screenshot_records = self.screenshots_table.get_all()
            stats['screenshots']['total'] = len(screenshot_records)
            
            for record in screenshot_records:
                fields = record['fields']
                
                # Count content types
                content_type = fields.get('Content Type', 'unknown')
                stats['screenshots']['content_types'][content_type] = \
                    stats['screenshots']['content_types'].get(content_type, 0) + 1
                
                # Count applications
                app = fields.get('Primary Application', 'unknown')
                stats['screenshots']['applications'][app] = \
                    stats['screenshots']['applications'].get(app, 0) + 1
                
                # Count special features
                if fields.get('Code Detected'):
                    stats['screenshots']['with_code'] += 1
                if fields.get('Error Detected'):
                    stats['screenshots']['with_errors'] += 1
            
            # Recording stats
            recording_records = self.recordings_table.get_all()
            stats['recordings']['total'] = len(recording_records)
            
            for record in recording_records:
                fields = record['fields']
                
                # Count content types
                content_type = fields.get('Content Type', 'unknown')
                stats['recordings']['content_types'][content_type] = \
                    stats['recordings']['content_types'].get(content_type, 0) + 1
                
                # Count applications
                app = fields.get('Primary Application', 'unknown')
                stats['recordings']['applications'][app] = \
                    stats['recordings']['applications'].get(app, 0) + 1
                
                # Sum duration
                duration = fields.get('Duration Seconds', 0)
                stats['recordings']['total_duration_minutes'] += duration / 60
                
                # Count special features
                if fields.get('Code Detected'):
                    stats['recordings']['with_code'] += 1
                if fields.get('Errors Detected'):
                    stats['recordings']['with_errors'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting media stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_records(self, days_old: int = 30) -> int:
        """Remove old records to manage storage"""
        if not self.enabled:
            return 0
        
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            deleted_count = 0
            
            # Clean up screenshots
            old_screenshots = self.screenshots_table.get_all()
            for record in old_screenshots:
                created_at = record['fields'].get('Created At', '')
                if created_at < cutoff_str:
                    self.screenshots_table.delete(record['id'])
                    deleted_count += 1
            
            # Clean up recordings
            old_recordings = self.recordings_table.get_all()
            for record in old_recordings:
                created_at = record['fields'].get('Created At', '')
                if created_at < cutoff_str:
                    self.recordings_table.delete(record['id'])
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0

def create_airtable_schema_documentation():
    """Generate documentation for the required Airtable schema"""
    schema_doc = """
# Media AI Pipeline - Airtable Schema

## Base Setup
Create a new Airtable base called "Media AI Pipeline" with the following tables:

### 1. Screenshots Table
| Field Name | Field Type | Description |
|------------|------------|-------------|
| Filename | Single Line Text (Primary) | Screenshot filename |
| File Path | Single Line Text | Full file path |
| File Size MB | Number | File size in megabytes |
| Created At | Date/Time | File creation timestamp |
| Modified At | Date/Time | File modification timestamp |
| Analyzed At | Date/Time | Analysis timestamp |
| Content Type | Single Select | code, documentation, design, email, browser, terminal, ide, meeting, presentation, error, configuration, other |
| Primary Application | Single Line Text | Main application shown |
| Summary | Long Text | Brief description |
| Extracted Text | Long Text | Text content from image |
| Code Detected | Checkbox | Whether code is present |
| Programming Language | Single Line Text | Detected language |
| Error Detected | Checkbox | Whether errors are present |
| Error Details | Long Text | Error descriptions |
| Technical Details | Long Text | Technical information |
| Workflow Step | Single Line Text | Workflow context |
| UI Elements | Long Text | JSON array of UI elements |
| Context Tags | Long Text | JSON array of tags |
| Searchable Keywords | Long Text | JSON array of keywords |
| Full Analysis JSON | Long Text | Complete analysis data |
| Last Updated | Date/Time | Last update timestamp |

### 2. Recordings Table
| Field Name | Field Type | Description |
|------------|------------|-------------|
| Filename | Single Line Text (Primary) | Video filename |
| File Path | Single Line Text | Full file path |
| File Size MB | Number | File size in megabytes |
| Duration Seconds | Number | Video length in seconds |
| Duration Formatted | Single Line Text | Human readable duration |
| Width | Number | Video width in pixels |
| Height | Number | Video height in pixels |
| FPS | Number | Frames per second |
| Codec | Single Line Text | Video codec |
| Bitrate | Number | Video bitrate |
| Created At | Date/Time | File creation timestamp |
| Modified At | Date/Time | File modification timestamp |
| Analyzed At | Date/Time | Analysis timestamp |
| Content Type | Single Select | Same options as Screenshots |
| Primary Application | Single Line Text | Main application shown |
| Summary | Long Text | Brief description |
| Code Detected | Checkbox | Whether code is present |
| Errors Detected | Checkbox | Whether errors are present |
| Error Count | Number | Number of errors found |
| Keyframe Count | Number | Number of analyzed keyframes |
| Context Tags | Long Text | JSON array of tags |
| Technical Details | Long Text | JSON array of technical info |
| Workflow Steps | Long Text | JSON array of workflow steps |
| Searchable Keywords | Long Text | JSON array of keywords |
| Full Analysis JSON | Long Text | Complete analysis data |
| Keyframes JSON | Long Text | Keyframe analysis data |
| Total Tokens Used | Number | AI tokens consumed |
| Last Updated | Date/Time | Last update timestamp |

### 3. Analysis Table (Optional)
For storing cross-media analytics and insights.

### 4. Tags Table (Optional)
For managing tag taxonomy and relationships.
"""
    
    with open('/Users/savantlab/Desktop/media_ai_pipeline/AIRTABLE_SCHEMA.md', 'w') as f:
        f.write(schema_doc)
    
    return schema_doc

if __name__ == "__main__":
    # Generate schema documentation
    create_airtable_schema_documentation()
    print("Airtable schema documentation created!")
    
    # Test connection
    manager = AirtableMediaManager()
    if manager.enabled:
        print("Airtable connection successful!")
        stats = manager.get_media_stats()
        print(f"Current stats: {stats}")
    else:
        print("Airtable not configured - running in local mode")
