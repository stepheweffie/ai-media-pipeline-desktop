"""
d6tflow Data Storage and Retrieval Tasks for Media AI Pipeline
Tasks for saving results to Airtable, local storage, and querying data
"""
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import d6tflow
import pandas as pd

from d6tflow_config import PipelineConfig
from d6tflow_screenshot_tasks import BatchScreenshotAnalysis
from d6tflow_video_tasks import BatchVideoAnalysis
from d6tflow_lecture_tasks import BatchLectureProcessing
from airtable_media_manager import AirtableMediaManager
from ai_query_interface import AIQueryInterface


class SaveScreenshotsToAirtable(d6tflow.tasks.TaskJson):
    """Save screenshot analysis results to Airtable"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_SCREENSHOT_LIMIT)
    batch_size = d6tflow.IntParameter(default=PipelineConfig.BATCH_SIZE_SCREENSHOTS)
    recent_hours = d6tflow.OptionalParameter(default=None)
    source_dir = d6tflow.Parameter(default=PipelineConfig.PNG_SOURCE_DIR)
    
    def requires(self):
        return BatchScreenshotAnalysis(
            limit=self.limit,
            batch_size=self.batch_size,
            recent_hours=self.recent_hours,
            source_dir=self.source_dir
        )
    
    def run(self):
        """Save screenshot analysis results to Airtable"""
        try:
            if not PipelineConfig.ENABLE_AIRTABLE:
                result = {
                    'storage_type': 'airtable_screenshots',
                    'saved_at': datetime.now().isoformat(),
                    'saved_count': 0,
                    'message': 'Airtable not configured - skipping storage',
                    'airtable_enabled': False
                }
                self.save(result)
                return
            
            print("💾 Saving screenshot analysis to Airtable...")
            
            # Load analysis results
            analysis_df = self.input().load()
            
            if analysis_df.empty:
                result = {
                    'storage_type': 'airtable_screenshots',
                    'saved_at': datetime.now().isoformat(),
                    'saved_count': 0,
                    'message': 'No screenshot analysis results to save',
                    'airtable_enabled': True
                }
                self.save(result)
                return
            
            # Initialize Airtable manager
            airtable_manager = AirtableMediaManager()
            
            # Process successful analyses and save to Airtable
            successful_analyses = analysis_df[analysis_df['analysis_status'] == 'success']
            saved_count = 0
            errors = []
            
            for idx, row in successful_analyses.iterrows():
                try:
                    # Create analysis record structure expected by Airtable manager
                    analysis_record = {
                        'metadata': {
                            'filename': row['filename'],
                            'filepath': row['filepath'],
                            'file_size_mb': row['file_size_mb'],
                            'analyzed_at': row['processed_at'].isoformat() if hasattr(row['processed_at'], 'isoformat') else str(row['processed_at'])
                        },
                        'analysis': {
                            'content_type': row['content_type'],
                            'primary_application': row['primary_application'],
                            'code_detected': row['code_detected'],
                            'error_detected': row['error_detected'],
                            'summary': row['summary']
                        },
                        'tokens_used': row['tokens_used']
                    }
                    
                    # Save to Airtable
                    saved_id = airtable_manager.save_screenshot_analysis(analysis_record)
                    if saved_id:
                        saved_count += 1
                    
                except Exception as e:
                    errors.append({'filename': row['filename'], 'error': str(e)})
            
            result = {
                'storage_type': 'airtable_screenshots',
                'saved_at': datetime.now().isoformat(),
                'total_processed': len(analysis_df),
                'successful_analyses': len(successful_analyses),
                'saved_count': saved_count,
                'error_count': len(errors),
                'errors': errors[:5],  # Limit error details
                'airtable_enabled': True,
                'success_rate': round(saved_count / len(successful_analyses) * 100, 1) if len(successful_analyses) > 0 else 0
            }
            
            print(f"💾 Airtable Storage Complete: {saved_count}/{len(successful_analyses)} screenshots saved")
            self.save(result)
            
        except Exception as e:
            error_result = {
                'storage_type': 'airtable_screenshots',
                'saved_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'saved_count': 0
            }
            print(f"❌ Error saving to Airtable: {e}")
            self.save(error_result)


class SaveVideosToAirtable(d6tflow.tasks.TaskJson):
    """Save video analysis results to Airtable"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_VIDEO_LIMIT)
    batch_size = d6tflow.IntParameter(default=PipelineConfig.BATCH_SIZE_VIDEOS)
    recent_hours = d6tflow.OptionalParameter(default=None)
    source_dir = d6tflow.Parameter(default=PipelineConfig.MP4_SOURCE_DIR)
    num_keyframes = d6tflow.IntParameter(default=5)
    
    def requires(self):
        return BatchVideoAnalysis(
            limit=self.limit,
            batch_size=self.batch_size,
            recent_hours=self.recent_hours,
            source_dir=self.source_dir,
            num_keyframes=self.num_keyframes
        )
    
    def run(self):
        """Save video analysis results to Airtable"""
        try:
            if not PipelineConfig.ENABLE_AIRTABLE:
                result = {
                    'storage_type': 'airtable_videos',
                    'saved_at': datetime.now().isoformat(),
                    'saved_count': 0,
                    'message': 'Airtable not configured - skipping storage',
                    'airtable_enabled': False
                }
                self.save(result)
                return
            
            print("💾 Saving video analysis to Airtable...")
            
            # Load analysis results
            analysis_df = self.input().load()
            
            if analysis_df.empty:
                result = {
                    'storage_type': 'airtable_videos',
                    'saved_at': datetime.now().isoformat(),
                    'saved_count': 0,
                    'message': 'No video analysis results to save',
                    'airtable_enabled': True
                }
                self.save(result)
                return
            
            # Initialize Airtable manager
            airtable_manager = AirtableMediaManager()
            
            # Process successful analyses and save to Airtable
            successful_analyses = analysis_df[analysis_df['analysis_status'] == 'success']
            saved_count = 0
            errors = []
            
            for idx, row in successful_analyses.iterrows():
                try:
                    # Create analysis record structure expected by Airtable manager
                    analysis_record = {
                        'metadata': {
                            'filename': row['filename'],
                            'filepath': row['filepath'],
                            'file_size_mb': row['file_size_mb'],
                            'duration_seconds': row['duration_seconds'],
                            'analyzed_at': row['processed_at'].isoformat() if hasattr(row['processed_at'], 'isoformat') else str(row['processed_at'])
                        },
                        'analysis': {
                            'content_type': row['content_type'],
                            'primary_application': row['primary_application'],
                            'code_detected': row['code_detected'],
                            'errors_detected': row['errors_detected'],
                            'duration_formatted': row['duration_formatted'],
                            'keyframe_count': row['keyframes_analyzed'],
                            'summary': row['summary']
                        },
                        'total_tokens_used': row['tokens_used']
                    }
                    
                    # Save to Airtable
                    saved_id = airtable_manager.save_video_analysis(analysis_record)
                    if saved_id:
                        saved_count += 1
                    
                except Exception as e:
                    errors.append({'filename': row['filename'], 'error': str(e)})
            
            result = {
                'storage_type': 'airtable_videos',
                'saved_at': datetime.now().isoformat(),
                'total_processed': len(analysis_df),
                'successful_analyses': len(successful_analyses),
                'saved_count': saved_count,
                'error_count': len(errors),
                'errors': errors[:5],  # Limit error details
                'airtable_enabled': True,
                'success_rate': round(saved_count / len(successful_analyses) * 100, 1) if len(successful_analyses) > 0 else 0
            }
            
            print(f"💾 Airtable Storage Complete: {saved_count}/{len(successful_analyses)} videos saved")
            self.save(result)
            
        except Exception as e:
            error_result = {
                'storage_type': 'airtable_videos',
                'saved_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'saved_count': 0
            }
            print(f"❌ Error saving videos to Airtable: {e}")
            self.save(error_result)


class QueryMediaData(d6tflow.tasks.TaskJson):
    """Query analyzed media data using natural language"""
    
    # Task parameters
    query = d6tflow.Parameter()
    media_type = d6tflow.Parameter(default='all')  # 'all', 'screenshots', 'videos', 'lectures'
    limit = d6tflow.IntParameter(default=10)
    
    def run(self):
        """Process natural language query about media"""
        try:
            print(f"🔍 Processing query: '{self.query}' (type: {self.media_type})")
            
            # Initialize query interface
            query_interface = AIQueryInterface()
            
            # Process the query
            if self.media_type == 'lectures':
                # For lecture queries, use the lecture search functionality
                from d6tflow_lecture_tasks import search_lectures
                results = search_lectures(self.query, limit=self.limit)
                
                query_result = {
                    'query': self.query,
                    'media_type': self.media_type,
                    'results_count': results.get('results_count', 0),
                    'results': results.get('results', []),
                    'query_type': 'lecture_search',
                    'processed_at': datetime.now().isoformat()
                }
            else:
                # For general media queries, use the AI query interface
                query_result = query_interface.process_query(self.query)
                query_result.update({
                    'query': self.query,
                    'media_type': self.media_type,
                    'limit': self.limit,
                    'query_type': 'general_media',
                    'processed_at': datetime.now().isoformat()
                })
            
            print(f"🔍 Query completed: {query_result.get('results_count', 0)} results found")
            self.save(query_result)
            
        except Exception as e:
            error_result = {
                'query': self.query,
                'media_type': self.media_type,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'results_count': 0,
                'results': [],
                'processed_at': datetime.now().isoformat()
            }
            print(f"❌ Error processing query: {e}")
            self.save(error_result)


class ExportAnalysisResults(d6tflow.tasks.TaskJson):
    """Export analysis results in various formats"""
    
    # Task parameters
    export_format = d6tflow.Parameter(default='json')  # 'json', 'csv', 'summary'
    include_screenshots = d6tflow.BoolParameter(default=True)
    include_videos = d6tflow.BoolParameter(default=True)
    include_lectures = d6tflow.BoolParameter(default=True)
    output_path = d6tflow.OptionalParameter(default=None)
    
    def requires(self):
        # Return all the analysis tasks we want to export
        requirements = {}
        
        if self.include_screenshots:
            requirements['screenshots'] = BatchScreenshotAnalysis()
        
        if self.include_videos:
            requirements['videos'] = BatchVideoAnalysis()
        
        if self.include_lectures:
            requirements['lectures'] = BatchLectureProcessing()
        
        return requirements
    
    def run(self):
        """Export analysis results"""
        try:
            print(f"📤 Exporting analysis results in {self.export_format} format...")
            
            export_data = {
                'export_info': {
                    'format': self.export_format,
                    'exported_at': datetime.now().isoformat(),
                    'include_screenshots': self.include_screenshots,
                    'include_videos': self.include_videos,
                    'include_lectures': self.include_lectures
                },
                'data': {}
            }
            
            # Load and process each type of analysis
            if self.include_screenshots and 'screenshots' in self.input():
                screenshots_df = self.input()['screenshots'].load()
                export_data['data']['screenshots'] = {
                    'count': len(screenshots_df),
                    'successful': len(screenshots_df[screenshots_df['analysis_status'] == 'success']),
                    'data': screenshots_df.to_dict('records') if self.export_format == 'json' else screenshots_df
                }
            
            if self.include_videos and 'videos' in self.input():
                videos_df = self.input()['videos'].load()
                export_data['data']['videos'] = {
                    'count': len(videos_df),
                    'successful': len(videos_df[videos_df['analysis_status'] == 'success']),
                    'data': videos_df.to_dict('records') if self.export_format == 'json' else videos_df
                }
            
            if self.include_lectures and 'lectures' in self.input():
                lectures_df = self.input()['lectures'].load()
                export_data['data']['lectures'] = {
                    'count': len(lectures_df),
                    'successful': len(lectures_df[lectures_df['processing_status'] == 'success']),
                    'data': lectures_df.to_dict('records') if self.export_format == 'json' else lectures_df
                }
            
            # Generate summary statistics
            total_files = sum(section.get('count', 0) for section in export_data['data'].values())
            total_successful = sum(section.get('successful', 0) for section in export_data['data'].values())
            
            export_data['summary'] = {
                'total_files_processed': total_files,
                'total_successful': total_successful,
                'success_rate': round(total_successful / total_files * 100, 1) if total_files > 0 else 0,
                'sections_included': list(export_data['data'].keys())
            }
            
            # If output path is specified, save to file as well
            if self.output_path:
                output_file = Path(self.output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                if self.export_format == 'json':
                    with open(output_file, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                elif self.export_format == 'csv':
                    # Create separate CSV files for each media type
                    base_path = output_file.with_suffix('')
                    for media_type, data in export_data['data'].items():
                        if isinstance(data['data'], pd.DataFrame):
                            csv_path = f"{base_path}_{media_type}.csv"
                            data['data'].to_csv(csv_path, index=False)
            
            print(f"📤 Export complete: {total_successful}/{total_files} successful analyses exported")
            self.save(export_data)
            
        except Exception as e:
            error_result = {
                'export_info': {
                    'format': self.export_format,
                    'exported_at': datetime.now().isoformat(),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                },
                'data': {},
                'summary': {
                    'total_files_processed': 0,
                    'total_successful': 0,
                    'success_rate': 0
                }
            }
            print(f"❌ Error exporting results: {e}")
            self.save(error_result)


class CleanupOldData(d6tflow.tasks.TaskJson):
    """Clean up old analysis data and temporary files"""
    
    # Task parameters
    days_to_keep = d6tflow.IntParameter(default=30)
    cleanup_airtable = d6tflow.BoolParameter(default=False)
    cleanup_local = d6tflow.BoolParameter(default=True)
    cleanup_cache = d6tflow.BoolParameter(default=True)
    
    def run(self):
        """Clean up old data"""
        try:
            print(f"🧹 Starting cleanup: removing data older than {self.days_to_keep} days...")
            
            cleanup_results = {
                'cleanup_started_at': datetime.now().isoformat(),
                'days_to_keep': self.days_to_keep,
                'cleanup_airtable': self.cleanup_airtable,
                'cleanup_local': self.cleanup_local,
                'cleanup_cache': self.cleanup_cache,
                'results': {}
            }
            
            # Cleanup Airtable records if enabled
            if self.cleanup_airtable and PipelineConfig.ENABLE_AIRTABLE:
                try:
                    airtable_manager = AirtableMediaManager()
                    deleted_records = airtable_manager.cleanup_old_records(self.days_to_keep)
                    cleanup_results['results']['airtable'] = {
                        'deleted_records': deleted_records,
                        'status': 'success'
                    }
                except Exception as e:
                    cleanup_results['results']['airtable'] = {
                        'deleted_records': 0,
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Cleanup local d6tflow cache if enabled
            if self.cleanup_cache:
                try:
                    import shutil
                    cache_path = PipelineConfig.DATA_DIR
                    
                    # Get cache size before cleanup
                    cache_size_mb = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024 * 1024)
                    
                    # For now, just report cache size (actual cleanup would be more complex)
                    cleanup_results['results']['cache'] = {
                        'cache_size_mb': round(cache_size_mb, 2),
                        'status': 'analyzed',
                        'message': f"Cache contains {cache_size_mb:.1f}MB of data"
                    }
                except Exception as e:
                    cleanup_results['results']['cache'] = {
                        'cache_size_mb': 0,
                        'status': 'error',
                        'error': str(e)
                    }
            
            cleanup_results['cleanup_completed_at'] = datetime.now().isoformat()
            cleanup_results['total_items_cleaned'] = sum(
                result.get('deleted_records', 0) 
                for result in cleanup_results['results'].values()
            )
            
            print(f"🧹 Cleanup complete: {cleanup_results['total_items_cleaned']} items processed")
            self.save(cleanup_results)
            
        except Exception as e:
            error_result = {
                'cleanup_started_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'total_items_cleaned': 0
            }
            print(f"❌ Error during cleanup: {e}")
            self.save(error_result)


# =============================================================================
# Helper Functions
# =============================================================================

def save_all_to_airtable(screenshot_limit=None, video_limit=None):
    """Save all analysis results to Airtable"""
    tasks = []
    
    if screenshot_limit is not None:
        tasks.append(SaveScreenshotsToAirtable(limit=screenshot_limit))
    
    if video_limit is not None:
        tasks.append(SaveVideosToAirtable(limit=video_limit))
    
    if tasks:
        d6tflow.run(tasks)
        
        # Collect results
        results = []
        for task in tasks:
            try:
                result = task.output().load()
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
    
    return []


def query_media(query: str, media_type='all', limit=10):
    """Query media data using natural language"""
    task = QueryMediaData(
        query=query,
        media_type=media_type,
        limit=limit
    )
    
    d6tflow.run(task)
    return task.output().load()


def export_all_results(export_format='json', output_path=None):
    """Export all analysis results"""
    task = ExportAnalysisResults(
        export_format=export_format,
        output_path=output_path
    )
    
    d6tflow.run(task)
    return task.output().load()


def cleanup_data(days_to_keep=30, cleanup_airtable=False):
    """Clean up old data"""
    task = CleanupOldData(
        days_to_keep=days_to_keep,
        cleanup_airtable=cleanup_airtable
    )
    
    d6tflow.run(task)
    return task.output().load()
