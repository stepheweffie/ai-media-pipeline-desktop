"""
Main d6tflow Pipeline Orchestrator for Media AI Pipeline
Coordinates all analysis workflows with configurable parameters
"""
import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import d6tflow
import pandas as pd

from d6tflow_config import PipelineConfig, TaskParams
from d6tflow_ingestion_tasks import MediaInventorySummary
from d6tflow_screenshot_tasks import ScreenshotAnalysisSummary
from d6tflow_video_tasks import VideoAnalysisSummary
from d6tflow_lecture_tasks import LectureProcessingSummary
from d6tflow_storage_tasks import (
    SaveScreenshotsToAirtable, 
    SaveVideosToAirtable, 
    ExportAnalysisResults,
    QueryMediaData
)


class FullMediaPipeline(d6tflow.tasks.TaskJson):
    """Complete media analysis pipeline - screenshots, videos, and lectures"""
    
    # Task parameters
    screenshot_limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_SCREENSHOT_LIMIT)
    video_limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_VIDEO_LIMIT)
    lecture_limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_LECTURE_LIMIT)
    recent_hours = d6tflow.OptionalParameter(default=None)
    enable_airtable = d6tflow.BoolParameter(default=PipelineConfig.ENABLE_AIRTABLE)
    include_screenshots = d6tflow.BoolParameter(default=True)
    include_videos = d6tflow.BoolParameter(default=True)
    include_lectures = d6tflow.BoolParameter(default=True)
    num_keyframes = d6tflow.IntParameter(default=5)
    advanced_diarization = d6tflow.BoolParameter(default=False)
    
    def requires(self):
        """Define all required tasks for the full pipeline"""
        requirements = {}
        
        # Always include media inventory
        requirements['inventory'] = MediaInventorySummary()
        
        # Include screenshot analysis if enabled
        if self.include_screenshots:
            requirements['screenshot_analysis'] = ScreenshotAnalysisSummary(
                limit=self.screenshot_limit,
                recent_hours=self.recent_hours
            )
            
            # Include Airtable storage if enabled
            if self.enable_airtable:
                requirements['screenshot_storage'] = SaveScreenshotsToAirtable(
                    limit=self.screenshot_limit,
                    recent_hours=self.recent_hours
                )
        
        # Include video analysis if enabled
        if self.include_videos:
            requirements['video_analysis'] = VideoAnalysisSummary(
                limit=self.video_limit,
                recent_hours=self.recent_hours,
                num_keyframes=self.num_keyframes
            )
            
            # Include Airtable storage if enabled
            if self.enable_airtable:
                requirements['video_storage'] = SaveVideosToAirtable(
                    limit=self.video_limit,
                    recent_hours=self.recent_hours,
                    num_keyframes=self.num_keyframes
                )
        
        # Include lecture processing if enabled
        if self.include_lectures:
            requirements['lecture_processing'] = LectureProcessingSummary(
                limit=self.lecture_limit,
                advanced_diarization=self.advanced_diarization
            )
        
        return requirements
    
    def run(self):
        """Execute the full pipeline and generate comprehensive summary"""
        try:
            print("🚀 Running Full Media AI Pipeline...")
            
            pipeline_summary = {
                'pipeline_type': 'full_media_pipeline',
                'started_at': datetime.now().isoformat(),
                'configuration': {
                    'screenshot_limit': self.screenshot_limit,
                    'video_limit': self.video_limit,
                    'lecture_limit': self.lecture_limit,
                    'recent_hours': self.recent_hours,
                    'enable_airtable': self.enable_airtable,
                    'include_screenshots': self.include_screenshots,
                    'include_videos': self.include_videos,
                    'include_lectures': self.include_lectures,
                    'num_keyframes': self.num_keyframes,
                    'advanced_diarization': self.advanced_diarization
                },
                'results': {},
                'summary': {}
            }
            
            # Process results from each component
            input_data = self.input()
            
            # Media inventory summary
            if 'inventory' in input_data:
                inventory_df = input_data['inventory'].load()
                inventory_stats = getattr(inventory_df, 'attrs', {}).get('summary_stats', {})
                pipeline_summary['results']['inventory'] = {
                    'total_files_discovered': inventory_stats.get('total_files', 0),
                    'screenshots_found': inventory_stats.get('screenshots', 0),
                    'videos_found': inventory_stats.get('videos', 0),
                    'lectures_found': inventory_stats.get('lectures', 0)
                }
            
            # Screenshot analysis results
            if 'screenshot_analysis' in input_data:
                screenshot_summary = input_data['screenshot_analysis'].load()
                pipeline_summary['results']['screenshots'] = {
                    'total_analyzed': screenshot_summary.get('total_files', 0),
                    'successful': screenshot_summary.get('successful_analyses', 0),
                    'failed': screenshot_summary.get('failed_analyses', 0),
                    'success_rate': screenshot_summary.get('success_rate', 0),
                    'tokens_used': screenshot_summary.get('total_tokens_used', 0),
                    'most_common_type': screenshot_summary.get('insights', {}).get('most_common_content_type'),
                    'code_detected': screenshot_summary.get('code_screenshots', 0)
                }
            
            # Video analysis results
            if 'video_analysis' in input_data:
                video_summary = input_data['video_analysis'].load()
                pipeline_summary['results']['videos'] = {
                    'total_analyzed': video_summary.get('total_files', 0),
                    'successful': video_summary.get('successful_analyses', 0),
                    'failed': video_summary.get('failed_analyses', 0),
                    'success_rate': video_summary.get('success_rate', 0),
                    'tokens_used': video_summary.get('total_tokens_used', 0),
                    'total_duration_hours': video_summary.get('total_duration_hours', 0),
                    'keyframes_analyzed': video_summary.get('total_keyframes_analyzed', 0),
                    'most_common_type': video_summary.get('insights', {}).get('most_common_content_type')
                }
            
            # Lecture processing results
            if 'lecture_processing' in input_data:
                lecture_summary = input_data['lecture_processing'].load()
                pipeline_summary['results']['lectures'] = {
                    'total_processed': lecture_summary.get('total_files', 0),
                    'successful': lecture_summary.get('successful_processing', 0),
                    'failed': lecture_summary.get('failed_processing', 0),
                    'success_rate': lecture_summary.get('success_rate', 0),
                    'total_duration_hours': lecture_summary.get('total_duration_hours', 0),
                    'transcript_segments': lecture_summary.get('total_transcript_segments', 0),
                    'vector_chunks_stored': lecture_summary.get('total_vector_chunks_stored', 0),
                    'estimated_words': lecture_summary.get('estimated_words', 0)
                }
            
            # Airtable storage results
            airtable_stats = {'enabled': self.enable_airtable, 'results': {}}
            
            if self.enable_airtable:
                if 'screenshot_storage' in input_data:
                    screenshot_storage = input_data['screenshot_storage'].load()
                    airtable_stats['results']['screenshots'] = {
                        'saved_count': screenshot_storage.get('saved_count', 0),
                        'success_rate': screenshot_storage.get('success_rate', 0)
                    }
                
                if 'video_storage' in input_data:
                    video_storage = input_data['video_storage'].load()
                    airtable_stats['results']['videos'] = {
                        'saved_count': video_storage.get('saved_count', 0),
                        'success_rate': video_storage.get('success_rate', 0)
                    }
            
            pipeline_summary['results']['airtable'] = airtable_stats
            
            # Generate overall summary
            total_files_processed = (
                pipeline_summary['results'].get('screenshots', {}).get('total_analyzed', 0) +
                pipeline_summary['results'].get('videos', {}).get('total_analyzed', 0) +
                pipeline_summary['results'].get('lectures', {}).get('total_processed', 0)
            )
            
            total_successful = (
                pipeline_summary['results'].get('screenshots', {}).get('successful', 0) +
                pipeline_summary['results'].get('videos', {}).get('successful', 0) +
                pipeline_summary['results'].get('lectures', {}).get('successful', 0)
            )
            
            total_tokens_used = (
                pipeline_summary['results'].get('screenshots', {}).get('tokens_used', 0) +
                pipeline_summary['results'].get('videos', {}).get('tokens_used', 0)
            )
            
            pipeline_summary['summary'] = {
                'total_files_processed': total_files_processed,
                'total_successful': total_successful,
                'overall_success_rate': round(total_successful / total_files_processed * 100, 1) if total_files_processed > 0 else 0,
                'total_tokens_used': total_tokens_used,
                'components_executed': list(pipeline_summary['results'].keys()),
                'processing_time_minutes': 0  # Would need to track actual time
            }
            
            pipeline_summary['completed_at'] = datetime.now().isoformat()
            
            print(f"🎉 Pipeline Complete! Processed {total_successful}/{total_files_processed} files successfully")
            print(f"   📊 Components: {', '.join(pipeline_summary['summary']['components_executed'])}")
            print(f"   💰 Tokens used: {total_tokens_used:,}")
            
            self.save(pipeline_summary)
            
        except Exception as e:
            error_summary = {
                'pipeline_type': 'full_media_pipeline',
                'started_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'configuration': {
                    'screenshot_limit': self.screenshot_limit,
                    'video_limit': self.video_limit,
                    'lecture_limit': self.lecture_limit
                }
            }
            print(f"❌ Pipeline failed: {e}")
            self.save(error_summary)


class ScreenshotOnlyPipeline(d6tflow.tasks.TaskJson):
    """Screenshot analysis only pipeline"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_SCREENSHOT_LIMIT)
    batch_size = d6tflow.IntParameter(default=PipelineConfig.BATCH_SIZE_SCREENSHOTS)
    recent_hours = d6tflow.OptionalParameter(default=None)
    enable_airtable = d6tflow.BoolParameter(default=PipelineConfig.ENABLE_AIRTABLE)
    
    def requires(self):
        requirements = {
            'analysis': ScreenshotAnalysisSummary(
                limit=self.limit,
                batch_size=self.batch_size,
                recent_hours=self.recent_hours
            )
        }
        
        if self.enable_airtable:
            requirements['storage'] = SaveScreenshotsToAirtable(
                limit=self.limit,
                batch_size=self.batch_size,
                recent_hours=self.recent_hours
            )
        
        return requirements
    
    def run(self):
        """Execute screenshot-only pipeline"""
        try:
            print("📸 Running Screenshot Analysis Pipeline...")
            
            # Load analysis results
            analysis_summary = self.input()['analysis'].load()
            
            pipeline_result = {
                'pipeline_type': 'screenshot_only',
                'completed_at': datetime.now().isoformat(),
                'parameters': {
                    'limit': self.limit,
                    'batch_size': self.batch_size,
                    'recent_hours': self.recent_hours,
                    'enable_airtable': self.enable_airtable
                },
                'results': analysis_summary
            }
            
            # Add storage results if enabled
            if self.enable_airtable and 'storage' in self.input():
                storage_results = self.input()['storage'].load()
                pipeline_result['airtable_storage'] = storage_results
            
            print(f"📸 Screenshot Pipeline Complete: {analysis_summary.get('successful_analyses', 0)} screenshots analyzed")
            self.save(pipeline_result)
            
        except Exception as e:
            error_result = {
                'pipeline_type': 'screenshot_only',
                'completed_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.save(error_result)


class VideoOnlyPipeline(d6tflow.tasks.TaskJson):
    """Video analysis only pipeline"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_VIDEO_LIMIT)
    batch_size = d6tflow.IntParameter(default=PipelineConfig.BATCH_SIZE_VIDEOS)
    recent_hours = d6tflow.OptionalParameter(default=None)
    num_keyframes = d6tflow.IntParameter(default=5)
    enable_airtable = d6tflow.BoolParameter(default=PipelineConfig.ENABLE_AIRTABLE)
    
    def requires(self):
        requirements = {
            'analysis': VideoAnalysisSummary(
                limit=self.limit,
                batch_size=self.batch_size,
                recent_hours=self.recent_hours,
                num_keyframes=self.num_keyframes
            )
        }
        
        if self.enable_airtable:
            requirements['storage'] = SaveVideosToAirtable(
                limit=self.limit,
                batch_size=self.batch_size,
                recent_hours=self.recent_hours,
                num_keyframes=self.num_keyframes
            )
        
        return requirements
    
    def run(self):
        """Execute video-only pipeline"""
        try:
            print("🎥 Running Video Analysis Pipeline...")
            
            # Load analysis results
            analysis_summary = self.input()['analysis'].load()
            
            pipeline_result = {
                'pipeline_type': 'video_only',
                'completed_at': datetime.now().isoformat(),
                'parameters': {
                    'limit': self.limit,
                    'batch_size': self.batch_size,
                    'recent_hours': self.recent_hours,
                    'num_keyframes': self.num_keyframes,
                    'enable_airtable': self.enable_airtable
                },
                'results': analysis_summary
            }
            
            # Add storage results if enabled
            if self.enable_airtable and 'storage' in self.input():
                storage_results = self.input()['storage'].load()
                pipeline_result['airtable_storage'] = storage_results
            
            print(f"🎥 Video Pipeline Complete: {analysis_summary.get('successful_analyses', 0)} videos analyzed")
            self.save(pipeline_result)
            
        except Exception as e:
            error_result = {
                'pipeline_type': 'video_only',
                'completed_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.save(error_result)


class LectureOnlyPipeline(d6tflow.tasks.TaskJson):
    """Lecture processing only pipeline"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_LECTURE_LIMIT)
    source_directory = d6tflow.OptionalParameter(default=None)
    advanced_diarization = d6tflow.BoolParameter(default=False)
    model_size = d6tflow.Parameter(default="large-v3")
    
    def requires(self):
        return LectureProcessingSummary(
            limit=self.limit,
            source_directory=self.source_directory,
            advanced_diarization=self.advanced_diarization,
            model_size=self.model_size
        )
    
    def run(self):
        """Execute lecture-only pipeline"""
        try:
            print("🎓 Running Lecture Processing Pipeline...")
            
            # Load processing results
            processing_summary = self.input().load()
            
            pipeline_result = {
                'pipeline_type': 'lecture_only',
                'completed_at': datetime.now().isoformat(),
                'parameters': {
                    'limit': self.limit,
                    'source_directory': self.source_directory,
                    'advanced_diarization': self.advanced_diarization,
                    'model_size': self.model_size
                },
                'results': processing_summary
            }
            
            print(f"🎓 Lecture Pipeline Complete: {processing_summary.get('successful_processing', 0)} lectures processed")
            self.save(pipeline_result)
            
        except Exception as e:
            error_result = {
                'pipeline_type': 'lecture_only',
                'completed_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.save(error_result)


# =============================================================================
# Helper Functions
# =============================================================================

def run_full_pipeline(**kwargs):
    """Run the complete media analysis pipeline"""
    # Use default parameters unless overridden
    params = TaskParams.full_pipeline()
    params.update(kwargs)
    
    task = FullMediaPipeline(**params)
    d6tflow.run(task)
    
    return task.output().load()


def run_screenshot_pipeline(limit=None, recent_hours=None, enable_airtable=None):
    """Run screenshot analysis only"""
    params = TaskParams.screenshot_analysis(limit=limit, recent_hours=recent_hours)
    if enable_airtable is not None:
        params['enable_airtable'] = enable_airtable
    
    task = ScreenshotOnlyPipeline(**params)
    d6tflow.run(task)
    
    return task.output().load()


def run_video_pipeline(limit=None, recent_hours=None, num_keyframes=5, enable_airtable=None):
    """Run video analysis only"""
    params = TaskParams.video_analysis(limit=limit, recent_hours=recent_hours)
    params['num_keyframes'] = num_keyframes
    if enable_airtable is not None:
        params['enable_airtable'] = enable_airtable
    
    task = VideoOnlyPipeline(**params)
    d6tflow.run(task)
    
    return task.output().load()


def run_lecture_pipeline(limit=None, source_directory=None, advanced_diarization=False, model_size="large-v3"):
    """Run lecture processing only"""
    params = TaskParams.lecture_processing(
        limit=limit, 
        directory=source_directory,
        advanced_diarization=advanced_diarization
    )
    params['model_size'] = model_size
    
    task = LectureOnlyPipeline(**params)
    d6tflow.run(task)
    
    return task.output().load()


def get_pipeline_status():
    """Get status of all pipeline components"""
    try:
        from d6tflow_config import get_task_status_summary
        return get_task_status_summary()
    except Exception as e:
        return {'error': str(e)}


def query_pipeline_results(query: str, media_type='all', limit=10):
    """Query processed media using natural language"""
    task = QueryMediaData(query=query, media_type=media_type, limit=limit)
    d6tflow.run(task)
    
    return task.output().load()


def export_pipeline_results(export_format='json', output_path=None, **kwargs):
    """Export all pipeline results"""
    params = {
        'export_format': export_format,
        'output_path': output_path
    }
    params.update(kwargs)
    
    task = ExportAnalysisResults(**params)
    d6tflow.run(task)
    
    return task.output().load()
