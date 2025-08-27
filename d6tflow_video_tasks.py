"""
d6tflow Video Analysis Tasks for Media AI Pipeline
Tasks for analyzing screen recordings using keyframe extraction and OpenAI Vision API
"""
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import d6tflow
import pandas as pd

from d6tflow_config import PipelineConfig
from d6tflow_ingestion_tasks import DiscoverVideos
from video_analyzer import VideoAnalyzer


class AnalyzeSingleVideo(d6tflow.tasks.TaskJson):
    """Analyze a single video file by extracting keyframes"""
    
    # Task parameters
    filepath = d6tflow.Parameter()
    filename = d6tflow.Parameter()
    num_keyframes = d6tflow.IntParameter(default=5)
    
    def run(self):
        """Analyze a single video and save results"""
        try:
            print(f"🎥 Analyzing video: {self.filename}")
            
            analyzer = VideoAnalyzer()
            result = analyzer.analyze_video(self.filepath, self.num_keyframes)
            
            # Add task metadata
            result['task_metadata'] = {
                'task_name': 'AnalyzeSingleVideo',
                'processed_at': datetime.now().isoformat(),
                'filepath': self.filepath,
                'filename': self.filename,
                'num_keyframes_requested': self.num_keyframes,
                'keyframes_analyzed': len(result.get('keyframes', []))
            }
            
            print(f"✅ Successfully analyzed: {self.filename} ({len(result.get('keyframes', []))} keyframes)")
            self.save(result)
            
        except Exception as e:
            error_result = {
                'metadata': {
                    'filename': self.filename,
                    'filepath': self.filepath,
                    'analyzed_at': datetime.now().isoformat()
                },
                'analysis': {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                },
                'keyframes': [],
                'task_metadata': {
                    'task_name': 'AnalyzeSingleVideo',
                    'processed_at': datetime.now().isoformat(),
                    'status': 'error'
                }
            }
            print(f"❌ Error analyzing {self.filename}: {e}")
            self.save(error_result)


class BatchVideoAnalysis(d6tflow.tasks.TaskPqPandas):
    """Analyze multiple videos in batches"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_VIDEO_LIMIT)
    batch_size = d6tflow.IntParameter(default=PipelineConfig.BATCH_SIZE_VIDEOS)
    recent_hours = d6tflow.OptionalParameter(default=None)
    source_dir = d6tflow.Parameter(default=PipelineConfig.MP4_SOURCE_DIR)
    num_keyframes = d6tflow.IntParameter(default=5)
    
    def requires(self):
        return DiscoverVideos(
            limit=self.limit,
            recent_hours=self.recent_hours,
            source_dir=self.source_dir
        )
    
    def run(self):
        """Analyze videos in batches"""
        try:
            # Load discovered videos
            videos_df = self.input().load()
            
            if videos_df.empty:
                print("🎥 No videos found to analyze")
                self.save(pd.DataFrame(columns=['filename', 'filepath', 'analysis_status', 'processed_at']))
                return
            
            print(f"🎥 Starting batch analysis of {len(videos_df)} videos...")
            
            analyzer = VideoAnalyzer()
            results = []
            
            # Process videos in batches (videos are more resource-intensive)
            for i in range(0, len(videos_df), self.batch_size):
                batch = videos_df.iloc[i:i + self.batch_size]
                print(f"🔄 Processing batch {i//self.batch_size + 1}: {len(batch)} videos")
                
                for idx, row in batch.iterrows():
                    try:
                        filepath = row['filepath']
                        filename = row['filename']
                        
                        print(f"  🎥 Analyzing: {filename} ({row['file_size_mb']}MB)")
                        analysis_result = analyzer.analyze_video(filepath, self.num_keyframes)
                        
                        # Extract key analysis data for the batch summary
                        analysis_summary = {
                            'filename': filename,
                            'filepath': filepath,
                            'file_size_mb': row['file_size_mb'],
                            'modified_at': row['modified_at'],
                            'analysis_status': 'success',
                            'processed_at': datetime.now(),
                            'content_type': analysis_result.get('analysis', {}).get('content_type', 'unknown'),
                            'primary_application': analysis_result.get('analysis', {}).get('primary_application', 'unknown'),
                            'duration_seconds': analysis_result.get('metadata', {}).get('duration_seconds', 0),
                            'duration_formatted': analysis_result.get('analysis', {}).get('duration_formatted', 'unknown'),
                            'code_detected': analysis_result.get('analysis', {}).get('code_detected', False),
                            'errors_detected': analysis_result.get('analysis', {}).get('errors_detected', False),
                            'keyframes_analyzed': len(analysis_result.get('keyframes', [])),
                            'tokens_used': analysis_result.get('total_tokens_used', 0),
                            'has_error': False,
                            'summary': analysis_result.get('analysis', {}).get('summary', '')[:200] + '...' if analysis_result.get('analysis', {}).get('summary', '') else ''
                        }
                        
                        results.append(analysis_summary)
                        print(f"    ✅ Success: {filename} ({analysis_summary['keyframes_analyzed']} keyframes)")
                        
                    except Exception as e:
                        error_summary = {
                            'filename': row['filename'],
                            'filepath': row['filepath'],
                            'file_size_mb': row['file_size_mb'],
                            'modified_at': row['modified_at'],
                            'analysis_status': 'error',
                            'processed_at': datetime.now(),
                            'content_type': 'unknown',
                            'primary_application': 'unknown',
                            'duration_seconds': 0,
                            'duration_formatted': 'unknown',
                            'code_detected': False,
                            'errors_detected': False,
                            'keyframes_analyzed': 0,
                            'tokens_used': 0,
                            'has_error': True,
                            'error_message': str(e),
                            'summary': ''
                        }
                        results.append(error_summary)
                        print(f"    ❌ Error: {filename} - {e}")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Add batch statistics
            total_analyzed = len(results_df)
            successful_analyses = len(results_df[results_df['analysis_status'] == 'success'])
            error_count = len(results_df[results_df['analysis_status'] == 'error'])
            total_tokens = results_df['tokens_used'].sum()
            total_keyframes = results_df['keyframes_analyzed'].sum()
            total_duration = results_df['duration_seconds'].sum()
            
            batch_stats = {
                'total_files': total_analyzed,
                'successful': successful_analyses,
                'errors': error_count,
                'total_tokens_used': int(total_tokens),
                'total_keyframes_analyzed': int(total_keyframes),
                'total_duration_seconds': float(total_duration),
                'total_duration_formatted': str(pd.Timedelta(seconds=total_duration)),
                'batch_size': self.batch_size,
                'processed_at': datetime.now().isoformat()
            }
            
            results_df.attrs['batch_stats'] = batch_stats
            
            print(f"📊 Batch Analysis Complete: {successful_analyses}/{total_analyzed} successful, "
                  f"{error_count} errors, {total_keyframes} keyframes analyzed, "
                  f"{total_tokens} tokens used")
            
            self.save(results_df)
            
        except Exception as e:
            print(f"❌ Error in batch video analysis: {e}")
            error_df = pd.DataFrame([{
                'filename': 'BATCH_ERROR',
                'filepath': '',
                'analysis_status': 'batch_error',
                'processed_at': datetime.now(),
                'error_message': str(e),
                'has_error': True
            }])
            error_df.attrs['batch_stats'] = {
                'total_files': 0,
                'successful': 0,
                'errors': 1,
                'batch_error': str(e),
                'processed_at': datetime.now().isoformat()
            }
            self.save(error_df)


class VideoAnalysisSummary(d6tflow.tasks.TaskJson):
    """Create a comprehensive summary of video analysis results"""
    
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
        """Generate comprehensive video analysis summary"""
        try:
            # Load batch analysis results
            batch_results_df = self.input().load()
            batch_stats = getattr(batch_results_df, 'attrs', {}).get('batch_stats', {})
            
            print("📊 Creating video analysis summary...")
            
            if batch_results_df.empty:
                summary = {
                    'summary_type': 'video_analysis',
                    'generated_at': datetime.now().isoformat(),
                    'total_files': 0,
                    'message': 'No videos were analyzed',
                    'batch_stats': batch_stats
                }
                self.save(summary)
                return
            
            # Generate content type distribution
            content_types = batch_results_df['content_type'].value_counts().to_dict()
            applications = batch_results_df['primary_application'].value_counts().to_dict()
            
            # Calculate success rates
            successful_files = batch_results_df[batch_results_df['analysis_status'] == 'success']
            error_files = batch_results_df[batch_results_df['analysis_status'] == 'error']
            
            # Analyze video patterns
            code_videos = len(successful_files[successful_files['code_detected'] == True])
            error_videos = len(successful_files[successful_files['errors_detected'] == True])
            
            # Duration analysis
            if not successful_files.empty:
                avg_duration = successful_files['duration_seconds'].mean()
                total_duration = successful_files['duration_seconds'].sum()
                longest_video = successful_files.loc[successful_files['duration_seconds'].idxmax()] if len(successful_files) > 0 else None
                shortest_video = successful_files.loc[successful_files['duration_seconds'].idxmin()] if len(successful_files) > 0 else None
            else:
                avg_duration = 0
                total_duration = 0
                longest_video = None
                shortest_video = None
            
            # File size analysis
            avg_file_size = batch_results_df['file_size_mb'].mean()
            total_size = batch_results_df['file_size_mb'].sum()
            
            # Recent vs older files
            recent_files = 0
            if self.recent_hours:
                cutoff_time = datetime.now() - pd.Timedelta(hours=self.recent_hours)
                recent_files = len(batch_results_df[batch_results_df['modified_at'] >= cutoff_time])
            
            summary = {
                'summary_type': 'video_analysis',
                'generated_at': datetime.now().isoformat(),
                'total_files': len(batch_results_df),
                'successful_analyses': len(successful_files),
                'failed_analyses': len(error_files),
                'success_rate': len(successful_files) / len(batch_results_df) * 100 if len(batch_results_df) > 0 else 0,
                
                # Content analysis
                'content_types': content_types,
                'primary_applications': applications,
                'code_videos': code_videos,
                'error_videos': error_videos,
                
                # Duration analysis
                'average_duration_seconds': round(avg_duration, 2) if avg_duration else 0,
                'total_duration_seconds': round(total_duration, 2),
                'total_duration_formatted': str(pd.Timedelta(seconds=total_duration)),
                'longest_video': {
                    'filename': longest_video['filename'] if longest_video is not None else None,
                    'duration': round(longest_video['duration_seconds'], 2) if longest_video is not None else 0
                },
                'shortest_video': {
                    'filename': shortest_video['filename'] if shortest_video is not None else None,
                    'duration': round(shortest_video['duration_seconds'], 2) if shortest_video is not None else 0
                },
                
                # File statistics
                'average_file_size_mb': round(avg_file_size, 2) if avg_file_size else 0,
                'total_size_mb': round(total_size, 2),
                'recent_files': recent_files,
                
                # Processing statistics
                'total_tokens_used': batch_stats.get('total_tokens_used', 0),
                'total_keyframes_analyzed': batch_stats.get('total_keyframes_analyzed', 0),
                'avg_keyframes_per_video': round(batch_stats.get('total_keyframes_analyzed', 0) / len(successful_files), 1) if len(successful_files) > 0 else 0,
                'processing_time': batch_stats.get('processed_at'),
                'batch_size': self.batch_size,
                
                # Configuration
                'parameters': {
                    'limit': self.limit,
                    'batch_size': self.batch_size,
                    'recent_hours': self.recent_hours,
                    'source_dir': self.source_dir,
                    'num_keyframes': self.num_keyframes
                },
                
                # Top insights
                'insights': {
                    'most_common_content_type': max(content_types.items(), key=lambda x: x[1])[0] if content_types else None,
                    'most_common_application': max(applications.items(), key=lambda x: x[1])[0] if applications else None,
                    'code_percentage': round(code_videos / len(successful_files) * 100, 1) if len(successful_files) > 0 else 0,
                    'error_percentage': round(error_videos / len(successful_files) * 100, 1) if len(successful_files) > 0 else 0,
                    'avg_tokens_per_video': round(batch_stats.get('total_tokens_used', 0) / len(successful_files), 1) if len(successful_files) > 0 else 0
                },
                
                'batch_stats': batch_stats
            }
            
            print(f"✅ Video Analysis Summary: {len(successful_files)}/{len(batch_results_df)} successful, "
                  f"{summary['insights']['most_common_content_type']} most common, "
                  f"{code_videos} with code, "
                  f"{summary['total_duration_formatted']} total duration")
            
            self.save(summary)
            
        except Exception as e:
            error_summary = {
                'summary_type': 'video_analysis',
                'generated_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'total_files': 0,
                'successful_analyses': 0,
                'failed_analyses': 1
            }
            print(f"❌ Error creating video analysis summary: {e}")
            self.save(error_summary)


# =============================================================================
# Helper Functions
# =============================================================================

def run_video_analysis(limit=None, batch_size=None, recent_hours=None, num_keyframes=5):
    """Convenience function to run video analysis workflow"""
    params = {
        'limit': limit or PipelineConfig.DEFAULT_VIDEO_LIMIT,
        'batch_size': batch_size or PipelineConfig.BATCH_SIZE_VIDEOS,
        'recent_hours': recent_hours,
        'num_keyframes': num_keyframes
    }
    
    task = VideoAnalysisSummary(**params)
    d6tflow.run(task)
    
    return task.output().load()


def analyze_specific_videos(filepaths: List[str], num_keyframes=5):
    """Analyze specific video files"""
    tasks = []
    for filepath in filepaths:
        filename = Path(filepath).name
        tasks.append(AnalyzeSingleVideo(
            filepath=filepath, 
            filename=filename, 
            num_keyframes=num_keyframes
        ))
    
    d6tflow.run(tasks)
    
    # Collect results
    results = []
    for task in tasks:
        try:
            result = task.output().load()
            results.append(result)
        except Exception as e:
            results.append({'error': str(e), 'filepath': filepath})
    
    return results
