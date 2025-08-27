"""
d6tflow Lecture Processing Tasks for Media AI Pipeline
Tasks for transcribing lectures, speaker diarization, and vector database storage
"""
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import d6tflow
import pandas as pd

from d6tflow_config import PipelineConfig
from d6tflow_ingestion_tasks import DiscoverLectures
from lecture_processor import LectureProcessor, TranscriptChunk


class ProcessSingleLecture(d6tflow.tasks.TaskJson):
    """Process a single lecture video with transcription and speaker diarization"""
    
    # Task parameters
    filepath = d6tflow.Parameter()
    filename = d6tflow.Parameter()
    advanced_diarization = d6tflow.BoolParameter(default=False)
    model_size = d6tflow.Parameter(default="large-v3")
    
    def run(self):
        """Process a single lecture and save results"""
        try:
            print(f"🎓 Processing lecture: {self.filename}")
            
            processor = LectureProcessor()
            
            # Set model size if different from default
            if self.model_size != "large-v3":
                processor.model_size = self.model_size
            
            result = processor.process_video(self.filepath, self.advanced_diarization)
            
            # Add task metadata
            result['task_metadata'] = {
                'task_name': 'ProcessSingleLecture',
                'processed_at': datetime.now().isoformat(),
                'filepath': self.filepath,
                'filename': self.filename,
                'advanced_diarization': self.advanced_diarization,
                'model_size': self.model_size,
                'segments_processed': len(result.get('transcript_segments', [])),
                'vector_storage_enabled': result.get('vector_storage', {}).get('enabled', False)
            }
            
            segments_count = len(result.get('transcript_segments', []))
            duration = result.get('metadata', {}).get('duration_seconds', 0)
            
            print(f"✅ Successfully processed: {self.filename} ({segments_count} segments, {duration:.1f}s)")
            self.save(result)
            
        except Exception as e:
            error_result = {
                'metadata': {
                    'filename': self.filename,
                    'filepath': self.filepath,
                    'processed_at': datetime.now().isoformat()
                },
                'transcript_segments': [],
                'error': str(e),
                'traceback': traceback.format_exc(),
                'task_metadata': {
                    'task_name': 'ProcessSingleLecture',
                    'processed_at': datetime.now().isoformat(),
                    'status': 'error'
                }
            }
            print(f"❌ Error processing lecture {self.filename}: {e}")
            self.save(error_result)


class BatchLectureProcessing(d6tflow.tasks.TaskPqPandas):
    """Process multiple lectures in batches"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_LECTURE_LIMIT)
    source_directory = d6tflow.OptionalParameter(default=None)
    advanced_diarization = d6tflow.BoolParameter(default=False)
    model_size = d6tflow.Parameter(default="large-v3")
    
    def requires(self):
        return DiscoverLectures(
            limit=self.limit,
            source_directory=self.source_directory
        )
    
    def run(self):
        """Process lectures in batches"""
        try:
            # Load discovered lectures
            lectures_df = self.input().load()
            
            if lectures_df.empty:
                print("🎓 No lectures found to process")
                self.save(pd.DataFrame(columns=['filename', 'filepath', 'processing_status', 'processed_at']))
                return
            
            print(f"🎓 Starting batch processing of {len(lectures_df)} lectures...")
            
            processor = LectureProcessor()
            if self.model_size != "large-v3":
                processor.model_size = self.model_size
                
            results = []
            
            # Process lectures one by one (they are CPU-intensive)
            for idx, row in lectures_df.iterrows():
                try:
                    filepath = row['filepath']
                    filename = row['filename']
                    
                    print(f"🎓 Processing lecture {idx + 1}/{len(lectures_df)}: {filename} ({row['file_size_mb']}MB)")
                    processing_result = processor.process_video(filepath, self.advanced_diarization)
                    
                    # Extract key processing data for the batch summary
                    processing_summary = {
                        'filename': filename,
                        'filepath': filepath,
                        'file_size_mb': row['file_size_mb'],
                        'modified_at': row['modified_at'],
                        'processing_status': 'success',
                        'processed_at': datetime.now(),
                        'duration_seconds': processing_result.get('metadata', {}).get('duration_seconds', 0),
                        'segments_count': len(processing_result.get('transcript_segments', [])),
                        'speaker_count': len(set(seg.get('speaker', 'Unknown') for seg in processing_result.get('transcript_segments', []))),
                        'vector_stored': processing_result.get('vector_storage', {}).get('chunks_stored', 0) > 0,
                        'chunks_stored': processing_result.get('vector_storage', {}).get('chunks_stored', 0),
                        'has_error': False,
                        'transcript_preview': processing_result.get('transcript_text', '')[:200] + '...' if processing_result.get('transcript_text', '') else ''
                    }
                    
                    results.append(processing_summary)
                    print(f"  ✅ Success: {filename} ({processing_summary['segments_count']} segments, "
                          f"{processing_summary['speaker_count']} speakers)")
                    
                except Exception as e:
                    error_summary = {
                        'filename': row['filename'],
                        'filepath': row['filepath'],
                        'file_size_mb': row['file_size_mb'],
                        'modified_at': row['modified_at'],
                        'processing_status': 'error',
                        'processed_at': datetime.now(),
                        'duration_seconds': 0,
                        'segments_count': 0,
                        'speaker_count': 0,
                        'vector_stored': False,
                        'chunks_stored': 0,
                        'has_error': True,
                        'error_message': str(e),
                        'transcript_preview': ''
                    }
                    results.append(error_summary)
                    print(f"  ❌ Error: {filename} - {e}")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Add batch statistics
            total_processed = len(results_df)
            successful_processing = len(results_df[results_df['processing_status'] == 'success'])
            error_count = len(results_df[results_df['processing_status'] == 'error'])
            total_duration = results_df['duration_seconds'].sum()
            total_segments = results_df['segments_count'].sum()
            total_chunks_stored = results_df['chunks_stored'].sum()
            
            batch_stats = {
                'total_files': total_processed,
                'successful': successful_processing,
                'errors': error_count,
                'total_duration_seconds': float(total_duration),
                'total_duration_formatted': str(pd.Timedelta(seconds=total_duration)),
                'total_segments': int(total_segments),
                'total_chunks_stored': int(total_chunks_stored),
                'advanced_diarization': self.advanced_diarization,
                'model_size': self.model_size,
                'processed_at': datetime.now().isoformat()
            }
            
            results_df.attrs['batch_stats'] = batch_stats
            
            print(f"📊 Batch Processing Complete: {successful_processing}/{total_processed} successful, "
                  f"{error_count} errors, {total_duration/3600:.1f} hours processed, "
                  f"{total_chunks_stored} chunks stored in vector DB")
            
            self.save(results_df)
            
        except Exception as e:
            print(f"❌ Error in batch lecture processing: {e}")
            error_df = pd.DataFrame([{
                'filename': 'BATCH_ERROR',
                'filepath': '',
                'processing_status': 'batch_error',
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


class LectureProcessingSummary(d6tflow.tasks.TaskJson):
    """Create a comprehensive summary of lecture processing results"""
    
    # Task parameters  
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_LECTURE_LIMIT)
    source_directory = d6tflow.OptionalParameter(default=None)
    advanced_diarization = d6tflow.BoolParameter(default=False)
    model_size = d6tflow.Parameter(default="large-v3")
    
    def requires(self):
        return BatchLectureProcessing(
            limit=self.limit,
            source_directory=self.source_directory,
            advanced_diarization=self.advanced_diarization,
            model_size=self.model_size
        )
    
    def run(self):
        """Generate comprehensive lecture processing summary"""
        try:
            # Load batch processing results
            batch_results_df = self.input().load()
            batch_stats = getattr(batch_results_df, 'attrs', {}).get('batch_stats', {})
            
            print("📊 Creating lecture processing summary...")
            
            if batch_results_df.empty:
                summary = {
                    'summary_type': 'lecture_processing',
                    'generated_at': datetime.now().isoformat(),
                    'total_files': 0,
                    'message': 'No lectures were processed',
                    'batch_stats': batch_stats
                }
                self.save(summary)
                return
            
            # Calculate success rates
            successful_files = batch_results_df[batch_results_df['processing_status'] == 'success']
            error_files = batch_results_df[batch_results_df['processing_status'] == 'error']
            
            # Duration analysis
            if not successful_files.empty:
                avg_duration = successful_files['duration_seconds'].mean()
                total_duration = successful_files['duration_seconds'].sum()
                longest_lecture = successful_files.loc[successful_files['duration_seconds'].idxmax()] if len(successful_files) > 0 else None
                shortest_lecture = successful_files.loc[successful_files['duration_seconds'].idxmin()] if len(successful_files) > 0 else None
            else:
                avg_duration = 0
                total_duration = 0
                longest_lecture = None
                shortest_lecture = None
            
            # Transcription analysis
            total_segments = successful_files['segments_count'].sum() if not successful_files.empty else 0
            avg_segments_per_lecture = successful_files['segments_count'].mean() if not successful_files.empty else 0
            
            # Speaker analysis
            avg_speakers_per_lecture = successful_files['speaker_count'].mean() if not successful_files.empty else 0
            max_speakers = successful_files['speaker_count'].max() if not successful_files.empty else 0
            
            # Vector database analysis
            vector_stored_count = len(successful_files[successful_files['vector_stored'] == True]) if not successful_files.empty else 0
            total_chunks_stored = successful_files['chunks_stored'].sum() if not successful_files.empty else 0
            
            # File size analysis
            avg_file_size = batch_results_df['file_size_mb'].mean()
            total_size = batch_results_df['file_size_mb'].sum()
            
            summary = {
                'summary_type': 'lecture_processing',
                'generated_at': datetime.now().isoformat(),
                'total_files': len(batch_results_df),
                'successful_processing': len(successful_files),
                'failed_processing': len(error_files),
                'success_rate': len(successful_files) / len(batch_results_df) * 100 if len(batch_results_df) > 0 else 0,
                
                # Duration analysis
                'average_duration_seconds': round(avg_duration, 2) if avg_duration else 0,
                'average_duration_formatted': str(pd.Timedelta(seconds=avg_duration)) if avg_duration else '0:00:00',
                'total_duration_seconds': round(total_duration, 2),
                'total_duration_formatted': str(pd.Timedelta(seconds=total_duration)),
                'total_duration_hours': round(total_duration / 3600, 2),
                'longest_lecture': {
                    'filename': longest_lecture['filename'] if longest_lecture is not None else None,
                    'duration': round(longest_lecture['duration_seconds'], 2) if longest_lecture is not None else 0,
                    'duration_formatted': str(pd.Timedelta(seconds=longest_lecture['duration_seconds'])) if longest_lecture is not None else '0:00:00'
                },
                'shortest_lecture': {
                    'filename': shortest_lecture['filename'] if shortest_lecture is not None else None,
                    'duration': round(shortest_lecture['duration_seconds'], 2) if shortest_lecture is not None else 0,
                    'duration_formatted': str(pd.Timedelta(seconds=shortest_lecture['duration_seconds'])) if shortest_lecture is not None else '0:00:00'
                },
                
                # Transcription analysis
                'total_transcript_segments': int(total_segments),
                'average_segments_per_lecture': round(avg_segments_per_lecture, 1) if avg_segments_per_lecture else 0,
                'estimated_words': int(total_segments * 15),  # Rough estimate: ~15 words per segment
                
                # Speaker analysis
                'average_speakers_per_lecture': round(avg_speakers_per_lecture, 1) if avg_speakers_per_lecture else 0,
                'max_speakers_in_single_lecture': int(max_speakers) if max_speakers else 0,
                
                # Vector database analysis
                'lectures_with_vector_storage': vector_stored_count,
                'total_vector_chunks_stored': int(total_chunks_stored),
                'vector_storage_rate': round(vector_stored_count / len(successful_files) * 100, 1) if len(successful_files) > 0 else 0,
                
                # File statistics
                'average_file_size_mb': round(avg_file_size, 2) if avg_file_size else 0,
                'total_size_mb': round(total_size, 2),
                'total_size_gb': round(total_size / 1024, 2),
                
                # Processing configuration
                'processing_config': {
                    'model_size': self.model_size,
                    'advanced_diarization': self.advanced_diarization,
                    'limit': self.limit,
                    'source_directory': self.source_directory
                },
                
                # Key insights
                'insights': {
                    'processing_efficiency': f"{len(successful_files)}/{len(batch_results_df)} lectures processed successfully",
                    'content_hours': f"{total_duration/3600:.1f} hours of lecture content processed",
                    'searchable_content': f"{total_chunks_stored} searchable chunks created",
                    'avg_processing_rate': f"{avg_duration/60:.1f} minutes per lecture on average" if avg_duration else "N/A"
                },
                
                'batch_stats': batch_stats
            }
            
            print(f"✅ Lecture Processing Summary: {len(successful_files)}/{len(batch_results_df)} successful, "
                  f"{summary['total_duration_hours']} hours processed, "
                  f"{total_chunks_stored} chunks stored")
            
            self.save(summary)
            
        except Exception as e:
            error_summary = {
                'summary_type': 'lecture_processing',
                'generated_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'total_files': 0,
                'successful_processing': 0,
                'failed_processing': 1
            }
            print(f"❌ Error creating lecture processing summary: {e}")
            self.save(error_summary)


class LectureSearchTask(d6tflow.tasks.TaskJson):
    """Search processed lectures using the vector database"""
    
    # Task parameters
    query = d6tflow.Parameter()
    limit = d6tflow.IntParameter(default=10)
    speaker_filter = d6tflow.OptionalParameter(default=None)
    
    def run(self):
        """Search lectures and return results"""
        try:
            print(f"🔍 Searching lectures: '{self.query}'")
            
            processor = LectureProcessor()
            results = processor.search_lectures(
                query=self.query,
                limit=self.limit,
                speaker=self.speaker_filter
            )
            
            search_summary = {
                'query': self.query,
                'results_count': len(results),
                'speaker_filter': self.speaker_filter,
                'limit': self.limit,
                'searched_at': datetime.now().isoformat(),
                'results': results,
                'task_metadata': {
                    'task_name': 'LectureSearchTask',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            print(f"🔍 Search complete: {len(results)} results found")
            self.save(search_summary)
            
        except Exception as e:
            error_result = {
                'query': self.query,
                'results_count': 0,
                'error': str(e),
                'searched_at': datetime.now().isoformat(),
                'results': [],
                'task_metadata': {
                    'task_name': 'LectureSearchTask',
                    'processed_at': datetime.now().isoformat(),
                    'status': 'error'
                }
            }
            print(f"❌ Error searching lectures: {e}")
            self.save(error_result)


# =============================================================================
# Helper Functions
# =============================================================================

def run_lecture_processing(limit=None, source_directory=None, advanced_diarization=False, model_size="large-v3"):
    """Convenience function to run lecture processing workflow"""
    params = {
        'limit': limit or PipelineConfig.DEFAULT_LECTURE_LIMIT,
        'source_directory': source_directory,
        'advanced_diarization': advanced_diarization,
        'model_size': model_size
    }
    
    task = LectureProcessingSummary(**params)
    d6tflow.run(task)
    
    return task.output().load()


def process_specific_lectures(filepaths: List[str], advanced_diarization=False, model_size="large-v3"):
    """Process specific lecture files"""
    tasks = []
    for filepath in filepaths:
        filename = Path(filepath).name
        tasks.append(ProcessSingleLecture(
            filepath=filepath, 
            filename=filename,
            advanced_diarization=advanced_diarization,
            model_size=model_size
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


def search_lectures(query: str, limit=10, speaker_filter=None):
    """Search processed lectures"""
    task = LectureSearchTask(
        query=query,
        limit=limit,
        speaker_filter=speaker_filter
    )
    
    d6tflow.run(task)
    return task.output().load()
