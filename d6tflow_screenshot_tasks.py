"""
d6tflow Screenshot Analysis Tasks for Media AI Pipeline
Tasks for analyzing screenshots using OpenAI Vision API
"""
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import d6tflow
import pandas as pd

from d6tflow_config import PipelineConfig
from d6tflow_ingestion_tasks import DiscoverScreenshots
from screenshot_analyzer import ScreenshotAnalyzer


class AnalyzeSingleScreenshot(d6tflow.tasks.TaskJson):
    """Analyze a single screenshot using OpenAI Vision API"""
    
    # Task parameters
    filepath = d6tflow.Parameter()
    filename = d6tflow.Parameter()
    
    def run(self):
        """Analyze a single screenshot and save results"""
        try:
            print(f"🔍 Analyzing screenshot: {self.filename}")
            
            analyzer = ScreenshotAnalyzer()
            result = analyzer.analyze_screenshot(self.filepath)
            
            # Add task metadata
            result['task_metadata'] = {
                'task_name': 'AnalyzeSingleScreenshot',
                'processed_at': datetime.now().isoformat(),
                'filepath': self.filepath,
                'filename': self.filename
            }
            
            print(f"✅ Successfully analyzed: {self.filename}")
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
                'task_metadata': {
                    'task_name': 'AnalyzeSingleScreenshot',
                    'processed_at': datetime.now().isoformat(),
                    'status': 'error'
                }
            }
            print(f"❌ Error analyzing {self.filename}: {e}")
            self.save(error_result)


class BatchScreenshotAnalysis(d6tflow.tasks.TaskPqPandas):
    """Analyze multiple screenshots in batches"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_SCREENSHOT_LIMIT)
    batch_size = d6tflow.IntParameter(default=PipelineConfig.BATCH_SIZE_SCREENSHOTS)
    recent_hours = d6tflow.OptionalParameter(default=None)
    source_dir = d6tflow.Parameter(default=PipelineConfig.PNG_SOURCE_DIR)
    
    def requires(self):
        return DiscoverScreenshots(
            limit=self.limit,
            recent_hours=self.recent_hours,
            source_dir=self.source_dir
        )
    
    def run(self):
        """Analyze screenshots in batches"""
        try:
            # Load discovered screenshots
            screenshots_df = self.input().load()
            
            if screenshots_df.empty:
                print("📸 No screenshots found to analyze")
                self.save(pd.DataFrame(columns=['filename', 'filepath', 'analysis_status', 'processed_at']))
                return
            
            print(f"📸 Starting batch analysis of {len(screenshots_df)} screenshots...")
            
            analyzer = ScreenshotAnalyzer()
            results = []
            
            # Process screenshots in batches
            for i in range(0, len(screenshots_df), self.batch_size):
                batch = screenshots_df.iloc[i:i + self.batch_size]
                print(f"🔄 Processing batch {i//self.batch_size + 1}: {len(batch)} files")
                
                for idx, row in batch.iterrows():
                    try:
                        filepath = row['filepath']
                        filename = row['filename']
                        
                        print(f"  🔍 Analyzing: {filename}")
                        analysis_result = analyzer.analyze_screenshot(filepath)
                        
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
                            'code_detected': analysis_result.get('analysis', {}).get('code_detected', False),
                            'error_detected': analysis_result.get('analysis', {}).get('error_detected', False),
                            'tokens_used': analysis_result.get('tokens_used', 0),
                            'has_error': False,
                            'summary': analysis_result.get('analysis', {}).get('summary', '')[:200] + '...' if analysis_result.get('analysis', {}).get('summary', '') else ''
                        }
                        
                        results.append(analysis_summary)
                        print(f"    ✅ Success: {filename}")
                        
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
                            'code_detected': False,
                            'error_detected': False,
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
            
            batch_stats = {
                'total_files': total_analyzed,
                'successful': successful_analyses,
                'errors': error_count,
                'total_tokens_used': int(total_tokens),
                'batch_size': self.batch_size,
                'processed_at': datetime.now().isoformat()
            }
            
            results_df.attrs['batch_stats'] = batch_stats
            
            print(f"📊 Batch Analysis Complete: {successful_analyses}/{total_analyzed} successful, "
                  f"{error_count} errors, {total_tokens} tokens used")
            
            self.save(results_df)
            
        except Exception as e:
            print(f"❌ Error in batch screenshot analysis: {e}")
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


class ScreenshotAnalysisSummary(d6tflow.tasks.TaskJson):
    """Create a comprehensive summary of screenshot analysis results"""
    
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
        """Generate comprehensive analysis summary"""
        try:
            # Load batch analysis results
            batch_results_df = self.input().load()
            batch_stats = getattr(batch_results_df, 'attrs', {}).get('batch_stats', {})
            
            print("📊 Creating screenshot analysis summary...")
            
            if batch_results_df.empty:
                summary = {
                    'summary_type': 'screenshot_analysis',
                    'generated_at': datetime.now().isoformat(),
                    'total_files': 0,
                    'message': 'No screenshots were analyzed',
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
            
            # Analyze patterns
            code_screenshots = len(successful_files[successful_files['code_detected'] == True])
            error_screenshots = len(successful_files[successful_files['error_detected'] == True])
            
            # File size analysis
            avg_file_size = batch_results_df['file_size_mb'].mean()
            total_size = batch_results_df['file_size_mb'].sum()
            
            # Recent vs older files
            recent_files = 0
            if self.recent_hours:
                cutoff_time = datetime.now() - pd.Timedelta(hours=self.recent_hours)
                recent_files = len(batch_results_df[batch_results_df['modified_at'] >= cutoff_time])
            
            summary = {
                'summary_type': 'screenshot_analysis',
                'generated_at': datetime.now().isoformat(),
                'total_files': len(batch_results_df),
                'successful_analyses': len(successful_files),
                'failed_analyses': len(error_files),
                'success_rate': len(successful_files) / len(batch_results_df) * 100 if len(batch_results_df) > 0 else 0,
                
                # Content analysis
                'content_types': content_types,
                'primary_applications': applications,
                'code_screenshots': code_screenshots,
                'error_screenshots': error_screenshots,
                
                # File statistics
                'average_file_size_mb': round(avg_file_size, 2) if avg_file_size else 0,
                'total_size_mb': round(total_size, 2),
                'recent_files': recent_files,
                
                # Processing statistics
                'total_tokens_used': batch_stats.get('total_tokens_used', 0),
                'processing_time': batch_stats.get('processed_at'),
                'batch_size': self.batch_size,
                
                # Configuration
                'parameters': {
                    'limit': self.limit,
                    'batch_size': self.batch_size,
                    'recent_hours': self.recent_hours,
                    'source_dir': self.source_dir
                },
                
                # Top insights
                'insights': {
                    'most_common_content_type': max(content_types.items(), key=lambda x: x[1])[0] if content_types else None,
                    'most_common_application': max(applications.items(), key=lambda x: x[1])[0] if applications else None,
                    'code_percentage': round(code_screenshots / len(successful_files) * 100, 1) if len(successful_files) > 0 else 0,
                    'error_percentage': round(error_screenshots / len(successful_files) * 100, 1) if len(successful_files) > 0 else 0
                },
                
                'batch_stats': batch_stats
            }
            
            print(f"✅ Analysis Summary: {len(successful_files)}/{len(batch_results_df)} successful, "
                  f"{summary['insights']['most_common_content_type']} most common, "
                  f"{code_screenshots} with code")
            
            self.save(summary)
            
        except Exception as e:
            error_summary = {
                'summary_type': 'screenshot_analysis',
                'generated_at': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'total_files': 0,
                'successful_analyses': 0,
                'failed_analyses': 1
            }
            print(f"❌ Error creating analysis summary: {e}")
            self.save(error_summary)


# =============================================================================
# Helper Functions
# =============================================================================

def run_screenshot_analysis(limit=None, batch_size=None, recent_hours=None):
    """Convenience function to run screenshot analysis workflow"""
    params = {
        'limit': limit or PipelineConfig.DEFAULT_SCREENSHOT_LIMIT,
        'batch_size': batch_size or PipelineConfig.BATCH_SIZE_SCREENSHOTS,
        'recent_hours': recent_hours
    }
    
    task = ScreenshotAnalysisSummary(**params)
    d6tflow.run(task)
    
    return task.output().load()


def analyze_specific_screenshots(filepaths: List[str]):
    """Analyze specific screenshot files"""
    tasks = []
    for filepath in filepaths:
        filename = Path(filepath).name
        tasks.append(AnalyzeSingleScreenshot(filepath=filepath, filename=filename))
    
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
