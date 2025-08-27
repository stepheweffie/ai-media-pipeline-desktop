"""
d6tflow Data Ingestion Tasks for Media AI Pipeline
Tasks for discovering and collecting screenshots, videos, and lecture files
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import d6tflow
import pandas as pd

from d6tflow_config import PipelineConfig


class DiscoverScreenshots(d6tflow.tasks.TaskPqPandas):
    """Discover and catalog PNG screenshots in the source directory"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_SCREENSHOT_LIMIT)
    recent_hours = d6tflow.OptionalParameter(default=None)
    source_dir = d6tflow.Parameter(default=PipelineConfig.PNG_SOURCE_DIR)
    
    def run(self):
        """Discover screenshot files and create a catalog"""
        try:
            source_path = Path(self.source_dir)
            if not source_path.exists():
                raise FileNotFoundError(f"Source directory not found: {source_path}")
            
            print(f"🔍 Discovering screenshots in {source_path}")
            
            # Find all supported image files
            screenshots = []
            for ext in PipelineConfig.SUPPORTED_IMAGE_FORMATS:
                pattern = f"*{ext}"
                for file_path in source_path.glob(pattern):
                    if file_path.is_file():
                        stat = file_path.stat()
                        screenshots.append({
                            'filename': file_path.name,
                            'filepath': str(file_path),
                            'file_size_bytes': stat.st_size,
                            'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                            'created_at': datetime.fromtimestamp(stat.st_ctime),
                            'modified_at': datetime.fromtimestamp(stat.st_mtime),
                            'discovered_at': datetime.now(),
                            'file_type': 'screenshot',
                            'extension': ext
                        })
            
            # Convert to DataFrame and sort by modification time (newest first)
            df = pd.DataFrame(screenshots)
            if not df.empty:
                df = df.sort_values('modified_at', ascending=False)
                
                # Filter by recent hours if specified
                if self.recent_hours is not None:
                    cutoff_time = datetime.now() - timedelta(hours=self.recent_hours)
                    df = df[df['modified_at'] >= cutoff_time]
                
                # Apply limit
                if self.limit is not None and self.limit > 0:
                    df = df.head(self.limit)
            
            print(f"📸 Found {len(df)} screenshots to process")
            
            # Save the catalog
            self.save(df)
            
        except Exception as e:
            print(f"❌ Error discovering screenshots: {e}")
            # Save empty DataFrame on error
            self.save(pd.DataFrame())


class DiscoverVideos(d6tflow.tasks.TaskPqPandas):
    """Discover and catalog MP4 screen recordings in the source directory"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_VIDEO_LIMIT)
    recent_hours = d6tflow.OptionalParameter(default=None)
    source_dir = d6tflow.Parameter(default=PipelineConfig.MP4_SOURCE_DIR)
    
    def run(self):
        """Discover video files and create a catalog"""
        try:
            source_path = Path(self.source_dir)
            if not source_path.exists():
                raise FileNotFoundError(f"Source directory not found: {source_path}")
            
            print(f"🔍 Discovering videos in {source_path}")
            
            # Find all supported video files
            videos = []
            for ext in PipelineConfig.SUPPORTED_VIDEO_FORMATS:
                pattern = f"*{ext}"
                for file_path in source_path.glob(pattern):
                    if file_path.is_file():
                        stat = file_path.stat()
                        # Skip files that are too large
                        file_size_mb = stat.st_size / (1024 * 1024)
                        if file_size_mb > PipelineConfig.MAX_FILE_SIZE_MB:
                            print(f"⚠️  Skipping large file: {file_path.name} ({file_size_mb:.1f}MB)")
                            continue
                            
                        videos.append({
                            'filename': file_path.name,
                            'filepath': str(file_path),
                            'file_size_bytes': stat.st_size,
                            'file_size_mb': round(file_size_mb, 2),
                            'created_at': datetime.fromtimestamp(stat.st_ctime),
                            'modified_at': datetime.fromtimestamp(stat.st_mtime),
                            'discovered_at': datetime.now(),
                            'file_type': 'video',
                            'extension': ext
                        })
            
            # Convert to DataFrame and sort by modification time (newest first)
            df = pd.DataFrame(videos)
            if not df.empty:
                df = df.sort_values('modified_at', ascending=False)
                
                # Filter by recent hours if specified
                if self.recent_hours is not None:
                    cutoff_time = datetime.now() - timedelta(hours=self.recent_hours)
                    df = df[df['modified_at'] >= cutoff_time]
                
                # Apply limit
                if self.limit is not None and self.limit > 0:
                    df = df.head(self.limit)
            
            print(f"🎥 Found {len(df)} videos to process")
            
            # Save the catalog
            self.save(df)
            
        except Exception as e:
            print(f"❌ Error discovering videos: {e}")
            # Save empty DataFrame on error
            self.save(pd.DataFrame())


class DiscoverLectures(d6tflow.tasks.TaskPqPandas):
    """Discover lecture video files for transcription"""
    
    # Task parameters
    limit = d6tflow.IntParameter(default=PipelineConfig.DEFAULT_LECTURE_LIMIT)
    source_directory = d6tflow.OptionalParameter(default=None)
    min_duration_minutes = d6tflow.IntParameter(default=5)  # Minimum duration to be considered a lecture
    
    def run(self):
        """Discover lecture files"""
        try:
            # Use provided directory or fall back to video directory
            source_dir = self.source_directory or PipelineConfig.MP4_SOURCE_DIR
            source_path = Path(source_dir)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Lecture source directory not found: {source_path}")
            
            print(f"🔍 Discovering lecture files in {source_path}")
            
            # Find video files that could be lectures
            lectures = []
            for ext in PipelineConfig.SUPPORTED_VIDEO_FORMATS:
                pattern = f"*{ext}"
                for file_path in source_path.glob(pattern):
                    if file_path.is_file():
                        stat = file_path.stat()
                        file_size_mb = stat.st_size / (1024 * 1024)
                        
                        # Skip very small files (likely not lectures)
                        if file_size_mb < 10:  # Assume lectures are at least 10MB
                            continue
                            
                        # Skip files that are too large for processing
                        if file_size_mb > PipelineConfig.MAX_FILE_SIZE_MB * 2:  # Allow larger files for lectures
                            print(f"⚠️  Skipping very large file: {file_path.name} ({file_size_mb:.1f}MB)")
                            continue
                        
                        lectures.append({
                            'filename': file_path.name,
                            'filepath': str(file_path),
                            'file_size_bytes': stat.st_size,
                            'file_size_mb': round(file_size_mb, 2),
                            'created_at': datetime.fromtimestamp(stat.st_ctime),
                            'modified_at': datetime.fromtimestamp(stat.st_mtime),
                            'discovered_at': datetime.now(),
                            'file_type': 'lecture',
                            'extension': ext
                        })
            
            # Convert to DataFrame and sort by modification time (newest first)
            df = pd.DataFrame(lectures)
            if not df.empty:
                df = df.sort_values('modified_at', ascending=False)
                
                # Apply limit
                if self.limit is not None and self.limit > 0:
                    df = df.head(self.limit)
            
            print(f"🎓 Found {len(df)} lecture files to process")
            
            # Save the catalog
            self.save(df)
            
        except Exception as e:
            print(f"❌ Error discovering lectures: {e}")
            # Save empty DataFrame on error
            self.save(pd.DataFrame())


class MediaInventorySummary(d6tflow.tasks.TaskPqPandas):
    """Create a comprehensive inventory of all discovered media files"""
    
    def requires(self):
        return {
            'screenshots': DiscoverScreenshots(),
            'videos': DiscoverVideos(), 
            'lectures': DiscoverLectures()
        }
    
    def run(self):
        """Combine all discovered media into a comprehensive inventory"""
        try:
            # Load all discovery results
            screenshots_df = self.input()['screenshots'].load()
            videos_df = self.input()['videos'].load()
            lectures_df = self.input()['lectures'].load()
            
            print("📊 Creating media inventory summary...")
            
            # Combine all media files
            all_media = []
            
            # Add screenshots
            if not screenshots_df.empty:
                all_media.append(screenshots_df.copy())
            
            # Add videos (excluding those already counted as lectures)
            if not videos_df.empty:
                videos_only = videos_df.copy()
                if not lectures_df.empty:
                    # Remove videos that are also in lectures
                    lecture_files = set(lectures_df['filepath'].tolist())
                    videos_only = videos_only[~videos_only['filepath'].isin(lecture_files)]
                all_media.append(videos_only)
            
            # Add lectures
            if not lectures_df.empty:
                all_media.append(lectures_df.copy())
            
            # Combine into single DataFrame
            if all_media:
                inventory_df = pd.concat(all_media, ignore_index=True)
                inventory_df = inventory_df.sort_values('modified_at', ascending=False)
            else:
                inventory_df = pd.DataFrame()
            
            # Add summary statistics
            summary_stats = {
                'total_files': len(inventory_df),
                'screenshots': len(screenshots_df),
                'videos': len(videos_df) - len(lectures_df) if not lectures_df.empty else len(videos_df),
                'lectures': len(lectures_df),
                'total_size_mb': inventory_df['file_size_mb'].sum() if not inventory_df.empty else 0,
                'created_at': datetime.now().isoformat(),
                'oldest_file': inventory_df['modified_at'].min().isoformat() if not inventory_df.empty else None,
                'newest_file': inventory_df['modified_at'].max().isoformat() if not inventory_df.empty else None
            }
            
            print(f"📈 Inventory Summary: {summary_stats['total_files']} total files "
                  f"({summary_stats['screenshots']} screenshots, "
                  f"{summary_stats['videos']} videos, "
                  f"{summary_stats['lectures']} lectures)")
            
            # Add summary as metadata to the DataFrame
            inventory_df.attrs['summary_stats'] = summary_stats
            
            self.save(inventory_df)
            
        except Exception as e:
            print(f"❌ Error creating inventory summary: {e}")
            # Save empty DataFrame on error
            empty_df = pd.DataFrame()
            empty_df.attrs['summary_stats'] = {
                'total_files': 0,
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
            self.save(empty_df)


# =============================================================================
# Helper Functions
# =============================================================================

def run_media_discovery(screenshot_limit=None, video_limit=None, lecture_limit=None):
    """Convenience function to run media discovery tasks"""
    tasks = []
    
    if screenshot_limit is not None:
        tasks.append(DiscoverScreenshots(limit=screenshot_limit))
    
    if video_limit is not None:
        tasks.append(DiscoverVideos(limit=video_limit))
    
    if lecture_limit is not None:
        tasks.append(DiscoverLectures(limit=lecture_limit))
    
    # Always run the summary
    tasks.append(MediaInventorySummary())
    
    # Execute tasks
    d6tflow.run(tasks)
    
    return MediaInventorySummary().output().load()
