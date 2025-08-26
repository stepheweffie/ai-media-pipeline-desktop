"""
Video Analyzer for Media AI Pipeline
Extracts keyframes from MP4 screen recordings and analyzes them
"""
import os
import subprocess
import tempfile
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import openai
from screenshot_analyzer import ScreenshotAnalyzer
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        """Initialize the video analyzer"""
        config.validate_config()
        openai.api_key = config.OPENAI_API_KEY
        self.screenshot_analyzer = ScreenshotAnalyzer()
        
    def get_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata using ffprobe"""
        try:
            # Use ffprobe to get video info
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {video_path}, using basic metadata")
                return self.get_basic_metadata(video_path)
            
            probe_data = json.loads(result.stdout)
            
            # Extract video stream info
            video_streams = [s for s in probe_data['streams'] if s['codec_type'] == 'video']
            if not video_streams:
                logger.warning(f"No video streams found in {video_path}")
                return self.get_basic_metadata(video_path)
            
            video_stream = video_streams[0]
            format_data = probe_data['format']
            
            # Calculate duration
            duration_seconds = float(format_data.get('duration', 0))
            duration_str = str(timedelta(seconds=int(duration_seconds)))
            
            # Basic file metadata
            stat = os.stat(video_path)
            
            metadata = {
                'filename': os.path.basename(video_path),
                'filepath': video_path,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'analyzed_at': datetime.now().isoformat(),
                'duration_seconds': duration_seconds,
                'duration_formatted': duration_str,
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'bitrate': int(format_data.get('bit_rate', 0))
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting video metadata for {video_path}: {e}")
            return self.get_basic_metadata(video_path)
    
    def get_basic_metadata(self, video_path: str) -> Dict:
        """Get basic file metadata when ffprobe is not available"""
        try:
            stat = os.stat(video_path)
            return {
                'filename': os.path.basename(video_path),
                'filepath': video_path,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'analyzed_at': datetime.now().isoformat(),
                'duration_seconds': None,
                'duration_formatted': 'unknown',
                'ffprobe_available': False
            }
        except Exception as e:
            logger.error(f"Error getting basic metadata for {video_path}: {e}")
            return {'filename': os.path.basename(video_path), 'error': str(e)}
    
    def extract_keyframes(self, video_path: str, num_frames: int = 5, output_dir: Optional[str] = None) -> List[str]:
        """Extract keyframes from video using ffmpeg"""
        try:
            # Create temporary directory if not provided
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix='video_analysis_')
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            # Get video metadata to determine frame extraction strategy
            metadata = self.get_video_metadata(video_path)
            duration = metadata.get('duration_seconds', 0)
            
            if duration <= 0:
                logger.warning(f"Cannot determine video duration for {video_path}, extracting frames manually")
                duration = 60  # Default assumption
            
            # Calculate frame timestamps
            if duration > num_frames:
                # Spread frames evenly across the video, skipping first and last 5%
                start_time = duration * 0.05
                end_time = duration * 0.95
                interval = (end_time - start_time) / (num_frames - 1) if num_frames > 1 else 0
                timestamps = [start_time + i * interval for i in range(num_frames)]
            else:
                # For very short videos, extract frames at 1-second intervals
                timestamps = [i for i in range(min(int(duration), num_frames))]
            
            # Extract frames
            frame_paths = []
            for i, timestamp in enumerate(timestamps):
                output_path = os.path.join(output_dir, f"frame_{i:03d}_{int(timestamp)}s.png")
                
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(timestamp),
                    '-vframes', '1',
                    '-y',  # Overwrite output files
                    '-loglevel', 'quiet',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0 and os.path.exists(output_path):
                    frame_paths.append(output_path)
                    logger.info(f"Extracted frame {i+1}/{num_frames} at {timestamp:.1f}s")
                else:
                    logger.warning(f"Failed to extract frame at {timestamp}s from {video_path}")
            
            logger.info(f"Successfully extracted {len(frame_paths)} frames from {video_path}")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting keyframes from {video_path}: {e}")
            return []
    
    def analyze_video(self, video_path: str, num_keyframes: int = 5) -> Dict:
        """Analyze a video by extracting keyframes and analyzing each one"""
        try:
            logger.info(f"Analyzing video: {os.path.basename(video_path)}")
            
            # Get video metadata
            metadata = self.get_video_metadata(video_path)
            
            # Check file size
            if metadata.get('file_size_mb', 0) > config.MAX_FILE_SIZE_MB * 10:  # Allow larger files for video
                logger.warning(f"Video {video_path} is very large ({metadata['file_size_mb']}MB)")
            
            # Extract keyframes
            with tempfile.TemporaryDirectory(prefix='video_analysis_') as temp_dir:
                frame_paths = self.extract_keyframes(video_path, num_keyframes, temp_dir)
                
                if not frame_paths:
                    return {
                        'metadata': metadata,
                        'analysis': {'error': 'Failed to extract keyframes from video'},
                        'keyframes': []
                    }
                
                # Analyze each keyframe
                keyframe_analyses = []
                for frame_path in frame_paths:
                    try:
                        frame_analysis = self.screenshot_analyzer.analyze_screenshot(frame_path)
                        # Add frame timestamp info
                        frame_name = os.path.basename(frame_path)
                        timestamp_match = frame_name.split('_')[-1].replace('s.png', '')
                        frame_analysis['timestamp_seconds'] = int(timestamp_match)
                        keyframe_analyses.append(frame_analysis)
                    except Exception as e:
                        logger.warning(f"Failed to analyze frame {frame_path}: {e}")
                        continue
                
                # Generate video summary
                video_summary = self.generate_video_summary(keyframe_analyses, metadata)
                
                result = {
                    'metadata': metadata,
                    'analysis': video_summary,
                    'keyframes': keyframe_analyses,
                    'total_tokens_used': sum(kf.get('tokens_used', 0) for kf in keyframe_analyses)
                }
                
                logger.info(f"Successfully analyzed video {os.path.basename(video_path)} with {len(keyframe_analyses)} keyframes")
                return result
                
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            return {
                'metadata': self.get_basic_metadata(video_path) if os.path.exists(video_path) else {},
                'analysis': {'error': str(e)},
                'keyframes': []
            }
    
    def generate_video_summary(self, keyframe_analyses: List[Dict], metadata: Dict) -> Dict:
        """Generate a comprehensive summary of the video based on keyframe analyses"""
        try:
            if not keyframe_analyses:
                return {'error': 'No keyframe analyses available'}
            
            # Extract common patterns
            content_types = []
            applications = []
            all_tags = []
            code_detected = False
            errors_detected = []
            technical_details = []
            workflows = []
            
            for kf in keyframe_analyses:
                analysis = kf.get('analysis', {})
                if 'error' in analysis:
                    continue
                
                content_types.append(analysis.get('content_type', ''))
                applications.append(analysis.get('primary_application', ''))
                all_tags.extend(analysis.get('context_tags', []))
                
                if analysis.get('code_detected'):
                    code_detected = True
                
                if analysis.get('error_detected'):
                    errors_detected.append(analysis.get('error_details', ''))
                
                if analysis.get('technical_details'):
                    technical_details.append(analysis.get('technical_details', ''))
                
                if analysis.get('workflow_step'):
                    workflows.append(analysis.get('workflow_step', ''))
            
            # Find most common content type and application
            most_common_type = max(set(content_types), key=content_types.count) if content_types else 'unknown'
            most_common_app = max(set(applications), key=applications.count) if applications else 'unknown'
            
            # Create summary
            duration_str = metadata.get('duration_formatted', 'unknown duration')
            summary_parts = [
                f"Screen recording ({duration_str})",
                f"primarily showing {most_common_type} content",
                f"in {most_common_app}"
            ]
            
            if code_detected:
                summary_parts.append("with code editing/development")
            
            if errors_detected:
                summary_parts.append(f"including {len(errors_detected)} error(s)")
            
            summary = ' '.join(summary_parts) + '.'
            
            # Generate searchable keywords
            keywords = list(set(all_tags + applications + content_types))
            keywords = [k for k in keywords if k and k != 'unknown']
            
            return {
                'content_type': most_common_type,
                'primary_application': most_common_app,
                'summary': summary,
                'duration_formatted': duration_str,
                'code_detected': code_detected,
                'errors_detected': len(errors_detected) > 0,
                'error_count': len(errors_detected),
                'context_tags': list(set(all_tags)),
                'technical_details': list(set(technical_details)),
                'workflow_steps': list(set(workflows)),
                'searchable_keywords': keywords,
                'keyframe_count': len(keyframe_analyses)
            }
            
        except Exception as e:
            logger.error(f"Error generating video summary: {e}")
            return {'error': f'Summary generation failed: {str(e)}'}
    
    def analyze_videos_in_directory(self, directory: str, limit: Optional[int] = None) -> List[Dict]:
        """Analyze all videos in a directory"""
        try:
            logger.info(f"Analyzing videos in directory: {directory}")
            
            if not os.path.exists(directory):
                logger.error(f"Directory not found: {directory}")
                return []
            
            # Find all supported video files
            video_files = []
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in config.SUPPORTED_VIDEO_FORMATS):
                    video_files.append(os.path.join(directory, filename))
            
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Apply limit if specified
            if limit:
                video_files = video_files[:limit]
            
            logger.info(f"Found {len(video_files)} video files to analyze")
            
            # Analyze each video
            results = []
            for video_path in video_files:
                try:
                    result = self.analyze_video(video_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze {video_path}: {e}")
                    continue
            
            logger.info(f"Successfully analyzed {len(results)} videos")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing directory {directory}: {e}")
            return []

def main():
    """Example usage"""
    analyzer = VideoAnalyzer()
    
    # Analyze recent videos
    mp4_directory = os.path.join(config.DESKTOP_PATH, config.MP4_PATH.lstrip('../'))
    results = analyzer.analyze_videos_in_directory(mp4_directory, limit=2)
    
    # Print results
    for result in results:
        print(f"\n--- {result['metadata']['filename']} ---")
        analysis = result['analysis']
        if 'error' not in analysis:
            print(f"Duration: {analysis.get('duration_formatted', 'unknown')}")
            print(f"Type: {analysis.get('content_type', 'unknown')}")
            print(f"App: {analysis.get('primary_application', 'unknown')}")
            print(f"Summary: {analysis.get('summary', 'no summary')}")
            print(f"Keyframes analyzed: {analysis.get('keyframe_count', 0)}")
        else:
            print(f"Error: {analysis['error']}")

if __name__ == "__main__":
    main()
