#!/usr/bin/env python3
"""
Lecture Processing CLI for Media AI Pipeline
Command-line interface for bulk lecture transcription and search
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

from lecture_processor import LectureProcessor
import config

def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_video_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all video files in directory"""
    if extensions is None:
        extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
    
    video_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return []
    
    for ext in extensions:
        video_files.extend(directory_path.glob(f"**/*{ext}"))
        video_files.extend(directory_path.glob(f"**/*{ext.upper()}"))
    
    return [str(f) for f in video_files]

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def cmd_process(args):
    """Process lecture videos"""
    processor = LectureProcessor(args.db_path)
    
    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} does not exist")
            return 1
        
        print(f"Processing single video: {args.file}")
        result = processor.process_video(args.file, args.advanced_diarization)
        
        if result["status"] == "success":
            print("✅ Processing successful!")
            print(f"   Duration: {format_duration(result['duration'])}")
            print(f"   Speakers: {', '.join(result['speakers'])}")
            print(f"   Segments: {result['total_segments']}")
            print(f"   Chunks: {result['total_chunks']}")
            print(f"   Processing time: {result['processing_time']}")
        else:
            print(f"❌ Processing failed: {result['error']}")
            return 1
    
    elif args.directory:
        # Process directory
        video_files = find_video_files(args.directory)
        
        if not video_files:
            print(f"No video files found in {args.directory}")
            return 1
        
        # Sort by modification time (newest first) and apply limit
        video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if args.limit:
            video_files = video_files[:args.limit]
        
        print(f"Found {len(video_files)} video files to process")
        
        if not args.yes:
            response = input(f"Process {len(video_files)} videos? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0
        
        # Process batch
        results = processor.process_video_batch(video_files, args.advanced_diarization)
        
        # Summary
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        
        if successful:
            total_duration = sum(r.get("duration", 0) for r in successful)
            total_chunks = sum(r.get("total_chunks", 0) for r in successful)
            speakers = set()
            for r in successful:
                speakers.update(r.get("speakers", []))
            
            print(f"   🎥 Total content: {format_duration(total_duration)}")
            print(f"   📝 Total chunks: {total_chunks}")
            print(f"   🗣️  Unique speakers: {len(speakers)}")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                print(f"   - {os.path.basename(result['video_file'])}: {result['error']}")
    
    else:
        print("Error: Must specify either --file or --directory")
        return 1
    
    return 0

def cmd_search(args):
    """Search lecture transcripts"""
    processor = LectureProcessor(args.db_path)
    
    results = processor.search_lectures(
        query=args.query,
        n_results=args.limit,
        speaker_filter=args.speaker,
        video_filter=args.video
    )
    
    if not results:
        print("No results found.")
        return 0
    
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        text = result["text"]
        
        print(f"🔍 Result {i} (similarity: {1 - result['distance']:.3f})")
        print(f"   🎥 Video: {os.path.basename(metadata['video_file'])}")
        print(f"   🗣️  Speaker: {metadata['speaker']}")
        print(f"   ⏱️  Time: {metadata['timestamp_formatted']}")
        print(f"   📝 Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print()
    
    return 0

def cmd_stats(args):
    """Show database statistics"""
    processor = LectureProcessor(args.db_path)
    stats = processor.get_statistics()
    
    if "error" in stats:
        print(f"Error getting statistics: {stats['error']}")
        return 1
    
    print("📊 Lecture Database Statistics\n")
    print(f"   📝 Total chunks: {stats['total_chunks']:,}")
    print(f"   🎥 Unique videos: {stats['unique_videos']}")
    print(f"   🗣️  Unique speakers: {stats['unique_speakers']}")
    print(f"   ⏱️  Total content: {stats['total_content_hours']} hours")
    
    if stats.get('speakers'):
        print(f"\n🗣️  Speakers found:")
        for speaker in sorted(stats['speakers']):
            print(f"   - {speaker}")
    
    if stats.get('videos'):
        print(f"\n🎥 Videos processed:")
        for video in sorted(stats['videos'])[:10]:  # Show first 10
            print(f"   - {video}")
        if len(stats['videos']) > 10:
            print(f"   ... and {len(stats['videos']) - 10} more")
    
    return 0

def cmd_export(args):
    """Export transcripts"""
    processor = LectureProcessor(args.db_path)
    
    # Get all chunks
    try:
        collection = processor.collection
        all_results = collection.get()
        
        if not all_results["documents"]:
            print("No transcripts found to export.")
            return 0
        
        # Organize by video and speaker
        export_data = {}
        
        for i, doc in enumerate(all_results["documents"]):
            metadata = all_results["metadatas"][i]
            video_file = metadata["video_file"]
            speaker = metadata["speaker"]
            
            if video_file not in export_data:
                export_data[video_file] = {}
            
            if speaker not in export_data[video_file]:
                export_data[video_file][speaker] = []
            
            export_data[video_file][speaker].append({
                "text": doc,
                "start_time": metadata["start_time"],
                "end_time": metadata["end_time"],
                "timestamp": metadata["timestamp_formatted"]
            })
        
        # Sort by timestamp
        for video in export_data:
            for speaker in export_data[video]:
                export_data[video][speaker].sort(key=lambda x: x["start_time"])
        
        # Write to file
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == 'json':
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:  # text format
                for video_file, speakers in export_data.items():
                    f.write(f"=== {os.path.basename(video_file)} ===\n\n")
                    
                    # Combine all speakers and sort by time
                    all_segments = []
                    for speaker, segments in speakers.items():
                        for segment in segments:
                            segment["speaker"] = speaker
                            all_segments.append(segment)
                    
                    all_segments.sort(key=lambda x: x["start_time"])
                    
                    for segment in all_segments:
                        f.write(f"[{segment['timestamp']}] {segment['speaker']}: {segment['text']}\n")
                    
                    f.write("\n" + "="*50 + "\n\n")
        
        print(f"Exported transcripts to {args.output}")
        print(f"Videos: {len(export_data)}")
        total_chunks = sum(len(speakers) for speakers in export_data.values() for segments in speakers.values())
        print(f"Total segments: {total_chunks}")
        
    except Exception as e:
        print(f"Error exporting: {e}")
        return 1
    
    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Lecture Processing CLI for Media AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python lecture_cli.py process --file lecture1.mp4
  
  # Process all videos in a directory (with advanced diarization)
  python lecture_cli.py process --directory ./lectures --advanced
  
  # Process only the 5 most recent videos
  python lecture_cli.py process --directory ./lectures --limit 5
  
  # Search for content about machine learning
  python lecture_cli.py search "machine learning algorithms"
  
  # Search for content by a specific speaker
  python lecture_cli.py search "neural networks" --speaker "Speaker_1"
  
  # Export all transcripts to a file
  python lecture_cli.py export transcripts.txt
  
  # Show database statistics
  python lecture_cli.py stats
        """
    )
    
    parser.add_argument('--db-path', default='./lecture_vectors',
                       help='Path to vector database (default: ./lecture_vectors)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process lecture videos')
    process_group = process_parser.add_mutually_exclusive_group(required=True)
    process_group.add_argument('--file', help='Process a single video file')
    process_group.add_argument('--directory', help='Process all videos in directory')
    process_parser.add_argument('--limit', type=int, help='Limit number of videos to process')
    process_parser.add_argument('--advanced', dest='advanced_diarization', action='store_true',
                               help='Use advanced diarization (requires pyannote.audio)')
    process_parser.add_argument('--yes', '-y', action='store_true',
                               help='Auto-confirm batch processing')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search lecture transcripts')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10,
                              help='Number of results to return (default: 10)')
    search_parser.add_argument('--speaker', help='Filter by speaker')
    search_parser.add_argument('--video', help='Filter by video filename')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export transcripts')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--format', choices=['text', 'json'], default='text',
                              help='Export format (default: text)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Validate config
    try:
        config.validate_config()
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file and API keys.")
        return 1
    
    # Route to appropriate command
    if args.command == 'process':
        return cmd_process(args)
    elif args.command == 'search':
        return cmd_search(args)
    elif args.command == 'stats':
        return cmd_stats(args)
    elif args.command == 'export':
        return cmd_export(args)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
