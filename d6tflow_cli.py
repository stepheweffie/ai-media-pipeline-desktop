#!/usr/bin/env python3
"""
d6tflow CLI for Media AI Pipeline
Command-line interface for running d6tflow-based media analysis workflows
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Import d6tflow pipeline functions
from d6tflow_pipeline import (
    run_full_pipeline, 
    run_screenshot_pipeline, 
    run_video_pipeline, 
    run_lecture_pipeline,
    get_pipeline_status,
    query_pipeline_results,
    export_pipeline_results
)
from d6tflow_storage_tasks import cleanup_data
from d6tflow_config import PipelineConfig


def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="d6tflow Media AI Pipeline - Analyze screenshots, videos, and lectures with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python d6tflow_cli.py run --full
  
  # Run screenshot analysis only
  python d6tflow_cli.py run --screenshots --limit 5
  
  # Run video analysis with custom keyframes
  python d6tflow_cli.py run --videos --limit 3 --keyframes 7
  
  # Run lecture processing with advanced diarization
  python d6tflow_cli.py run --lectures --limit 2 --advanced-diarization
  
  # Query processed media
  python d6tflow_cli.py query "screenshots with Python code"
  
  # Export results
  python d6tflow_cli.py export --format json --output results.json
  
  # Check pipeline status
  python d6tflow_cli.py status
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =============================================================================
    # RUN command
    # =============================================================================
    run_parser = subparsers.add_parser('run', help='Run analysis pipelines')
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--full', action='store_true', help='Run complete pipeline (screenshots + videos + lectures)')
    run_group.add_argument('--screenshots', action='store_true', help='Run screenshot analysis only')
    run_group.add_argument('--videos', action='store_true', help='Run video analysis only')
    run_group.add_argument('--lectures', action='store_true', help='Run lecture processing only')
    
    # Common parameters
    run_parser.add_argument('--limit', type=int, help='Limit number of files to process')
    run_parser.add_argument('--recent-hours', type=int, help='Only process files modified in last N hours')
    run_parser.add_argument('--no-airtable', action='store_true', help='Disable Airtable storage')
    
    # Screenshot-specific parameters
    run_parser.add_argument('--batch-size', type=int, help='Screenshot batch size for processing')
    
    # Video-specific parameters
    run_parser.add_argument('--keyframes', type=int, default=5, help='Number of keyframes to extract per video')
    
    # Lecture-specific parameters
    run_parser.add_argument('--lecture-dir', type=str, help='Directory containing lecture videos')
    run_parser.add_argument('--advanced-diarization', action='store_true', help='Use advanced speaker diarization (requires pyannote.audio)')
    run_parser.add_argument('--model-size', type=str, default='large-v3', 
                           choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                           help='Whisper model size for lecture transcription')
    
    # Full pipeline parameters
    run_parser.add_argument('--screenshot-limit', type=int, help='Limit for screenshots (full pipeline only)')
    run_parser.add_argument('--video-limit', type=int, help='Limit for videos (full pipeline only)')
    run_parser.add_argument('--lecture-limit', type=int, help='Limit for lectures (full pipeline only)')
    run_parser.add_argument('--no-screenshots', action='store_true', help='Skip screenshots in full pipeline')
    run_parser.add_argument('--no-videos', action='store_true', help='Skip videos in full pipeline')
    run_parser.add_argument('--no-lectures', action='store_true', help='Skip lectures in full pipeline')
    
    # =============================================================================
    # QUERY command
    # =============================================================================
    query_parser = subparsers.add_parser('query', help='Query processed media with natural language')
    query_parser.add_argument('query', type=str, help='Natural language query (e.g., "screenshots with Python code")')
    query_parser.add_argument('--type', type=str, default='all', 
                             choices=['all', 'screenshots', 'videos', 'lectures'],
                             help='Media type to search')
    query_parser.add_argument('--limit', type=int, default=10, help='Maximum number of results')
    
    # =============================================================================
    # STATUS command
    # =============================================================================
    status_parser = subparsers.add_parser('status', help='Show pipeline status and configuration')
    status_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed status information')
    
    # =============================================================================
    # EXPORT command
    # =============================================================================
    export_parser = subparsers.add_parser('export', help='Export analysis results')
    export_parser.add_argument('--format', type=str, default='json', 
                              choices=['json', 'csv'],
                              help='Export format')
    export_parser.add_argument('--output', type=str, help='Output file path')
    export_parser.add_argument('--no-screenshots', action='store_true', help='Exclude screenshots from export')
    export_parser.add_argument('--no-videos', action='store_true', help='Exclude videos from export')
    export_parser.add_argument('--no-lectures', action='store_true', help='Exclude lectures from export')
    
    # =============================================================================
    # CLEANUP command
    # =============================================================================
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data and temporary files')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Remove data older than N days')
    cleanup_parser.add_argument('--airtable', action='store_true', help='Also clean up Airtable records')
    cleanup_parser.add_argument('--cache', action='store_true', help='Clean up local cache')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without actually doing it')
    
    return parser


def run_command(args):
    """Execute the run command"""
    try:
        print("🚀 Starting d6tflow Media AI Pipeline...")
        print(f"📅 Started at: {datetime.now().isoformat()}")
        
        if args.full:
            print("🔄 Running FULL pipeline (screenshots + videos + lectures)")
            
            params = {}
            if args.screenshot_limit is not None:
                params['screenshot_limit'] = args.screenshot_limit
            if args.video_limit is not None:
                params['video_limit'] = args.video_limit
            if args.lecture_limit is not None:
                params['lecture_limit'] = args.lecture_limit
            if args.recent_hours is not None:
                params['recent_hours'] = args.recent_hours
            if args.no_airtable:
                params['enable_airtable'] = False
            if args.no_screenshots:
                params['include_screenshots'] = False
            if args.no_videos:
                params['include_videos'] = False
            if args.no_lectures:
                params['include_lectures'] = False
            if args.keyframes != 5:
                params['num_keyframes'] = args.keyframes
            if args.advanced_diarization:
                params['advanced_diarization'] = True
            
            result = run_full_pipeline(**params)
            
        elif args.screenshots:
            print("📸 Running SCREENSHOT analysis pipeline")
            
            params = {}
            if args.limit is not None:
                params['limit'] = args.limit
            if args.recent_hours is not None:
                params['recent_hours'] = args.recent_hours
            if args.no_airtable:
                params['enable_airtable'] = False
            
            result = run_screenshot_pipeline(**params)
            
        elif args.videos:
            print("🎥 Running VIDEO analysis pipeline")
            
            params = {}
            if args.limit is not None:
                params['limit'] = args.limit
            if args.recent_hours is not None:
                params['recent_hours'] = args.recent_hours
            if args.no_airtable:
                params['enable_airtable'] = False
            if args.keyframes != 5:
                params['num_keyframes'] = args.keyframes
            
            result = run_video_pipeline(**params)
            
        elif args.lectures:
            print("🎓 Running LECTURE processing pipeline")
            
            params = {}
            if args.limit is not None:
                params['limit'] = args.limit
            if args.lecture_dir is not None:
                params['source_directory'] = args.lecture_dir
            if args.advanced_diarization:
                params['advanced_diarization'] = True
            if args.model_size != 'large-v3':
                params['model_size'] = args.model_size
            
            result = run_lecture_pipeline(**params)
        
        # Display results summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED!")
        print("="*60)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return 1
        
        # Show results based on pipeline type
        pipeline_type = result.get('pipeline_type', 'unknown')
        
        if pipeline_type == 'full_media_pipeline':
            summary = result.get('summary', {})
            print(f"📊 Total files processed: {summary.get('total_files_processed', 0)}")
            print(f"✅ Successful: {summary.get('total_successful', 0)}")
            print(f"📈 Success rate: {summary.get('overall_success_rate', 0):.1f}%")
            print(f"💰 Tokens used: {summary.get('total_tokens_used', 0):,}")
            print(f"🔧 Components: {', '.join(summary.get('components_executed', []))}")
            
            # Component-specific details
            results = result.get('results', {})
            if 'screenshots' in results:
                s = results['screenshots']
                print(f"  📸 Screenshots: {s.get('successful', 0)}/{s.get('total_analyzed', 0)} successful")
            if 'videos' in results:
                v = results['videos']
                print(f"  🎥 Videos: {v.get('successful', 0)}/{v.get('total_analyzed', 0)} successful")
            if 'lectures' in results:
                l = results['lectures']
                print(f"  🎓 Lectures: {l.get('successful', 0)}/{l.get('total_processed', 0)} successful")
        
        else:
            # Single pipeline results
            if 'results' in result:
                pipeline_results = result['results']
                if 'successful_analyses' in pipeline_results:
                    print(f"✅ Successful analyses: {pipeline_results['successful_analyses']}")
                if 'successful_processing' in pipeline_results:
                    print(f"✅ Successful processing: {pipeline_results['successful_processing']}")
                if 'success_rate' in pipeline_results:
                    print(f"📈 Success rate: {pipeline_results['success_rate']:.1f}%")
        
        print(f"📅 Completed at: {datetime.now().isoformat()}")
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1


def query_command(args):
    """Execute the query command"""
    try:
        print(f"🔍 Searching for: '{args.query}' in {args.type}")
        
        result = query_pipeline_results(args.query, args.type, args.limit)
        
        if 'error' in result:
            print(f"❌ Query failed: {result['error']}")
            return 1
        
        results_count = result.get('results_count', 0)
        print(f"📊 Found {results_count} results")
        
        if results_count > 0:
            print("\nResults:")
            print("-" * 40)
            
            results = result.get('results', [])
            for i, item in enumerate(results[:10], 1):  # Show first 10
                if isinstance(item, dict):
                    # Handle different result formats
                    filename = item.get('filename', item.get('file_name', 'Unknown'))
                    content = item.get('text', item.get('summary', item.get('content', 'No content')))
                    score = item.get('score', item.get('confidence', 'N/A'))
                    
                    print(f"{i}. {filename}")
                    if content:
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"   {preview}")
                    if score != 'N/A':
                        print(f"   Score: {score}")
                    print()
        else:
            print("No results found. Try a different query or check if media has been processed.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return 1


def status_command(args):
    """Execute the status command"""
    try:
        print("📊 d6tflow Media AI Pipeline Status")
        print("=" * 40)
        
        status = get_pipeline_status()
        
        if 'error' in status:
            print(f"❌ Error getting status: {status['error']}")
            return 1
        
        # Show configuration
        print("🔧 Configuration:")
        print(f"  PNG Directory: {PipelineConfig.PNG_SOURCE_DIR}")
        print(f"  MP4 Directory: {PipelineConfig.MP4_SOURCE_DIR}")
        print(f"  Data Directory: {PipelineConfig.DATA_DIR}")
        print(f"  Output Directory: {PipelineConfig.OUTPUT_DIR}")
        print(f"  Airtable Enabled: {PipelineConfig.ENABLE_AIRTABLE}")
        print(f"  OpenAI API Key: {'✅ Set' if PipelineConfig.OPENAI_API_KEY else '❌ Not set'}")
        print()
        
        # Show default limits
        print("📋 Default Processing Limits:")
        print(f"  Screenshots: {PipelineConfig.DEFAULT_SCREENSHOT_LIMIT}")
        print(f"  Videos: {PipelineConfig.DEFAULT_VIDEO_LIMIT}")
        print(f"  Lectures: {PipelineConfig.DEFAULT_LECTURE_LIMIT}")
        print()
        
        # Show directory status
        print("📁 Directory Status:")
        for key, path in status.get('source_dirs', {}).items():
            if Path(path).exists():
                file_count = len(list(Path(path).iterdir())) if Path(path).is_dir() else 0
                print(f"  {key.upper()}: ✅ {path} ({file_count} items)")
            else:
                print(f"  {key.upper()}: ❌ {path} (not found)")
        print()
        
        if args.verbose:
            print("🔍 Detailed Status:")
            print(json.dumps(status, indent=2, default=str))
        
        return 0
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return 1


def export_command(args):
    """Execute the export command"""
    try:
        print(f"📤 Exporting results in {args.format} format...")
        
        params = {
            'export_format': args.format,
            'include_screenshots': not args.no_screenshots,
            'include_videos': not args.no_videos,
            'include_lectures': not args.no_lectures
        }
        
        if args.output:
            params['output_path'] = args.output
        
        result = export_pipeline_results(**params)
        
        if 'error' in result.get('export_info', {}):
            print(f"❌ Export failed: {result['export_info']['error']}")
            return 1
        
        summary = result.get('summary', {})
        print(f"✅ Export complete!")
        print(f"📊 Files exported: {summary.get('total_successful', 0)}/{summary.get('total_files_processed', 0)}")
        print(f"📈 Success rate: {summary.get('success_rate', 0):.1f}%")
        
        if args.output:
            print(f"💾 Saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


def cleanup_command(args):
    """Execute the cleanup command"""
    try:
        if args.dry_run:
            print(f"🔍 DRY RUN: Showing what would be cleaned (older than {args.days} days)")
        else:
            print(f"🧹 Cleaning up data older than {args.days} days...")
        
        if not args.dry_run:
            result = cleanup_data(
                days_to_keep=args.days,
                cleanup_airtable=args.airtable
            )
            
            if 'error' in result:
                print(f"❌ Cleanup failed: {result['error']}")
                return 1
            
            total_cleaned = result.get('total_items_cleaned', 0)
            print(f"✅ Cleanup complete! Processed {total_cleaned} items")
            
            # Show detailed results
            for component, details in result.get('results', {}).items():
                status = details.get('status', 'unknown')
                if status == 'success':
                    count = details.get('deleted_records', 0)
                    print(f"  {component}: {count} items cleaned")
                elif status == 'error':
                    print(f"  {component}: ❌ {details.get('error', 'Unknown error')}")
                else:
                    print(f"  {component}: {details.get('message', 'No action taken')}")
        else:
            print("This would clean up old pipeline data and temporary files.")
            print("Use --airtable to also clean Airtable records.")
            print("Use --cache to clean local d6tflow cache.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute the appropriate command
    if args.command == 'run':
        return run_command(args)
    elif args.command == 'query':
        return query_command(args)
    elif args.command == 'status':
        return status_command(args)
    elif args.command == 'export':
        return export_command(args)
    elif args.command == 'cleanup':
        return cleanup_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
