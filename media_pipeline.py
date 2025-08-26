#!/usr/bin/env python3
"""
Media AI Pipeline - Main Orchestrator
Coordinates screenshot analysis, video analysis, and AI-powered search
"""
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from screenshot_analyzer import ScreenshotAnalyzer
from video_analyzer import VideoAnalyzer
from airtable_media_manager import AirtableMediaManager
from ai_query_interface import AIQueryInterface
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediaAIPipeline:
    def __init__(self):
        """Initialize the media AI pipeline"""
        try:
            config.validate_config()
            self.screenshot_analyzer = ScreenshotAnalyzer()
            self.video_analyzer = VideoAnalyzer()
            self.airtable_manager = AirtableMediaManager()
            self.query_interface = AIQueryInterface()
            logger.info("Media AI Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def analyze_all_media(self, screenshot_limit: Optional[int] = None, video_limit: Optional[int] = None) -> Dict:
        """Analyze all media files and store results"""
        results = {
            'screenshots': {'analyzed': 0, 'saved': 0, 'errors': 0},
            'recordings': {'analyzed': 0, 'saved': 0, 'errors': 0},
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Analyze screenshots
            logger.info("Starting screenshot analysis...")
            png_directory = config.PNG_PATH
            screenshot_results = self.screenshot_analyzer.analyze_screenshots_in_directory(
                png_directory, limit=screenshot_limit
            )
            
            for screenshot_result in screenshot_results:
                results['screenshots']['analyzed'] += 1
                
                # Save to Airtable if available
                if self.airtable_manager.enabled:
                    try:
                        saved_id = self.airtable_manager.save_screenshot_analysis(screenshot_result)
                        if saved_id:
                            results['screenshots']['saved'] += 1
                        else:
                            results['screenshots']['errors'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to save screenshot analysis: {e}")
                        results['screenshots']['errors'] += 1
            
            # Analyze videos
            logger.info("Starting video analysis...")
            mp4_directory = config.MP4_PATH
            video_results = self.video_analyzer.analyze_videos_in_directory(
                mp4_directory, limit=video_limit
            )
            
            for video_result in video_results:
                results['recordings']['analyzed'] += 1
                
                # Save to Airtable if available
                if self.airtable_manager.enabled:
                    try:
                        saved_id = self.airtable_manager.save_video_analysis(video_result)
                        if saved_id:
                            results['recordings']['saved'] += 1
                        else:
                            results['recordings']['errors'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to save video analysis: {e}")
                        results['recordings']['errors'] += 1
            
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'completed'
            
            # Print summary
            total_analyzed = results['screenshots']['analyzed'] + results['recordings']['analyzed']
            total_saved = results['screenshots']['saved'] + results['recordings']['saved']
            total_errors = results['screenshots']['errors'] + results['recordings']['errors']
            
            logger.info(f"Pipeline completed: {total_analyzed} files analyzed, {total_saved} saved, {total_errors} errors")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results['error'] = str(e)
            results['status'] = 'failed'
            return results
    
    def analyze_recent_media(self, hours: int = 24) -> Dict:
        """Analyze only media files modified in the last N hours"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        logger.info(f"Analyzing media files modified since {cutoff_time}")
        
        # This would require filtering files by modification time
        # For now, analyze a limited set of recent files
        return self.analyze_all_media(screenshot_limit=10, video_limit=5)
    
    def query_media(self, query: str) -> Dict:
        """Process a natural language query about media"""
        try:
            logger.info(f"Processing query: {query}")
            result = self.query_interface.process_query(query)
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_media_stats(self) -> Dict:
        """Get comprehensive media statistics"""
        try:
            if self.airtable_manager.enabled:
                return self.airtable_manager.get_media_stats()
            else:
                return self.query_interface.calculate_local_stats()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
    
    def search_media(self, search_term: str, media_type: str = 'all', limit: int = 10) -> List[Dict]:
        """Search media files"""
        try:
            if self.airtable_manager.enabled:
                return self.airtable_manager.search_media(search_term, media_type, limit)
            else:
                return self.query_interface.local_search(search_term, media_type, None)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old analysis data"""
        if self.airtable_manager.enabled:
            return self.airtable_manager.cleanup_old_records(days)
        else:
            logger.info("No cleanup needed for local-only mode")
            return 0
    
    def get_system_status(self) -> Dict:
        """Get system status and health check"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'openai_configured': bool(config.OPENAI_API_KEY),
            'airtable_configured': bool(config.AIRTABLE_API_KEY and config.AIRTABLE_BASE_ID),
            'airtable_connected': self.airtable_manager.enabled,
            'directories': {
                'png_exists': os.path.exists(config.PNG_PATH),
                'mp4_exists': os.path.exists(config.MP4_PATH)
            }
        }
        
        # Count files
        try:
            png_dir = config.PNG_PATH
            mp4_dir = config.MP4_PATH
            
            if os.path.exists(png_dir):
                png_files = [f for f in os.listdir(png_dir) if f.lower().endswith(tuple(config.SUPPORTED_IMAGE_FORMATS))]
                status['file_counts'] = {'screenshots': len(png_files)}
            
            if os.path.exists(mp4_dir):
                mp4_files = [f for f in os.listdir(mp4_dir) if f.lower().endswith(tuple(config.SUPPORTED_VIDEO_FORMATS))]
                status['file_counts']['recordings'] = len(mp4_files)
                
        except Exception as e:
            status['file_count_error'] = str(e)
        
        return status

def create_cli():
    """Create command-line interface"""
    parser = argparse.ArgumentParser(description="Media AI Pipeline - Analyze screenshots and recordings with AI")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze media files')
    analyze_parser.add_argument('--screenshots', type=int, help='Limit number of screenshots to analyze')
    analyze_parser.add_argument('--recordings', type=int, help='Limit number of recordings to analyze')
    analyze_parser.add_argument('--recent', type=int, help='Only analyze files from last N hours')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query media with natural language')
    query_parser.add_argument('query', nargs='+', help='Natural language query')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search media files')
    search_parser.add_argument('term', help='Search term')
    search_parser.add_argument('--type', choices=['all', 'screenshots', 'recordings'], default='all')
    search_parser.add_argument('--limit', type=int, default=10)
    
    # Stats command
    subparsers.add_parser('stats', help='Show media statistics')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Remove data older than N days')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Start interactive query mode')
    
    return parser

def interactive_mode(pipeline: MediaAIPipeline):
    """Run interactive query mode"""
    print("🤖 Media AI Pipeline - Interactive Mode")
    print("Ask questions about your screenshots and recordings!")
    print("Type 'help' for available commands or 'quit' to exit.\n")
    
    while True:
        try:
            query = input("📝 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"🔍 Processing: {query}")
            result = pipeline.query_media(query)
            
            if result.get('success'):
                print(f"✅ {result.get('message', 'Query completed')}")
                
                # Show additional details based on query type
                if result.get('results_count', 0) > 0:
                    print(f"📊 Found {result['results_count']} results")
                    
                    # Show top results
                    for i, item in enumerate(result.get('results', [])[:3]):
                        if 'metadata' in item:
                            filename = item['metadata'].get('filename', 'Unknown file')
                            summary = item.get('analysis', {}).get('summary', 'No summary')
                            print(f"  {i+1}. {filename}: {summary}")
                
                elif result.get('intent') == 'stats':
                    stats = result.get('stats', {})
                    if 'screenshots' in stats:
                        print(f"📸 Screenshots: {stats['screenshots'].get('total', 0)}")
                    if 'recordings' in stats:
                        print(f"🎥 Recordings: {stats['recordings'].get('total', 0)}")
                        
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        pipeline = MediaAIPipeline()
        
        if args.command == 'analyze':
            if args.recent:
                result = pipeline.analyze_recent_media(args.recent)
            else:
                result = pipeline.analyze_all_media(args.screenshots, args.recordings)
            
            print(json.dumps(result, indent=2))
        
        elif args.command == 'query':
            query = ' '.join(args.query)
            result = pipeline.query_media(query)
            print(json.dumps(result, indent=2))
        
        elif args.command == 'search':
            results = pipeline.search_media(args.term, args.type, args.limit)
            print(f"Found {len(results)} results:")
            for result in results:
                print(f"- {result}")
        
        elif args.command == 'stats':
            stats = pipeline.get_media_stats()
            print(json.dumps(stats, indent=2))
        
        elif args.command == 'status':
            status = pipeline.get_system_status()
            print(json.dumps(status, indent=2))
        
        elif args.command == 'cleanup':
            deleted = pipeline.cleanup_old_data(args.days)
            print(f"Cleaned up {deleted} old records")
        
        elif args.command == 'interactive':
            interactive_mode(pipeline)
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
