#!/usr/bin/env python3
"""
Example: d6tflow Media AI Pipeline Usage
Demonstrates how to use the d6tflow-based media analysis pipeline
"""
import json
from datetime import datetime

# Import the d6tflow pipeline functions
from d6tflow_pipeline import (
    run_screenshot_pipeline,
    run_video_pipeline,
    run_lecture_pipeline,
    run_full_pipeline,
    query_pipeline_results,
    export_pipeline_results,
    get_pipeline_status
)
from d6tflow_storage_tasks import cleanup_data


def example_1_basic_screenshot_analysis():
    """Example 1: Basic screenshot analysis"""
    print("🔍 Example 1: Basic Screenshot Analysis")
    print("=" * 50)
    
    # Run screenshot analysis with a small limit for testing
    results = run_screenshot_pipeline(limit=3)
    
    print(f"✅ Analysis completed!")
    print(f"   Pipeline type: {results.get('pipeline_type')}")
    print(f"   Files analyzed: {results.get('results', {}).get('total_files', 0)}")
    print(f"   Success rate: {results.get('results', {}).get('success_rate', 0):.1f}%")
    print()
    
    return results


def example_2_video_analysis_with_custom_settings():
    """Example 2: Video analysis with custom settings"""
    print("🎥 Example 2: Video Analysis with Custom Settings")
    print("=" * 50)
    
    # Run video analysis with custom keyframes and recent files only
    results = run_video_pipeline(
        limit=2,
        recent_hours=48,  # Only files from last 48 hours
        num_keyframes=7,  # Extract 7 keyframes per video
        enable_airtable=False  # Skip Airtable storage for speed
    )
    
    print(f"✅ Video analysis completed!")
    if 'results' in results:
        r = results['results']
        print(f"   Videos processed: {r.get('total_files', 0)}")
        print(f"   Successful: {r.get('successful_analyses', 0)}")
        print(f"   Total duration: {r.get('total_duration_hours', 0):.1f} hours")
        print(f"   Keyframes analyzed: {r.get('total_keyframes_analyzed', 0)}")
    print()
    
    return results


def example_3_lecture_processing():
    """Example 3: Lecture processing with transcription"""
    print("🎓 Example 3: Lecture Processing")
    print("=" * 50)
    
    # Run lecture processing with a small limit
    results = run_lecture_pipeline(
        limit=1,
        advanced_diarization=False,  # Use simple diarization for speed
        model_size="base"  # Use smaller Whisper model for testing
    )
    
    print(f"✅ Lecture processing completed!")
    if 'results' in results:
        r = results['results']
        print(f"   Lectures processed: {r.get('total_files', 0)}")
        print(f"   Successful: {r.get('successful_processing', 0)}")
        print(f"   Content hours: {r.get('total_duration_hours', 0):.1f}")
        print(f"   Transcript segments: {r.get('total_transcript_segments', 0)}")
        print(f"   Vector chunks stored: {r.get('total_vector_chunks_stored', 0)}")
    print()
    
    return results


def example_4_full_pipeline():
    """Example 4: Full pipeline with all components"""
    print("🚀 Example 4: Full Pipeline")
    print("=" * 50)
    
    # Run the complete pipeline with small limits for testing
    results = run_full_pipeline(
        screenshot_limit=2,
        video_limit=1,
        lecture_limit=1,
        recent_hours=72,  # Last 3 days
        enable_airtable=False,  # Skip Airtable for this example
        num_keyframes=5,
        advanced_diarization=False
    )
    
    print(f"✅ Full pipeline completed!")
    summary = results.get('summary', {})
    print(f"   Total files processed: {summary.get('total_files_processed', 0)}")
    print(f"   Total successful: {summary.get('total_successful', 0)}")
    print(f"   Overall success rate: {summary.get('overall_success_rate', 0):.1f}%")
    print(f"   Total tokens used: {summary.get('total_tokens_used', 0):,}")
    print(f"   Components executed: {', '.join(summary.get('components_executed', []))}")
    
    # Show component-specific results
    pipeline_results = results.get('results', {})
    for component, data in pipeline_results.items():
        if component in ['screenshots', 'videos', 'lectures']:
            successful = data.get('successful', data.get('successful_processing', 0))
            total = data.get('total_analyzed', data.get('total_processed', 0))
            print(f"   {component.title()}: {successful}/{total} successful")
    print()
    
    return results


def example_5_querying_results():
    """Example 5: Querying processed media"""
    print("🔍 Example 5: Querying Processed Media")
    print("=" * 50)
    
    # Try different types of queries
    queries = [
        "screenshots with Python code",
        "error messages",
        "terminal commands",
        "machine learning"
    ]
    
    for query in queries:
        try:
            print(f"Searching for: '{query}'")
            results = query_pipeline_results(query, limit=3)
            
            count = results.get('results_count', 0)
            print(f"  Found {count} results")
            
            if count > 0:
                for i, result in enumerate(results.get('results', [])[:2], 1):
                    filename = result.get('filename', result.get('file_name', 'Unknown'))
                    print(f"    {i}. {filename}")
            
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
        
        print()


def example_6_export_and_cleanup():
    """Example 6: Export results and cleanup"""
    print("📤 Example 6: Export and Cleanup")
    print("=" * 50)
    
    # Export results to a JSON file
    try:
        export_path = f"media_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = export_pipeline_results(
            export_format='json',
            output_path=export_path,
            include_screenshots=True,
            include_videos=True,
            include_lectures=True
        )
        
        summary = results.get('summary', {})
        print(f"✅ Export completed!")
        print(f"   Exported {summary.get('total_successful', 0)} successful analyses")
        print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"   Saved to: {export_path}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    print()
    
    # Demonstrate cleanup (dry run)
    try:
        print("🧹 Cleanup preview (dry run):")
        print("   This would clean up data older than 30 days")
        print("   Use cleanup_data(days_to_keep=30, cleanup_airtable=False) to actually clean")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
    
    print()


def example_7_status_check():
    """Example 7: Check system status"""
    print("📊 Example 7: System Status Check")
    print("=" * 50)
    
    try:
        status = get_pipeline_status()
        
        if 'error' in status:
            print(f"❌ Status check failed: {status['error']}")
        else:
            print("✅ System status retrieved successfully!")
            
            # Show key status information
            source_dirs = status.get('source_dirs', {})
            for media_type, path in source_dirs.items():
                print(f"   {media_type.upper()} directory: {path}")
            
            airtable_enabled = status.get('airtable_enabled', False)
            print(f"   Airtable enabled: {'Yes' if airtable_enabled else 'No'}")
            
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    print()


def main():
    """Run all examples"""
    print("🤖 d6tflow Media AI Pipeline Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use the d6tflow-based")
    print("media analysis pipeline. Each example shows different features.")
    print("=" * 60)
    print()
    
    # Check system status first
    example_7_status_check()
    
    # Run examples in order
    try:
        # Start with simple screenshot analysis
        example_1_basic_screenshot_analysis()
        
        # Video analysis with custom settings
        example_2_video_analysis_with_custom_settings()
        
        # Lecture processing (if video files are available)
        example_3_lecture_processing()
        
        # Full pipeline demonstration
        example_4_full_pipeline()
        
        # Query the processed media
        example_5_querying_results()
        
        # Export and cleanup
        example_6_export_and_cleanup()
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("This might be due to missing media files or configuration issues.")
        print("Check the D6TFLOW_GUIDE.md for troubleshooting steps.")
    
    print("🎉 Examples completed!")
    print()
    print("Next steps:")
    print("1. Try running individual examples with your own media files")
    print("2. Adjust limits and parameters in d6tflow_config.py")
    print("3. Use the CLI: python d6tflow_cli.py --help")
    print("4. Read the full guide: D6TFLOW_GUIDE.md")


if __name__ == '__main__':
    main()
