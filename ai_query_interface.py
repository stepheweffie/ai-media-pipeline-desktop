"""
AI-Powered Query Interface for Media AI Pipeline
Allows natural language questions about screenshots and recordings
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import openai
from screenshot_analyzer import ScreenshotAnalyzer
from video_analyzer import VideoAnalyzer
from airtable_media_manager import AirtableMediaManager
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIQueryInterface:
    def __init__(self):
        """Initialize the AI query interface"""
        config.validate_config()
        openai.api_key = config.OPENAI_API_KEY
        
        self.screenshot_analyzer = ScreenshotAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.airtable_manager = AirtableMediaManager()
        
        # Query types and intent detection
        self.query_patterns = {
            'search': ['find', 'search', 'show me', 'look for', 'where is'],
            'recent': ['recent', 'latest', 'newest', 'last', 'yesterday'],
            'content_type': ['code', 'error', 'browser', 'terminal', 'email', 'design'],
            'stats': ['how many', 'count', 'statistics', 'stats', 'total'],
            'when': ['when', 'date', 'time', 'created', 'modified'],
            'what': ['what is', 'describe', 'explain', 'tell me about'],
            'help': ['help', 'how to', 'what can', 'commands']
        }
    
    def detect_query_intent(self, query: str) -> Tuple[str, Dict]:
        """Detect the intent and extract parameters from a natural language query"""
        query_lower = query.lower()
        intent = 'general'
        params = {}
        
        # Extract time references
        time_patterns = {
            'today': r'\btoday\b',
            'yesterday': r'\byesterday\b',
            'this week': r'\bthis week\b',
            'last week': r'\blast week\b',
            'recent': r'\brecent\b|\blast\b'
        }
        
        for time_ref, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                params['time_filter'] = time_ref
                break
        
        # Extract content type references
        for content_type in config.CONTENT_CATEGORIES:
            if content_type in query_lower:
                params['content_type'] = content_type
                break
        
        # Extract media type
        if 'screenshot' in query_lower or 'image' in query_lower or 'png' in query_lower:
            params['media_type'] = 'screenshots'
        elif 'recording' in query_lower or 'video' in query_lower or 'mp4' in query_lower:
            params['media_type'] = 'recordings'
        else:
            params['media_type'] = 'all'
        
        # Detect intent based on patterns
        for intent_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                intent = intent_type
                break
        
        # Extract search terms (remove common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'show', 'me', 'find', 'search', 'look', 'where', 'is', 'are', 'was', 'were'}
        search_terms = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
        params['search_terms'] = search_terms
        
        logger.info(f"Detected intent: {intent}, params: {params}")
        return intent, params
    
    def process_query(self, query: str) -> Dict:
        """Process a natural language query and return results"""
        try:
            intent, params = self.detect_query_intent(query)
            
            # Route to appropriate handler
            if intent == 'search':
                return self.handle_search_query(query, params)
            elif intent == 'recent':
                return self.handle_recent_query(params)
            elif intent == 'stats':
                return self.handle_stats_query(params)
            elif intent == 'what':
                return self.handle_description_query(query, params)
            elif intent == 'help':
                return self.handle_help_query()
            else:
                return self.handle_general_query(query, params)
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'message': "Sorry, I encountered an error processing your query."
            }
    
    def handle_search_query(self, query: str, params: Dict) -> Dict:
        """Handle search queries"""
        try:
            search_term = ' '.join(params.get('search_terms', []))
            if not search_term:
                # Extract search term from query more intelligently
                search_term = self.extract_search_term_with_ai(query)
            
            media_type = params.get('media_type', 'all')
            content_type = params.get('content_type')
            
            # Search in Airtable if available
            if self.airtable_manager.enabled:
                if content_type:
                    results = self.airtable_manager.get_media_by_content_type(content_type)
                else:
                    results = self.airtable_manager.search_media(search_term, media_type)
            else:
                # Local search
                results = self.local_search(search_term, media_type, content_type)
            
            return {
                'success': True,
                'intent': 'search',
                'query': query,
                'search_term': search_term,
                'results_count': len(results),
                'results': results[:10],  # Limit to top 10
                'message': f"Found {len(results)} results for '{search_term}'"
            }
            
        except Exception as e:
            logger.error(f"Error in search query: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_recent_query(self, params: Dict) -> Dict:
        """Handle queries for recent media"""
        try:
            media_type = params.get('media_type', 'all')
            time_filter = params.get('time_filter', 'recent')
            
            # Get recent files based on type
            if media_type == 'screenshots' or media_type == 'all':
                png_directory = config.PNG_PATH
                recent_screenshots = self.screenshot_analyzer.analyze_screenshots_in_directory(png_directory, limit=5)
            else:
                recent_screenshots = []
            
            if media_type == 'recordings' or media_type == 'all':
                mp4_directory = config.MP4_PATH
                recent_recordings = self.video_analyzer.analyze_videos_in_directory(mp4_directory, limit=3)
            else:
                recent_recordings = []
            
            total_results = len(recent_screenshots) + len(recent_recordings)
            
            return {
                'success': True,
                'intent': 'recent',
                'time_filter': time_filter,
                'media_type': media_type,
                'screenshots': recent_screenshots,
                'recordings': recent_recordings,
                'total_count': total_results,
                'message': f"Found {total_results} recent {media_type} files"
            }
            
        except Exception as e:
            logger.error(f"Error in recent query: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_stats_query(self, params: Dict) -> Dict:
        """Handle statistics queries"""
        try:
            if self.airtable_manager.enabled:
                stats = self.airtable_manager.get_media_stats()
            else:
                # Local stats calculation
                stats = self.calculate_local_stats()
            
            # Generate human-readable summary
            summary_parts = []
            if 'screenshots' in stats:
                screenshot_stats = stats['screenshots']
                summary_parts.append(f"{screenshot_stats.get('total', 0)} screenshots")
                if screenshot_stats.get('with_code', 0) > 0:
                    summary_parts.append(f"{screenshot_stats['with_code']} with code")
            
            if 'recordings' in stats:
                recording_stats = stats['recordings']
                summary_parts.append(f"{recording_stats.get('total', 0)} recordings")
                duration_min = recording_stats.get('total_duration_minutes', 0)
                if duration_min > 0:
                    summary_parts.append(f"{duration_min:.1f} minutes total")
            
            summary = ', '.join(summary_parts)
            
            return {
                'success': True,
                'intent': 'stats',
                'stats': stats,
                'message': f"Media statistics: {summary}"
            }
            
        except Exception as e:
            logger.error(f"Error in stats query: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_description_query(self, query: str, params: Dict) -> Dict:
        """Handle 'what is' or description queries"""
        try:
            # Try to identify specific file mentioned in query
            filename = self.extract_filename_from_query(query)
            
            if filename:
                # Look up specific file analysis
                file_analysis = self.get_file_analysis(filename)
                if file_analysis:
                    analysis = file_analysis.get('analysis', {})
                    return {
                        'success': True,
                        'intent': 'description',
                        'filename': filename,
                        'file_analysis': file_analysis,
                        'message': f"Analysis of {filename}: {analysis.get('summary', 'No summary available')}"
                    }
            
            # General description based on search terms
            search_term = ' '.join(params.get('search_terms', []))
            search_results = self.handle_search_query(f"find {search_term}", params)
            
            if search_results.get('success') and search_results.get('results_count', 0) > 0:
                top_result = search_results['results'][0]
                return {
                    'success': True,
                    'intent': 'description',
                    'search_term': search_term,
                    'top_result': top_result,
                    'message': f"Most relevant result for '{search_term}'"
                }
            
            return {
                'success': False,
                'message': f"Could not find information about '{search_term}'"
            }
            
        except Exception as e:
            logger.error(f"Error in description query: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_help_query(self) -> Dict:
        """Handle help queries"""
        help_text = """AI Media Query Interface - Available Commands:

📸 SEARCH QUERIES:
• "Find screenshots with code" - Search for coding screenshots
• "Show me browser recordings" - Find browser-related videos  
• "Look for error messages" - Find error screenshots/recordings
• "Find recent terminal sessions" - Search recent terminal content

⏰ TIME QUERIES:
• "Show me today's screenshots" - Recent captures
• "What did I record yesterday?" - Yesterday's recordings
• "Latest coding sessions" - Most recent code-related content

📊 STATISTICS:
• "How many screenshots do I have?" - Get counts
• "Show me stats" - Overall statistics
• "Count of error screenshots" - Specific type counts

🔍 SPECIFIC QUERIES:
• "What is in Screen Shot 2025-08-25.png?" - Describe specific file
• "Explain this recording" - Get details about content
• "When was this created?" - File timestamps

💡 EXAMPLES:
• "Find screenshots from VSCode with Python code"
• "Show me all design mockups from this week"
• "What terminal commands did I run recently?"
• "Find recordings with error messages"

Type any natural language question about your screenshots and recordings!"""
        
        return {
            'success': True,
            'intent': 'help',
            'message': help_text
        }
    
    def handle_general_query(self, query: str, params: Dict) -> Dict:
        """Handle general queries using AI interpretation"""
        try:
            # Use GPT to interpret the query and suggest actions
            interpretation = self.interpret_query_with_ai(query)
            
            # Try to execute the interpreted query
            if interpretation.get('suggested_action'):
                action = interpretation['suggested_action']
                if action == 'search':
                    return self.handle_search_query(query, params)
                elif action == 'recent':
                    return self.handle_recent_query(params)
                elif action == 'stats':
                    return self.handle_stats_query(params)
            
            return {
                'success': True,
                'intent': 'general',
                'interpretation': interpretation,
                'message': interpretation.get('response', "I understand your query but need more specific information.")
            }
            
        except Exception as e:
            logger.error(f"Error in general query: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_search_term_with_ai(self, query: str) -> str:
        """Use AI to extract the main search term from a natural language query"""
        try:
            prompt = f"""Extract the main search term or concept from this query about screenshots/recordings:

Query: "{query}"

Return just the key search term or phrase that would be most useful for finding relevant media files. Focus on:
- Technical terms, application names, error types
- Content descriptions, workflow steps  
- Programming languages, commands, URLs
- Visual elements or UI components

Search term:"""
            
            response = openai.chat.completions.create(
                model=config.TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            
            search_term = response.choices[0].message.content.strip()
            return search_term
            
        except Exception as e:
            logger.warning(f"AI search term extraction failed: {e}")
            # Fallback to simple extraction
            words = query.lower().split()
            stop_words = {'find', 'search', 'show', 'me', 'look', 'for', 'where', 'is', 'the', 'a', 'an'}
            return ' '.join([w for w in words if w not in stop_words])
    
    def interpret_query_with_ai(self, query: str) -> Dict:
        """Use AI to interpret complex queries"""
        try:
            prompt = f"""You are an AI assistant for a media analysis system that manages screenshots and screen recordings. 

The user asked: "{query}"

Available actions:
- search: Find specific content in media files
- recent: Show recent screenshots/recordings  
- stats: Show statistics about media collection
- describe: Explain what's in a specific file

Respond with JSON:
{{
    "suggested_action": "search|recent|stats|describe|none",
    "confidence": 0.8,
    "response": "Brief helpful response to the user",
    "search_terms": ["relevant", "terms"],
    "reasoning": "Why you chose this action"
}}"""
            
            response = openai.chat.completions.create(
                model=config.TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            # Parse JSON response
            try:
                interpretation = json.loads(response.choices[0].message.content)
                return interpretation
            except json.JSONDecodeError:
                return {
                    "suggested_action": "none",
                    "confidence": 0.5,
                    "response": "I can help you search your screenshots and recordings. Try asking 'find screenshots with code' or 'show me recent recordings'.",
                    "reasoning": "Could not parse AI response"
                }
            
        except Exception as e:
            logger.warning(f"AI query interpretation failed: {e}")
            return {
                "suggested_action": "none",
                "response": "I understand you want to search your media files. Please try a more specific query."
            }
    
    def local_search(self, search_term: str, media_type: str, content_type: Optional[str]) -> List[Dict]:
        """Perform local search when Airtable is not available"""
        results = []
        
        try:
            if media_type in ['all', 'screenshots']:
                png_directory = config.PNG_PATH
                screenshot_results = self.screenshot_analyzer.search_screenshots_by_content(
                    png_directory, search_term, limit=10
                )
                for result in screenshot_results:
                    result['media_type'] = 'screenshot'
                    results.append(result)
            
            # Note: Video search would require pre-analysis or on-demand analysis
            # For now, we'll skip local video search for performance
            
        except Exception as e:
            logger.error(f"Local search error: {e}")
        
        return results
    
    def calculate_local_stats(self) -> Dict:
        """Calculate basic statistics from local files"""
        import os
        
        stats = {
            'screenshots': {'total': 0},
            'recordings': {'total': 0}
        }
        
        try:
            # Count screenshots
            png_directory = config.PNG_PATH
            if os.path.exists(png_directory):
                png_files = [f for f in os.listdir(png_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                stats['screenshots']['total'] = len(png_files)
            
            # Count recordings
            mp4_directory = config.MP4_PATH
            if os.path.exists(mp4_directory):
                mp4_files = [f for f in os.listdir(mp4_directory) if f.lower().endswith(('.mp4', '.mov'))]
                stats['recordings']['total'] = len(mp4_files)
        
        except Exception as e:
            logger.error(f"Error calculating local stats: {e}")
        
        return stats
    
    def extract_filename_from_query(self, query: str) -> Optional[str]:
        """Extract filename from query if mentioned"""
        # Look for common screenshot/recording filename patterns
        import re
        
        patterns = [
            r'Screen Shot \d{4}-\d{2}-\d{2} at \d{1,2}\.\d{2}\.\d{2} [AP]M\.png',
            r'Screen Recording \d{4}-\d{2}-\d{2} at \d{1,2}\.\d{2}\.\d{2} [AP]M\.mp4',
            r'[\w\s-]+\.(png|jpg|jpeg|mp4|mov)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group()
        
        return None
    
    def get_file_analysis(self, filename: str) -> Optional[Dict]:
        """Get analysis for a specific file"""
        # This would query the Airtable or analyze the file directly
        # For now, return None as placeholder
        return None

def main():
    """Example usage"""
    interface = AIQueryInterface()
    
    # Example queries
    test_queries = [
        "Find screenshots with code",
        "Show me recent recordings",
        "How many screenshots do I have?",
        "What terminal commands did I run today?",
        "Find error messages in my screenshots"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = interface.process_query(query)
        print(f"📋 Response: {result.get('message', 'No message')}")
        if result.get('success'):
            print(f"✅ Results: {result.get('results_count', 0)} items found")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
