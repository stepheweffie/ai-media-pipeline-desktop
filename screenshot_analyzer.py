"""
Screenshot Analyzer for Media AI Pipeline
Uses OpenAI Vision API to analyze PNG screenshots
"""
import os
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import openai
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScreenshotAnalyzer:
    def __init__(self):
        """Initialize the screenshot analyzer"""
        config.validate_config()
        openai.api_key = config.OPENAI_API_KEY
        self.vision_model = config.VISION_MODEL
        self.max_tokens = config.MAX_TOKENS
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
            
    def get_file_metadata(self, file_path: str) -> Dict:
        """Extract basic file metadata"""
        try:
            stat = os.stat(file_path)
            return {
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'analyzed_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return {}
    
    def build_analysis_prompt(self) -> str:
        """Build the analysis prompt for the vision model"""
        return """
You are an expert at analyzing desktop screenshots. Please analyze this screenshot and provide a comprehensive breakdown in the following JSON format:

{
    "content_type": "one of: code, documentation, design, email, browser, terminal, ide, meeting, presentation, error, configuration, other",
    "primary_application": "main application visible (e.g., VSCode, Chrome, Terminal, Figma)",
    "summary": "brief 2-3 sentence description of what's shown",
    "extracted_text": "key text content visible in the image",
    "ui_elements": ["list of notable UI elements, buttons, menus visible"],
    "code_detected": true/false,
    "programming_language": "if code is detected, what language",
    "error_detected": true/false,
    "error_details": "if error detected, describe the error",
    "context_tags": ["relevant tags for categorization"],
    "technical_details": "any technical information, URLs, commands, etc.",
    "workflow_step": "what step in a workflow this might represent",
    "searchable_keywords": ["keywords that would be useful for finding this screenshot later"]
}

Focus on extracting actionable information that would help someone:
1. Find this screenshot later based on content
2. Understand the context and workflow
3. Identify any issues or important details
4. Categorize the content appropriately

Be thorough but concise. If you can't determine something clearly, use null or indicate uncertainty.
"""
    
    def analyze_screenshot(self, image_path: str) -> Dict:
        """Analyze a single screenshot using OpenAI Vision API"""
        try:
            logger.info(f"Analyzing screenshot: {os.path.basename(image_path)}")
            
            # Get file metadata
            metadata = self.get_file_metadata(image_path)
            
            # Check file size
            if metadata.get('file_size_mb', 0) > config.MAX_FILE_SIZE_MB:
                logger.warning(f"File {image_path} is too large ({metadata['file_size_mb']}MB)")
                return {
                    'metadata': metadata,
                    'analysis': {
                        'error': f"File too large: {metadata['file_size_mb']}MB > {config.MAX_FILE_SIZE_MB}MB limit"
                    }
                }
            
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Call OpenAI Vision API
            response = openai.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.build_analysis_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response (handle cases where there might be additional text)
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_content = analysis_text[json_start:json_end]
                    analysis_data = json.loads(json_content)
                else:
                    # Fallback if JSON extraction fails
                    analysis_data = {'raw_response': analysis_text}
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from analysis response for {image_path}")
                analysis_data = {'raw_response': analysis_text}
            
            # Combine metadata and analysis
            result = {
                'metadata': metadata,
                'analysis': analysis_data,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
            logger.info(f"Successfully analyzed {os.path.basename(image_path)}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot {image_path}: {e}")
            return {
                'metadata': self.get_file_metadata(image_path) if os.path.exists(image_path) else {},
                'analysis': {'error': str(e)}
            }
    
    def analyze_screenshots_in_directory(self, directory: str, limit: Optional[int] = None) -> List[Dict]:
        """Analyze all screenshots in a directory"""
        try:
            logger.info(f"Analyzing screenshots in directory: {directory}")
            
            if not os.path.exists(directory):
                logger.error(f"Directory not found: {directory}")
                return []
            
            # Find all supported image files
            image_files = []
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS):
                    image_files.append(os.path.join(directory, filename))
            
            # Sort by modification time (newest first)
            image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Apply limit if specified
            if limit:
                image_files = image_files[:limit]
            
            logger.info(f"Found {len(image_files)} image files to analyze")
            
            # Analyze each image
            results = []
            for image_path in image_files:
                try:
                    result = self.analyze_screenshot(image_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze {image_path}: {e}")
                    continue
            
            logger.info(f"Successfully analyzed {len(results)} screenshots")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing directory {directory}: {e}")
            return []
    
    def search_screenshots_by_content(self, directory: str, query: str, limit: Optional[int] = 10) -> List[Dict]:
        """Search through analyzed screenshots based on content query"""
        # This would be enhanced with a vector database for semantic search
        # For now, we'll do basic text matching on analysis results
        try:
            # First analyze recent screenshots if not already done
            results = self.analyze_screenshots_in_directory(directory, limit=50)
            
            # Simple keyword matching for now
            query_lower = query.lower()
            matching_results = []
            
            for result in results:
                analysis = result.get('analysis', {})
                
                # Check various fields for matches
                searchable_fields = [
                    analysis.get('summary', ''),
                    analysis.get('extracted_text', ''),
                    analysis.get('technical_details', ''),
                    ' '.join(analysis.get('searchable_keywords', [])),
                    ' '.join(analysis.get('context_tags', [])),
                    analysis.get('primary_application', ''),
                    analysis.get('content_type', '')
                ]
                
                # Calculate simple relevance score
                relevance_score = 0
                for field in searchable_fields:
                    if isinstance(field, str) and query_lower in field.lower():
                        relevance_score += field.lower().count(query_lower)
                
                if relevance_score > 0:
                    result['relevance_score'] = relevance_score
                    matching_results.append(result)
            
            # Sort by relevance
            matching_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return matching_results[:limit] if limit else matching_results
            
        except Exception as e:
            logger.error(f"Error searching screenshots: {e}")
            return []

def main():
    """Example usage"""
    analyzer = ScreenshotAnalyzer()
    
    # Analyze recent screenshots
    png_directory = os.path.join(config.DESKTOP_PATH, config.PNG_PATH.lstrip('../'))
    results = analyzer.analyze_screenshots_in_directory(png_directory, limit=5)
    
    # Print results
    for result in results:
        print(f"\n--- {result['metadata']['filename']} ---")
        analysis = result['analysis']
        if 'error' not in analysis:
            print(f"Type: {analysis.get('content_type', 'unknown')}")
            print(f"App: {analysis.get('primary_application', 'unknown')}")
            print(f"Summary: {analysis.get('summary', 'no summary')}")
        else:
            print(f"Error: {analysis['error']}")

if __name__ == "__main__":
    main()
