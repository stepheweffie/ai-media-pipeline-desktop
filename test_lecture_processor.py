#!/usr/bin/env python3
"""
Test Suite for Lecture Processor
Tests transcription, diarization, vectorization, and search functionality
"""
import os
import sys
import unittest
import tempfile
import shutil
import json
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lecture_processor import LectureProcessor, TranscriptChunk
import config

class TestLectureProcessor(unittest.TestCase):
    """Test cases for LectureProcessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_vectors")
        
        # Create a test processor instance
        self.processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        # Sample test data
        self.sample_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.sample_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Mock Whisper segments for testing
        self.mock_segments = [
            {
                "start": 0.0,
                "end": 5.5,
                "text": "Hello everyone, welcome to today's lecture on machine learning."
            },
            {
                "start": 6.0,
                "end": 12.3,
                "text": "Today we'll be discussing neural networks and deep learning algorithms."
            },
            {
                "start": 15.0,
                "end": 20.8,
                "text": "Neural networks are inspired by biological neurons in the brain."
            }
        ]
        
        # Mock transcript chunks
        self.sample_chunks = [
            TranscriptChunk(
                text="Hello everyone, welcome to today's lecture on machine learning.",
                start_time=0.0,
                end_time=5.5,
                speaker="Speaker_1",
                confidence=0.95,
                chunk_id="test_video_0000",
                video_file=self.sample_video_path
            ),
            TranscriptChunk(
                text="Today we'll be discussing neural networks and deep learning algorithms.",
                start_time=6.0,
                end_time=12.3,
                speaker="Speaker_1",
                confidence=0.92,
                chunk_id="test_video_0001", 
                video_file=self.sample_video_path
            )
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def create_dummy_video(self, duration_seconds=30):
        """Create a dummy video file for testing"""
        # Create a simple test video using ffmpeg if available
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc2=duration={}:size=640x480:rate=1'.format(duration_seconds),
                '-f', 'lavfi', '-i', 'sine=frequency=440:duration={}'.format(duration_seconds),
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-y', self.sample_video_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg is not available, create a dummy file
            with open(self.sample_video_path, 'wb') as f:
                f.write(b'fake video data for testing')
            return False
    
    def create_dummy_audio(self, duration_seconds=30):
        """Create a dummy audio file for testing"""
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration={}'.format(duration_seconds),
                '-ar', '16000', '-ac', '1', '-y', self.sample_audio_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg is not available, create a dummy file
            with open(self.sample_audio_path, 'wb') as f:
                f.write(b'fake audio data for testing')
            return False
    
    def test_initialization(self):
        """Test LectureProcessor initialization"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.vector_db_path, self.test_db_path)
        self.assertTrue(os.path.exists(self.test_db_path))
        self.assertIsNotNone(self.processor.collection)
    
    @patch('lecture_processor.whisper.load_model')
    def test_whisper_model_loading(self, mock_load_model):
        """Test Whisper model loading"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        self.processor.load_whisper_model()
        
        mock_load_model.assert_called_once_with("large-v3")
        self.assertEqual(self.processor.whisper_model, mock_model)
    
    @patch('lecture_processor.subprocess.run')
    def test_audio_extraction_success(self, mock_subprocess):
        """Test successful audio extraction"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Create dummy video file
        with open(self.sample_video_path, 'w') as f:
            f.write("dummy video")
        
        # Mock the existence of output file
        with patch('os.path.exists', return_value=True):
            result = self.processor.extract_audio(self.sample_video_path, self.sample_audio_path)
            
        self.assertEqual(result, self.sample_audio_path)
        mock_subprocess.assert_called_once()
    
    @patch('lecture_processor.subprocess.run')
    def test_audio_extraction_failure(self, mock_subprocess):
        """Test audio extraction failure"""
        mock_subprocess.return_value = Mock(returncode=1, stderr="ffmpeg error")
        
        with open(self.sample_video_path, 'w') as f:
            f.write("dummy video")
        
        with self.assertRaises(Exception):
            self.processor.extract_audio(self.sample_video_path, self.sample_audio_path)
    
    def test_simple_speaker_assignment(self):
        """Test simple speaker assignment based on pauses"""
        # Test segments with short pauses (same speaker)
        segments_short_pause = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 5.2, "end": 10.0, "text": "Second segment"}  # 0.2s pause
        ]
        
        result = self.processor.assign_speakers_simple(segments_short_pause)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], "Speaker_1")
        self.assertEqual(result[1]["speaker"], "Speaker_1")  # Same speaker
        
        # Test segments with long pause (different speaker)
        segments_long_pause = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 8.0, "end": 12.0, "text": "Second segment"}  # 3s pause
        ]
        
        result = self.processor.assign_speakers_simple(segments_long_pause)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], "Speaker_1")
        self.assertEqual(result[1]["speaker"], "Speaker_2")  # Different speaker
    
    def test_transcript_chunking(self):
        """Test transcript chunking by speaker and time"""
        # Create segments with speaker changes
        segments = []
        for i, base_segment in enumerate(self.mock_segments):
            segment = base_segment.copy()
            segment["speaker"] = "Speaker_1" if i < 2 else "Speaker_2"
            segments.append(segment)
        
        chunks = self.processor.chunk_transcript(segments, self.sample_video_path, chunk_duration=60.0)
        
        self.assertGreater(len(chunks), 0)
        
        # Check chunk properties
        for chunk in chunks:
            self.assertIsInstance(chunk, TranscriptChunk)
            self.assertGreater(len(chunk.text), 0)
            self.assertGreaterEqual(chunk.end_time, chunk.start_time)
            self.assertIn("Speaker", chunk.speaker)
            self.assertEqual(chunk.video_file, self.sample_video_path)
    
    @patch('lecture_processor.openai.embeddings.create')
    def test_embedding_generation(self, mock_embeddings):
        """Test OpenAI embedding generation"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),
            Mock(embedding=[0.6, 0.7, 0.8, 0.9, 1.0])
        ]
        mock_embeddings.return_value = mock_response
        
        texts = ["First text", "Second text"]
        embeddings = self.processor.get_embeddings(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 5)
        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    @patch.object(LectureProcessor, 'get_embeddings')
    def test_vector_storage(self, mock_get_embeddings):
        """Test storing chunks in vector database"""
        # Mock embeddings
        mock_get_embeddings.return_value = [
            [0.1, 0.2, 0.3] * 100,  # Simulate typical embedding size
            [0.4, 0.5, 0.6] * 100
        ]
        
        self.processor.store_chunks_in_vector_db(self.sample_chunks[:2])
        
        # Verify chunks were stored
        stats = self.processor.get_statistics()
        self.assertEqual(stats['total_chunks'], 2)
        self.assertEqual(stats['unique_speakers'], 1)
    
    @patch.object(LectureProcessor, 'get_embeddings')
    def test_semantic_search(self, mock_get_embeddings):
        """Test semantic search functionality"""
        # Mock embeddings for storage
        mock_get_embeddings.return_value = [
            [0.1, 0.2, 0.3] * 100,
            [0.4, 0.5, 0.6] * 100
        ]
        
        # Store test chunks
        self.processor.store_chunks_in_vector_db(self.sample_chunks[:2])
        
        # Test search
        results = self.processor.search_lectures("machine learning", n_results=2)
        
        self.assertIsInstance(results, list)
        if results:  # If search returns results
            self.assertIn('text', results[0])
            self.assertIn('metadata', results[0])
            self.assertIn('distance', results[0])
    
    def test_statistics_generation(self):
        """Test database statistics generation"""
        stats = self.processor.get_statistics()
        
        self.assertIn('total_chunks', stats)
        self.assertIn('unique_speakers', stats)
        self.assertIn('unique_videos', stats) 
        self.assertIn('total_content_hours', stats)
        
        # Initially should be empty
        self.assertEqual(stats['total_chunks'], 0)
    
    @patch.object(LectureProcessor, 'extract_audio')
    @patch.object(LectureProcessor, 'transcribe_with_diarization')
    @patch.object(LectureProcessor, 'chunk_transcript')
    @patch.object(LectureProcessor, 'store_chunks_in_vector_db')
    def test_video_processing_success(self, mock_store, mock_chunk, mock_transcribe, mock_extract):
        """Test successful video processing"""
        # Mock all the processing steps
        mock_extract.return_value = self.sample_audio_path
        mock_transcribe.return_value = self.mock_segments
        mock_chunk.return_value = self.sample_chunks
        mock_store.return_value = None
        
        # Create dummy video file
        with open(self.sample_video_path, 'w') as f:
            f.write("dummy video")
        
        result = self.processor.process_video(self.sample_video_path)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('processing_time', result)
        self.assertIn('total_segments', result)
        self.assertIn('total_chunks', result)
        self.assertIn('speakers', result)
    
    @patch.object(LectureProcessor, 'extract_audio')
    def test_video_processing_failure(self, mock_extract):
        """Test video processing failure handling"""
        mock_extract.side_effect = Exception("Audio extraction failed")
        
        with open(self.sample_video_path, 'w') as f:
            f.write("dummy video")
        
        result = self.processor.process_video(self.sample_video_path)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)
    
    def test_batch_processing_empty_list(self):
        """Test batch processing with empty video list"""
        results = self.processor.process_video_batch([])
        self.assertEqual(len(results), 0)
    
    @patch.object(LectureProcessor, 'process_video')
    def test_batch_processing_multiple_videos(self, mock_process):
        """Test batch processing with multiple videos"""
        # Mock individual video processing
        mock_process.return_value = {'status': 'success', 'video_file': 'test.mp4'}
        
        video_paths = [f"video_{i}.mp4" for i in range(3)]
        results = self.processor.process_video_batch(video_paths)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_process.call_count, 3)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_config_validation_success(self):
        """Test successful config validation"""
        # This should not raise an exception
        config.validate_config()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_config_validation_missing_key(self):
        """Test config validation with missing API key"""
        with self.assertRaises(ValueError):
            config.validate_config()


class TestCLIIntegration(unittest.TestCase):
    """Test CLI functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_cli_vectors")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cli_stats_command(self):
        """Test CLI stats command"""
        try:
            result = subprocess.run([
                'python', 'lecture_cli.py', 'stats', 
                '--db-path', self.test_db_path
            ], capture_output=True, text=True, timeout=30)
            
            self.assertEqual(result.returncode, 0)
            self.assertIn('Total chunks', result.stdout)
        except subprocess.TimeoutExpired:
            self.fail("CLI command timed out")
        except FileNotFoundError:
            self.skipTest("lecture_cli.py not found or Python not in PATH")


class TestTranscriptChunk(unittest.TestCase):
    """Test TranscriptChunk dataclass"""
    
    def test_transcript_chunk_creation(self):
        """Test creating TranscriptChunk instances"""
        chunk = TranscriptChunk(
            text="Test text",
            start_time=0.0,
            end_time=5.0,
            speaker="Speaker_1",
            confidence=0.95,
            chunk_id="test_001",
            video_file="test.mp4"
        )
        
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.start_time, 0.0)
        self.assertEqual(chunk.end_time, 5.0)
        self.assertEqual(chunk.speaker, "Speaker_1")
        self.assertEqual(chunk.confidence, 0.95)
        self.assertEqual(chunk.chunk_id, "test_001")
        self.assertEqual(chunk.video_file, "test.mp4")


def run_basic_tests():
    """Run a subset of basic tests for quick validation"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add basic tests
    suite.addTest(TestLectureProcessor('test_initialization'))
    suite.addTest(TestLectureProcessor('test_simple_speaker_assignment'))
    suite.addTest(TestLectureProcessor('test_transcript_chunking'))
    suite.addTest(TestLectureProcessor('test_statistics_generation'))
    suite.addTest(TestTranscriptChunk('test_transcript_chunk_creation'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run lecture processor tests')
    parser.add_argument('--basic', action='store_true', 
                       help='Run only basic tests (faster)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.basic:
        print("Running basic tests...")
        result = run_basic_tests()
    else:
        print("Running all tests...")
        result = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
