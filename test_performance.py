#!/usr/bin/env python3
"""
Performance and Integration Tests for Lecture Processor
Tests system performance, memory usage, and real-world scenarios
"""
import os
import sys
import time
import tempfile
import shutil
import subprocess
import psutil
from pathlib import Path
import unittest
from unittest.mock import patch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lecture_processor import LectureProcessor
from test_lecture_processor import TestLectureProcessor

class PerformanceTestCase(unittest.TestCase):
    """Base class for performance tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "perf_test_vectors")
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
    
    def tearDown(self):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        print(f"\n📊 Performance Metrics:")
        print(f"   ⏱️  Duration: {duration:.2f}s")
        print(f"   💾 Memory delta: {memory_delta:+.1f}MB")
        print(f"   💾 Final memory: {end_memory:.1f}MB")
        
        shutil.rmtree(self.temp_dir)

class TestProcessorPerformance(PerformanceTestCase):
    """Test processor performance with various loads"""
    
    def test_processor_initialization_speed(self):
        """Test how quickly the processor initializes"""
        start = time.time()
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        init_time = time.time() - start
        
        self.assertLess(init_time, 5.0, "Processor initialization should be fast")
        print(f"   🚀 Initialization: {init_time:.2f}s")
    
    def test_chunk_processing_speed(self):
        """Test chunking performance with many segments"""
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        # Create many mock segments
        segments = []
        for i in range(1000):  # 1000 segments
            segments.append({
                "start": i * 5.0,
                "end": (i + 1) * 5.0 - 0.1,
                "text": f"This is segment number {i} with some sample text for testing chunking performance.",
                "speaker": f"Speaker_{(i % 3) + 1}"
            })
        
        start = time.time()
        chunks = processor.chunk_transcript(segments, "test_video.mp4", chunk_duration=30.0)
        chunk_time = time.time() - start
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(chunk_time, 10.0, "Chunking should complete within reasonable time")
        
        chunks_per_second = len(segments) / chunk_time
        print(f"   📝 Chunking rate: {chunks_per_second:.0f} segments/sec")
        print(f"   📦 Generated {len(chunks)} chunks from {len(segments)} segments")
    
    @patch('lecture_processor.openai.embeddings.create')
    def test_vector_storage_performance(self, mock_embeddings):
        """Test vector storage performance"""
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        # Mock embeddings
        embedding_size = 1536  # text-embedding-3-small size
        mock_response = type('MockResponse', (), {})()
        mock_response.data = []
        
        # Create test chunks
        from lecture_processor import TranscriptChunk
        chunks = []
        num_chunks = 100
        
        for i in range(num_chunks):
            chunks.append(TranscriptChunk(
                text=f"Test chunk {i} with some content to embed and store in the vector database.",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                speaker=f"Speaker_{(i % 2) + 1}",
                confidence=0.9,
                chunk_id=f"perf_test_{i:04d}",
                video_file="performance_test.mp4"
            ))
            
            # Add mock embedding for this chunk
            mock_response.data.append(
                type('MockEmbedding', (), {'embedding': [0.1] * embedding_size})()
            )
        
        mock_embeddings.return_value = mock_response
        
        start = time.time()
        processor.store_chunks_in_vector_db(chunks)
        storage_time = time.time() - start
        
        chunks_per_second = num_chunks / storage_time
        print(f"   💾 Storage rate: {chunks_per_second:.0f} chunks/sec")
        
        # Verify storage
        stats = processor.get_statistics()
        self.assertEqual(stats['total_chunks'], num_chunks)
    
    @patch('lecture_processor.openai.embeddings.create')
    def test_search_performance(self, mock_embeddings):
        """Test search performance with large dataset"""
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        # Mock embeddings for storage
        embedding_size = 1536
        mock_response = type('MockResponse', (), {})()
        mock_response.data = [
            type('MockEmbedding', (), {'embedding': [0.1] * embedding_size})()
            for _ in range(50)  # 50 chunks
        ]
        mock_embeddings.return_value = mock_response
        
        # Store test data
        from lecture_processor import TranscriptChunk
        chunks = []
        for i in range(50):
            chunks.append(TranscriptChunk(
                text=f"Machine learning content {i} discussing algorithms, neural networks, and deep learning concepts.",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                speaker=f"Speaker_{(i % 3) + 1}",
                confidence=0.9,
                chunk_id=f"search_test_{i:04d}",
                video_file="search_test.mp4"
            ))
        
        processor.store_chunks_in_vector_db(chunks)
        
        # Test search performance
        search_queries = [
            "machine learning algorithms",
            "neural networks", 
            "deep learning concepts",
            "artificial intelligence",
            "data science methods"
        ]
        
        total_search_time = 0
        for query in search_queries:
            start = time.time()
            results = processor.search_lectures(query, n_results=10)
            search_time = time.time() - start
            total_search_time += search_time
            
            self.assertIsInstance(results, list)
        
        avg_search_time = total_search_time / len(search_queries)
        print(f"   🔍 Average search time: {avg_search_time:.3f}s")
        print(f"   🔍 Total searches: {len(search_queries)}")

class TestMemoryUsage(PerformanceTestCase):
    """Test memory usage patterns"""
    
    def test_memory_usage_during_processing(self):
        """Monitor memory usage during typical operations"""
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        # Measure baseline memory
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Simulate processing steps
        segments = []
        for i in range(500):
            segments.append({
                "start": i * 2.0,
                "end": (i + 1) * 2.0,
                "text": f"Memory test segment {i} " * 10,  # Make it longer
                "speaker": f"Speaker_{(i % 2) + 1}"
            })
        
        # Test chunking memory usage
        chunks = processor.chunk_transcript(segments, "memory_test.mp4")
        chunk_memory = psutil.Process().memory_info().rss / 1024 / 1024
        chunk_delta = chunk_memory - baseline_memory
        
        print(f"   📊 Memory after chunking: +{chunk_delta:.1f}MB")
        
        # Memory should not grow excessively
        self.assertLess(chunk_delta, 100, "Memory usage should be reasonable")

class TestScalability(PerformanceTestCase):
    """Test scalability with increasing loads"""
    
    def test_chunking_scalability(self):
        """Test how chunking scales with input size"""
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        test_sizes = [10, 100, 500, 1000]
        results = []
        
        for size in test_sizes:
            segments = []
            for i in range(size):
                segments.append({
                    "start": i * 3.0,
                    "end": (i + 1) * 3.0,
                    "text": f"Scalability test segment {i}",
                    "speaker": f"Speaker_{(i % 2) + 1}"
                })
            
            start = time.time()
            chunks = processor.chunk_transcript(segments, "scalability_test.mp4")
            duration = time.time() - start
            
            rate = size / duration if duration > 0 else float('inf')
            results.append((size, duration, rate))
            
            print(f"   📈 {size} segments: {duration:.3f}s ({rate:.0f} seg/s)")
        
        # Check that performance scales reasonably (not exponentially worse)
        if len(results) >= 2:
            first_rate = results[0][2]
            last_rate = results[-1][2]
            degradation = first_rate / last_rate if last_rate > 0 else float('inf')
            
            self.assertLess(degradation, 10, "Performance should not degrade severely")

class TestIntegration(PerformanceTestCase):
    """Integration tests with real file operations"""
    
    def test_dummy_video_processing(self):
        """Test processing with actual dummy files"""
        processor = LectureProcessor(vector_db_path=self.test_db_path)
        
        # Create dummy video file
        video_path = os.path.join(self.temp_dir, "integration_test.mp4")
        with open(video_path, 'wb') as f:
            f.write(b'fake video data' * 1000)  # Make it somewhat substantial
        
        # Mock the processing pipeline
        with patch.object(processor, 'extract_audio') as mock_extract, \
             patch.object(processor, 'transcribe_with_diarization') as mock_transcribe, \
             patch.object(processor, 'store_chunks_in_vector_db') as mock_store:
            
            # Setup mocks
            audio_path = os.path.join(self.temp_dir, "audio.wav")
            mock_extract.return_value = audio_path
            mock_transcribe.return_value = [
                {"start": 0.0, "end": 10.0, "text": "Integration test content", "speaker": "Speaker_1"}
            ]
            mock_store.return_value = None
            
            start = time.time()
            result = processor.process_video(video_path)
            processing_time = time.time() - start
            
            self.assertEqual(result['status'], 'success')
            print(f"   🎬 Video processing: {processing_time:.2f}s")

def benchmark_full_pipeline():
    """Benchmark the complete pipeline with realistic data"""
    print("\n🏃‍♂️ Running Full Pipeline Benchmark...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        processor = LectureProcessor(vector_db_path=os.path.join(temp_dir, "benchmark_vectors"))
        
        # Simulate 1 hour of lecture content (realistic scenario)
        segments = []
        for i in range(1200):  # 1200 segments = ~1 hour at 3s per segment
            segments.append({
                "start": i * 3.0,
                "end": (i + 1) * 3.0,
                "text": f"Lecture segment {i} discussing various academic topics including "
                       f"mathematics, science, literature, and critical thinking skills.",
                "speaker": f"Speaker_{(i % 4) + 1}"  # 4 speakers (professor + 3 students)
            })
        
        print(f"📚 Processing {len(segments)} segments (~1 hour lecture)")
        
        # Time chunking
        start = time.time()
        chunks = processor.chunk_transcript(segments, "benchmark_lecture.mp4", chunk_duration=60.0)
        chunk_time = time.time() - start
        
        print(f"📝 Chunking: {chunk_time:.2f}s ({len(chunks)} chunks)")
        
        # Simulate storage (without actual OpenAI calls)
        with patch.object(processor, 'get_embeddings') as mock_embeddings:
            mock_embeddings.return_value = [[0.1] * 1536 for _ in chunks]  # Mock embeddings
            
            start = time.time()
            processor.store_chunks_in_vector_db(chunks)
            storage_time = time.time() - start
            
            print(f"💾 Storage: {storage_time:.2f}s")
        
        # Test search performance
        start = time.time()
        results = processor.search_lectures("mathematics and science", n_results=20)
        search_time = time.time() - start
        
        print(f"🔍 Search: {search_time:.3f}s ({len(results)} results)")
        
        # Get final stats
        stats = processor.get_statistics()
        print(f"📊 Final stats: {stats['total_chunks']} chunks, {stats['unique_speakers']} speakers")
        
    finally:
        shutil.rmtree(temp_dir)

def run_performance_tests():
    """Run all performance tests"""
    print("🚀 Running Performance Test Suite...\n")
    
    # Run benchmark first
    benchmark_full_pipeline()
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add performance test cases
    suite.addTests(loader.loadTestsFromTestCase(TestProcessorPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryUsage))
    suite.addTests(loader.loadTestsFromTestCase(TestScalability))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance tests')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Run only the full pipeline benchmark')
    
    args = parser.parse_args()
    
    if args.benchmark_only:
        benchmark_full_pipeline()
    else:
        result = run_performance_tests()
        sys.exit(0 if result.wasSuccessful() else 1)
