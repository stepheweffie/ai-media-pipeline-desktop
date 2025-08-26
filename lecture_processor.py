"""
Lecture Processor for Media AI Pipeline
Handles bulk transcription with speaker diarization and vectorization
"""
import os
import json
import logging
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import openai
import whisper
from pathlib import Path
import chromadb
from chromadb.config import Settings
import hashlib
import numpy as np
from dataclasses import dataclass
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript with metadata"""
    text: str
    start_time: float
    end_time: float
    speaker: str
    confidence: float
    chunk_id: str
    video_file: str
    
class LectureProcessor:
    def __init__(self, vector_db_path: str = "./lecture_vectors"):
        """Initialize the lecture processor"""
        config.validate_config()
        openai.api_key = config.OPENAI_API_KEY
        
        # Initialize Whisper model (you can change size based on accuracy needs)
        # Options: tiny, base, small, medium, large, large-v2, large-v3
        self.whisper_model = None  # Will load on first use
        self.model_size = "large-v3"  # Best accuracy for lectures
        
        # Initialize ChromaDB
        self.vector_db_path = vector_db_path
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection for lectures
        self.collection = self.chroma_client.get_or_create_collection(
            name="lecture_transcripts",
            metadata={"description": "Transcribed lecture content with speaker diarization"}
        )
        
        logger.info(f"LectureProcessor initialized with vector DB at {vector_db_path}")
    
    def load_whisper_model(self):
        """Load Whisper model on demand"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract audio from video file"""
        try:
            if output_path is None:
                # Create temp file for audio
                temp_dir = tempfile.mkdtemp()
                filename = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(temp_dir, f"{filename}.wav")
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz sample rate (good for speech)
                '-ac', '1',      # Mono
                '-y',            # Overwrite output file
                '-loglevel', 'quiet',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            if not os.path.exists(output_path):
                raise Exception("Audio file was not created")
            
            logger.info(f"Audio extracted to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            raise
    
    def transcribe_with_diarization(self, audio_path: str) -> List[Dict]:
        """Transcribe audio with speaker diarization using Whisper + pyannote"""
        try:
            # Load Whisper model if not already loaded
            self.load_whisper_model()
            
            # First, get basic transcription with timestamps from Whisper
            logger.info("Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en"  # Change if needed
            )
            
            # For now, we'll use a simple speaker assignment based on pause detection
            # In production, you'd want to use pyannote.audio for proper diarization
            segments_with_speakers = self.assign_speakers_simple(result["segments"])
            
            logger.info(f"Transcription completed with {len(segments_with_speakers)} segments")
            return segments_with_speakers
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            raise
    
    def assign_speakers_simple(self, segments: List[Dict]) -> List[Dict]:
        """Simple speaker assignment based on pause detection"""
        # This is a simplified approach - for production use pyannote.audio
        enriched_segments = []
        current_speaker = "Speaker_1"
        speaker_count = 1
        
        for i, segment in enumerate(segments):
            # Simple heuristic: if there's a long pause (>2s), might be speaker change
            if i > 0:
                prev_end = segments[i-1]["end"]
                current_start = segment["start"]
                pause_duration = current_start - prev_end
                
                # If long pause, potentially new speaker
                if pause_duration > 2.0:
                    # Simple alternating logic - in reality, use voice fingerprinting
                    speaker_count += 1
                    current_speaker = f"Speaker_{((speaker_count - 1) % 3) + 1}"  # Max 3 speakers
            
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = current_speaker
            enriched_segments.append(segment_with_speaker)
        
        return enriched_segments
    
    def transcribe_with_pyannote_diarization(self, audio_path: str) -> List[Dict]:
        """Advanced transcription with pyannote.audio diarization (requires additional setup)"""
        try:
            # This requires: pip install pyannote.audio
            # And a Hugging Face token for the diarization model
            from pyannote.audio import Pipeline
            
            # Load the diarization pipeline (requires HF token)
            if not config.HUGGING_FACE_TOKEN:
                raise Exception("HUGGING_FACE_TOKEN not found in .env file. Please add it for advanced diarization.")
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=config.HUGGING_FACE_TOKEN
            )
            
            # Apply diarization
            logger.info("Performing speaker diarization...")
            diarization = pipeline(audio_path)
            
            # Load Whisper for transcription
            self.load_whisper_model()
            
            # Get Whisper transcription with word timestamps
            logger.info("Transcribing with Whisper...")
            whisper_result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en"
            )
            
            # Align Whisper segments with speaker diarization
            aligned_segments = self.align_transcription_with_diarization(
                whisper_result["segments"], 
                diarization
            )
            
            return aligned_segments
            
        except ImportError:
            logger.warning("pyannote.audio not available, falling back to simple diarization")
            return self.transcribe_with_diarization(audio_path)
        except Exception as e:
            logger.error(f"Error with pyannote diarization: {e}")
            # Fall back to simple method
            return self.transcribe_with_diarization(audio_path)
    
    def align_transcription_with_diarization(self, segments: List[Dict], diarization) -> List[Dict]:
        """Align Whisper transcription with pyannote speaker diarization"""
        aligned_segments = []
        
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Find the dominant speaker for this segment
            speaker_times = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(start_time, turn.start)
                overlap_end = min(end_time, turn.end)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    if speaker not in speaker_times:
                        speaker_times[speaker] = 0
                    speaker_times[speaker] += overlap_duration
            
            # Assign to the speaker with most overlap
            if speaker_times:
                dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            else:
                dominant_speaker = "Unknown"
            
            aligned_segment = segment.copy()
            aligned_segment["speaker"] = dominant_speaker
            aligned_segments.append(aligned_segment)
        
        return aligned_segments
    
    def chunk_transcript(self, segments: List[Dict], video_file: str, chunk_duration: float = 60.0) -> List[TranscriptChunk]:
        """Chunk transcript into manageable pieces for vectorization"""
        chunks = []
        current_chunk = []
        current_start = None
        current_speaker = None
        chunk_counter = 0
        
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_text = segment["text"].strip()
            segment_speaker = segment["speaker"]
            
            if not segment_text:
                continue
            
            # Start new chunk if:
            # 1. No current chunk
            # 2. Speaker changed
            # 3. Time gap is too long
            start_new_chunk = (
                current_start is None or
                current_speaker != segment_speaker or
                (segment_start - current_start) >= chunk_duration
            )
            
            if start_new_chunk and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join([s["text"].strip() for s in current_chunk])
                if chunk_text:
                    chunk_end = current_chunk[-1]["end"]
                    chunk_id = f"{os.path.basename(video_file)}_{chunk_counter:04d}"
                    
                    chunks.append(TranscriptChunk(
                        text=chunk_text,
                        start_time=current_start,
                        end_time=chunk_end,
                        speaker=current_speaker,
                        confidence=np.mean([s.get("confidence", 0.9) for s in current_chunk]),
                        chunk_id=chunk_id,
                        video_file=video_file
                    ))
                    chunk_counter += 1
                
                # Start new chunk
                current_chunk = [segment]
                current_start = segment_start
                current_speaker = segment_speaker
            else:
                # Add to current chunk
                if current_start is None:
                    current_start = segment_start
                    current_speaker = segment_speaker
                current_chunk.append(segment)
        
        # Finalize last chunk
        if current_chunk:
            chunk_text = " ".join([s["text"].strip() for s in current_chunk])
            if chunk_text:
                chunk_end = current_chunk[-1]["end"]
                chunk_id = f"{os.path.basename(video_file)}_{chunk_counter:04d}"
                
                chunks.append(TranscriptChunk(
                    text=chunk_text,
                    start_time=current_start,
                    end_time=chunk_end,
                    speaker=current_speaker,
                    confidence=np.mean([s.get("confidence", 0.9) for s in current_chunk]),
                    chunk_id=chunk_id,
                    video_file=video_file
                ))
        
        logger.info(f"Created {len(chunks)} transcript chunks")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using OpenAI"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",  # or text-embedding-3-large for better quality
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def store_chunks_in_vector_db(self, chunks: List[TranscriptChunk]):
        """Store transcript chunks in ChromaDB with embeddings"""
        try:
            if not chunks:
                logger.warning("No chunks to store")
                return
            
            # Prepare data for ChromaDB
            texts = [chunk.text for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    "video_file": chunk.video_file,
                    "speaker": chunk.speaker,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.end_time - chunk.start_time,
                    "confidence": chunk.confidence,
                    "timestamp_formatted": str(timedelta(seconds=int(chunk.start_time))),
                    "processed_at": datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.get_embeddings(texts)
            
            # Store in ChromaDB
            logger.info("Storing chunks in vector database...")
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"Error storing chunks in vector DB: {e}")
            raise
    
    def process_video(self, video_path: str, use_advanced_diarization: bool = False) -> Dict:
        """Process a single video file"""
        try:
            logger.info(f"Processing video: {os.path.basename(video_path)}")
            start_time = datetime.now()
            
            # Extract audio
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = self.extract_audio(video_path, 
                    os.path.join(temp_dir, "audio.wav"))
                
                # Transcribe with diarization
                if use_advanced_diarization:
                    segments = self.transcribe_with_pyannote_diarization(audio_path)
                else:
                    segments = self.transcribe_with_diarization(audio_path)
                
                # Chunk transcript
                chunks = self.chunk_transcript(segments, video_path)
                
                # Store in vector database
                self.store_chunks_in_vector_db(chunks)
                
                processing_time = datetime.now() - start_time
                
                result = {
                    "video_file": video_path,
                    "processing_time": str(processing_time),
                    "total_segments": len(segments),
                    "total_chunks": len(chunks),
                    "speakers": list(set(segment["speaker"] for segment in segments)),
                    "duration": segments[-1]["end"] if segments else 0,
                    "status": "success"
                }
                
                logger.info(f"Successfully processed {os.path.basename(video_path)} "
                          f"in {processing_time}")
                return result
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return {
                "video_file": video_path,
                "status": "error",
                "error": str(e)
            }
    
    def process_video_batch(self, video_paths: List[str], 
                           use_advanced_diarization: bool = False) -> List[Dict]:
        """Process multiple videos"""
        results = []
        total_videos = len(video_paths)
        
        logger.info(f"Starting batch processing of {total_videos} videos")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"Processing video {i}/{total_videos}")
            result = self.process_video(video_path, use_advanced_diarization)
            results.append(result)
            
            # Log progress
            if i % 5 == 0 or i == total_videos:
                successful = sum(1 for r in results if r["status"] == "success")
                logger.info(f"Progress: {i}/{total_videos} videos processed "
                          f"({successful} successful)")
        
        return results
    
    def search_lectures(self, query: str, n_results: int = 10, 
                       speaker_filter: Optional[str] = None,
                       video_filter: Optional[str] = None) -> List[Dict]:
        """Search lecture transcripts semantically"""
        try:
            # Build where clause for filters
            where = {}
            if speaker_filter:
                where["speaker"] = speaker_filter
            if video_filter:
                where["video_file"] = {"$contains": video_filter}
            
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching lectures: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get statistics about processed lectures"""
        try:
            collection_count = self.collection.count()
            
            # Get some sample metadata to analyze
            sample_results = self.collection.get(limit=min(1000, collection_count))
            
            if sample_results["metadatas"]:
                speakers = set()
                videos = set()
                total_duration = 0
                
                for metadata in sample_results["metadatas"]:
                    speakers.add(metadata.get("speaker", "Unknown"))
                    videos.add(metadata.get("video_file", "Unknown"))
                    total_duration += metadata.get("duration", 0)
                
                return {
                    "total_chunks": collection_count,
                    "unique_speakers": len(speakers),
                    "unique_videos": len(videos),
                    "total_content_hours": round(total_duration / 3600, 2),
                    "speakers": list(speakers),
                    "videos": [os.path.basename(v) for v in videos if v != "Unknown"]
                }
            else:
                return {
                    "total_chunks": collection_count,
                    "unique_speakers": 0,
                    "unique_videos": 0,
                    "total_content_hours": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

def main():
    """Example usage"""
    processor = LectureProcessor()
    
    # Get statistics
    stats = processor.get_statistics()
    print(f"Current database stats: {stats}")
    
    # Example: process a single video
    # video_path = "/path/to/your/lecture.mp4"
    # result = processor.process_video(video_path)
    # print(f"Processing result: {result}")
    
    # Example: search lectures
    # results = processor.search_lectures("machine learning algorithms")
    # for result in results[:3]:
    #     print(f"Speaker: {result['metadata']['speaker']}")
    #     print(f"Time: {result['metadata']['timestamp_formatted']}")
    #     print(f"Text: {result['text'][:200]}...")
    #     print("---")

if __name__ == "__main__":
    main()
