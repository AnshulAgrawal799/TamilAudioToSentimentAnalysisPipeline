#!/usr/bin/env python3
"""
Audio file splitting module for handling long audio files.
Splits audio files longer than 30 seconds into chunks that can be processed by the sync API.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Tuple
import wave
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.info("librosa not available. Falling back to wave-based splitting.")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.info("pydub not available. Falling back to wave-based splitting.")


class AudioSplitter:
    """Splits long audio files into chunks for processing."""
    
    def __init__(self, max_chunk_duration: float = 25.0, overlap_duration: float = 1.0):
        """
        Initialize the audio splitter.
        
        Args:
            max_chunk_duration: Maximum duration of each chunk in seconds (default: 25s)
            overlap_duration: Overlap between chunks in seconds (default: 1s)
        """
        # Sanitize inputs
        self.max_chunk_duration = max(0.1, float(max_chunk_duration))
        self.overlap_duration = max(0.0, float(overlap_duration))
    
    def split_audio_file(self, audio_path: Path, output_dir: Path = None) -> List[Tuple[Path, float, float]]:
        """
        Split an audio file into chunks.
        
        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save chunks (default: same directory as input)
            
        Returns:
            List of tuples: (chunk_path, start_time, end_time)
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if output_dir is None:
            output_dir = audio_path.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get audio duration
        duration = self._get_audio_duration(audio_path)
        
        if duration <= self.max_chunk_duration:
            logger.info(f"Audio file {audio_path.name} is {duration:.1f}s, no splitting needed")
            return [(audio_path, 0.0, duration)]
        
        logger.info(f"Splitting {audio_path.name} ({duration:.1f}s) into chunks")
        
        # Calculate chunk boundaries
        chunks = self._calculate_chunks(duration)
        
        # Split the audio
        chunk_files = []
        for i, (start_time, end_time) in enumerate(chunks):
            chunk_path = self._create_chunk(audio_path, output_dir, i, start_time, end_time)
            chunk_files.append((chunk_path, start_time, end_time))
            logger.info(f"Created chunk {i+1}/{len(chunks)}: {chunk_path.name} ({start_time:.1f}s - {end_time:.1f}s)")
        
        return chunk_files
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file."""
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except Exception as e:
            logger.error(f"Failed to get duration for {audio_path}: {e}")
            raise
    
    def _calculate_chunks(self, total_duration: float) -> List[Tuple[float, float]]:
        """Calculate chunk boundaries with overlap."""
        chunks = []
        start_time = 0.0
        # Ensure overlap is strictly less than chunk duration to guarantee progress
        effective_overlap = min(self.overlap_duration, max(self.max_chunk_duration - 0.01, 0.0))

        max_chunks = int(total_duration / max(self.max_chunk_duration - effective_overlap, 0.01)) + 5
        iterations = 0
        
        while start_time < total_duration and iterations < max_chunks:
            end_time = min(start_time + self.max_chunk_duration, total_duration)
            chunks.append((start_time, end_time))
            next_start = end_time - effective_overlap
            if next_start <= start_time:
                # Force progress
                next_start = start_time + self.max_chunk_duration
            start_time = next_start
            iterations += 1
        
        return chunks
    
    def _create_chunk(self, audio_path: Path, output_dir: Path, chunk_index: int, 
                     start_time: float, end_time: float) -> Path:
        """Create a chunk file from the original audio."""
        chunk_filename = f"{audio_path.stem}_chunk_{chunk_index:03d}.wav"
        chunk_path = output_dir / chunk_filename
        
        try:
            # Prefer wave-based slicing for reliability; other libs are optional upgrades
            self._create_chunk_wave(audio_path, chunk_path, start_time, end_time)
        except Exception as wave_err:
            logger.warning(f"Wave slicing failed for chunk {chunk_index} ({start_time:.1f}-{end_time:.1f}s): {wave_err}")
            try:
                if PYDUB_AVAILABLE:
                    self._create_chunk_pydub(audio_path, chunk_path, start_time, end_time)
                elif LIBROSA_AVAILABLE:
                    self._create_chunk_librosa(audio_path, chunk_path, start_time, end_time)
                else:
                    raise wave_err
            except Exception as e:
                import traceback
                logger.error(f"Failed to create chunk {chunk_index} for {audio_path.name}: {e}\n{traceback.format_exc()}")
                # If all methods fail, try a simple copy of the original file
                logger.warning(f"Falling back to copying original file for {audio_path.name}")
                import shutil
                shutil.copy2(audio_path, chunk_path)
        
        return chunk_path

    def create_single_chunk(self, audio_path: Path, output_dir: Path, start_time: float, end_time: float) -> Tuple[Path, float, float]:
        """Create exactly one chunk without enumerating all boundaries."""
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = self._create_chunk(audio_path, output_dir, 0, start_time, end_time)
        logger.info(f"Created single chunk: {chunk_path.name} ({start_time:.1f}s - {end_time:.1f}s)")
        return (chunk_path, start_time, end_time)
    
    def _create_chunk_pydub(self, audio_path: Path, chunk_path: Path, 
                           start_time: float, end_time: float):
        """Create chunk using pydub (most reliable)."""
        try:
            audio = AudioSegment.from_wav(str(audio_path))
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            chunk = audio[start_ms:end_ms]
            chunk.export(str(chunk_path), format="wav")
        except Exception as e:
            logger.error(f"Failed to create chunk with pydub: {e}")
            raise
    
    def _create_chunk_librosa(self, audio_path: Path, chunk_path: Path, 
                             start_time: float, end_time: float):
        """Create chunk using librosa."""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=None)
            
            # Calculate sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract chunk
            chunk = y[start_sample:end_sample]
            
            # Save chunk
            import soundfile as sf
            sf.write(str(chunk_path), chunk, sr)
        except Exception as e:
            logger.error(f"Failed to create chunk with librosa: {e}")
            raise
    
    def _create_chunk_wave(self, audio_path: Path, chunk_path: Path, 
                          start_time: float, end_time: float):
        """Create chunk using basic wave module (fallback)."""
        try:
            with wave.open(str(audio_path), 'rb') as input_wav:
                # Get audio parameters
                channels = input_wav.getnchannels()
                sample_width = input_wav.getsampwidth()
                frame_rate = input_wav.getframerate()
                
                # Calculate frame indices
                start_frame = int(start_time * frame_rate)
                end_frame = int(end_time * frame_rate)
                
                # Ensure we don't read beyond the file
                total_frames = input_wav.getnframes()
                if end_frame > total_frames:
                    end_frame = total_frames
                if start_frame >= total_frames:
                    raise ValueError(f"Start time {start_time}s is beyond audio duration")
                
                # Read frames
                input_wav.setpos(start_frame)
                frames = input_wav.readframes(end_frame - start_frame)
                
                # Write chunk
                with wave.open(str(chunk_path), 'wb') as output_wav:
                    output_wav.setnchannels(channels)
                    output_wav.setsampwidth(sample_width)
                    output_wav.setframerate(frame_rate)
                    output_wav.writeframes(frames)
                    
                logger.debug(f"Created chunk {chunk_path.name} ({start_time:.1f}s - {end_time:.1f}s)")
                
        except Exception as e:
            logger.error(f"Failed to create chunk with wave module: {e}")
            raise
    
    def cleanup_chunks(self, chunk_files: List[Tuple[Path, float, float]]):
        """Clean up temporary chunk files."""
        for chunk_path, _, _ in chunk_files:
            try:
                if chunk_path.exists() and chunk_path.name.endswith('_chunk_'):
                    chunk_path.unlink()
                    logger.debug(f"Cleaned up chunk file: {chunk_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup chunk file {chunk_path}: {e}")


def split_audio_for_transcription(audio_path: Path, max_duration: float = 25.0) -> List[Tuple[Path, float, float]]:
    """
    Convenience function to split audio files for transcription.
    
    Args:
        audio_path: Path to the audio file
        max_duration: Maximum duration per chunk in seconds
        
    Returns:
        List of (chunk_path, start_time, end_time) tuples
    """
    splitter = AudioSplitter(max_chunk_duration=max_duration)
    return splitter.split_audio_file(audio_path)
