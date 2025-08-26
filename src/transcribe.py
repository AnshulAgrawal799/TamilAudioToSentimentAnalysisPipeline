#!/usr/bin/env python3
"""
Transcription module using Whisper ASR for Tamil audio.
Handles audio transcription and segment generation.
"""

import logging
import whisper
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """Container for transcription results."""
    
    def __init__(self, audio_file: Path, metadata: Dict[str, Any]):
        self.audio_file = audio_file
        self.metadata = metadata
        self.segments = []
        self.language = "ta"  # Tamil
        self.model_used = None
        
    def add_segment(self, start: float, end: float, text: str, confidence: float = None):
        """Add a transcription segment."""
        segment = {
            'start': start,
            'end': end,
            'text': text.strip(),
            'confidence': confidence or 0.8
        }
        self.segments.append(segment)
    
    def get_segments(self) -> List[Dict[str, Any]]:
        """Get all segments."""
        return self.segments


class Transcriber:
    """Handles audio transcription using Whisper."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.model_name = config.get('asr_model', 'whisper-small')
        self.language = config.get('language', 'ta')
        self.model = None
        
    def load_model(self):
        """Load Whisper model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            try:
                self.model = whisper.load_model(self.model_name)
                logger.info(f"Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def transcribe(self, audio_file: Path) -> TranscriptionResult:
        """Transcribe audio file and return results."""
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load model if not already loaded
        self.load_model()
        
        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(audio_file.name)
        
        # Create result container
        result = TranscriptionResult(audio_file, metadata)
        result.model_used = self.model_name
        
        try:
            logger.info(f"Transcribing: {audio_file.name}")
            
            # Run Whisper transcription with improved settings
            whisper_result = self.model.transcribe(
                str(audio_file),
                language=self.language,
                task="transcribe",
                verbose=False,
                word_timestamps=True,  # Get word-level timestamps for better segmentation
                condition_on_previous_text=True,  # Use context from previous segments
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,  # Filter out low-quality segments
                logprob_threshold=-1.0,  # Filter out low-confidence segments
                no_speech_threshold=0.6  # Better silence detection
            )
            
            # Extract segments from Whisper result with quality filtering
            if 'segments' in whisper_result:
                for segment in whisper_result['segments']:
                    start = segment.get('start', 0.0)
                    end = segment.get('end', start + 1.0)
                    text = segment.get('text', '').strip()
                    avg_logprob = segment.get('avg_logprob', -10.0)  # Whisper confidence
                    
                    # Quality filtering: only include segments with reasonable confidence
                    if text and avg_logprob > -2.0:  # Filter out very low confidence segments
                        result.add_segment(start, end, text, confidence=min(0.95, max(0.3, (avg_logprob + 10) / 10)))
                        
                logger.info(f"Generated {len(result.segments)} quality-filtered segments from {audio_file.name}")
            else:
                logger.warning(f"No segments found in Whisper result for {audio_file.name}")
                
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file.name}: {e}")
            raise
        
        return result
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract metadata from filename if it follows the convention."""
        # Expected format: seller_<SELLERID>_STOP<STOPID>_<YYYYMMDD>_<HHMMSS>.wav
        parts = filename.replace('.wav', '').split('_')
        metadata = {}
        
        try:
            if len(parts) >= 5 and parts[0].lower() == 'seller':
                metadata['seller_id'] = parts[1]
                metadata['stop_id'] = parts[2]
                
                # Parse date and time
                date_part = parts[3]  # YYYYMMDD
                time_part = parts[4]  # HHMMSS
                
                if len(date_part) == 8 and len(time_part) == 6:
                    timestamp = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}T{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}Z"
                    metadata['recording_start'] = timestamp
                    
        except (IndexError, ValueError) as e:
            logger.debug(f"Could not parse metadata from filename {filename}: {e}")
        
        return metadata


def main():
    """Standalone execution for testing."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe audio files using Whisper')
    parser.add_argument('--config', default='../config.yaml', help='Configuration file path')
    parser.add_argument('--audio-file', required=True, help='Audio file to transcribe')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run transcription
    transcriber = Transcriber(config)
    audio_path = Path(args.audio_file)
    
    try:
        result = transcriber.transcribe(audio_path)
        print(f"Transcription complete for {audio_path.name}")
        print(f"Generated {len(result.segments)} segments")
        
        # Print first few segments
        for i, segment in enumerate(result.segments[:3]):
            print(f"Segment {i+1}: {segment['start']:.1f}s - {segment['end']:.1f}s")
            print(f"  Text: {segment['text']}")
            print()
            
    except Exception as e:
        print(f"Transcription failed: {e}")


if __name__ == "__main__":
    main()
