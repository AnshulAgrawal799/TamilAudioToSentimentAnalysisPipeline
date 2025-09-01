#!/usr/bin/env python3
"""
Transcriber factory for selecting between Sarvam and Whisper providers.
Provides a unified interface for transcription regardless of the underlying provider.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union

from transcribe import Transcriber as WhisperTranscriber, TranscriptionResult as WhisperTranscriptionResult
from sarvam_transcribe import SarvamTranscriber, SarvamTranscriptionResult

logger = logging.getLogger(__name__)


class UnifiedTranscriptionResult:
    """Unified container for transcription results from any provider."""
    
    def __init__(self, audio_file: Path, metadata: Dict[str, Any], provider: str, model_used: str):
        self.audio_file = audio_file
        self.metadata = metadata
        self.segments = []
        self.provider = provider
        self.model_used = model_used
        
    def add_segment(self, start: float, end: float, text: str, confidence: float = None, speaker: str = None):
        """Add a transcription segment."""
        segment = {
            'start': start,
            'end': end,
            'text': text.strip(),
            'confidence': confidence or 0.8,
            'speaker': speaker
        }
        self.segments.append(segment)
    
    def get_segments(self) -> List[Dict[str, Any]]:
        """Get all segments."""
        return self.segments


class TranscriberFactory:
    """Factory for creating appropriate transcriber instances."""
    
    @staticmethod
    def create_transcriber(config: Dict[str, Any]):
        """Create transcriber based on configuration."""
        provider = config.get('asr_provider', 'sarvam').lower()
        
        if provider == 'sarvam':
            logger.info("Creating Sarvam transcriber")
            return SarvamTranscriber(config)
        elif provider == 'whisper':
            logger.info("Creating Whisper transcriber (fallback)")
            return WhisperTranscriber(config)
        else:
            raise ValueError(f"Unsupported ASR provider: {provider}")


class UnifiedTranscriber:
    """Unified transcriber that works with any provider."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.provider = config.get('asr_provider', 'sarvam').lower()
        self.transcriber = TranscriberFactory.create_transcriber(config)
        
    def transcribe(self, audio_file: Path) -> UnifiedTranscriptionResult:
        """Transcribe audio file and return unified results."""
        if self.provider == 'sarvam':
            result = self.transcriber.transcribe(audio_file)
            return self._convert_sarvam_result(result)
        elif self.provider == 'whisper':
            result = self.transcriber.transcribe(audio_file)
            return self._convert_whisper_result(result)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def transcribe_batch(self, audio_files: List[Path]) -> List[UnifiedTranscriptionResult]:
        """Transcribe multiple audio files."""
        if self.provider == 'sarvam':
            results = self.transcriber.transcribe_batch(audio_files)
            return [self._convert_sarvam_result(result) for result in results]
        elif self.provider == 'whisper':
            # Whisper doesn't have batch processing, so process individually
            results = []
            for audio_file in audio_files:
                result = self.transcriber.transcribe(audio_file)
                results.append(self._convert_whisper_result(result))
            return results
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _convert_sarvam_result(self, result: SarvamTranscriptionResult) -> UnifiedTranscriptionResult:
        """Convert Sarvam result to unified format."""
        unified_result = UnifiedTranscriptionResult(
            result.audio_file,
            result.metadata,
            'sarvam',
            result.model_used
        )
        
        for segment in result.segments:
            unified_result.add_segment(
                segment['start'],
                segment['end'],
                segment['text'],
                segment.get('confidence'),
                segment.get('speaker')
            )
        
        return unified_result
    
    def _convert_whisper_result(self, result: WhisperTranscriptionResult) -> UnifiedTranscriptionResult:
        """Convert Whisper result to unified format."""
        unified_result = UnifiedTranscriptionResult(
            result.audio_file,
            result.metadata,
            'whisper',
            result.model_used
        )
        
        for segment in result.segments:
            unified_result.add_segment(
                segment['start'],
                segment['end'],
                segment['text'],
                segment.get('confidence')
            )
        
        return unified_result


def main():
    """Standalone execution for testing."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe audio files using unified transcriber')
    parser.add_argument('--config', default='../config.yaml', help='Configuration file path')
    parser.add_argument('--audio-file', required=True, help='Audio file to transcribe')
    parser.add_argument('--provider', choices=['sarvam', 'whisper'], help='Override provider from config')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override provider if specified
    if args.provider:
        config['asr_provider'] = args.provider
    
    # Run transcription
    transcriber = UnifiedTranscriber(config)
    audio_path = Path(args.audio_file)
    
    try:
        result = transcriber.transcribe(audio_path)
        print(f"Transcription complete for {audio_path.name}")
        print(f"Provider: {result.provider}")
        print(f"Model: {result.model_used}")
        print(f"Generated {len(result.segments)} segments")
        
        # Print first few segments
        for i, segment in enumerate(result.segments[:3]):
            print(f"Segment {i+1}: {segment['start']:.1f}s - {segment['end']:.1f}s")
            print(f"  Text: {segment['text']}")
            if segment.get('speaker'):
                print(f"  Speaker: {segment['speaker']}")
            print()
            
    except Exception as e:
        print(f"Transcription failed: {e}")


if __name__ == "__main__":
    main()
