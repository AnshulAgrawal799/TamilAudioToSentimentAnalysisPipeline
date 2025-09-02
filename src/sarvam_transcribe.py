#!/usr/bin/env python3
"""
Sarvam Speech-to-Text transcription module for Tamil audio.
Handles audio transcription using Sarvam's Saarika model with batch processing.
"""

import logging
import hashlib
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SarvamTranscriptionResult:
    """Container for Sarvam transcription results."""
    
    def __init__(self, audio_file: Path, metadata: Dict[str, Any]):
        self.audio_file = audio_file
        self.metadata = metadata
        self.segments = []
        self.language = "ta-IN"  # Tamil
        self.model_used = "saarika:v2.5"
        
    def add_segment(self, start: float, end: float, text: str, confidence: float = None, speaker: str = None):
        """Add a transcription segment."""
        # Calculate improved confidence based on text quality
        if confidence is None:
            # Base confidence from Sarvam API
            base_confidence = 0.8
            
            # Boost confidence for longer, well-formed text
            if len(text.strip()) > 10:
                base_confidence += 0.1
            
            # Boost confidence for text with proper Tamil characters
            tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
            if tamil_chars > len(text) * 0.3:  # More than 30% Tamil characters
                base_confidence += 0.05
            
            # Reduce confidence for very short text
            if len(text.strip()) < 5:
                base_confidence -= 0.1
            
            confidence = min(0.95, max(0.6, base_confidence))
        
        segment = {
            'start': start,
            'end': end,
            'text': text.strip(),
            'confidence': confidence,
            'speaker': speaker
        }
        self.segments.append(segment)
    
    def get_segments(self) -> List[Dict[str, Any]]:
        """Get all segments."""
        return self.segments


class TranscriptCache:
    """Simple file-based transcript cache to avoid re-transcribing."""
    
    def __init__(self, cache_dir: str, ttl_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        
    def _get_cache_key(self, audio_path: Path, model: str, language_code: str) -> str:
        """Generate cache key from audio file hash and model parameters."""
        h = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return f"{h.hexdigest()}_{model}_{language_code}.json"
    
    def get(self, audio_path: Path, model: str, language_code: str) -> Optional[Dict[str, Any]]:
        """Get cached transcript if available and not expired."""
        cache_key = self._get_cache_key(audio_path, model, language_code)
        cache_file = self.cache_dir / cache_key
        
        if not cache_file.exists():
            return None
            
        # Check TTL
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.days > self.ttl_days:
            cache_file.unlink()  # Remove expired cache
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None
    
    def set(self, audio_path: Path, model: str, language_code: str, transcript_data: Dict[str, Any]):
        """Cache transcript data."""
        cache_key = self._get_cache_key(audio_path, model, language_code)
        cache_file = self.cache_dir / cache_key
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")


class SarvamTranscriber:
    """Handles audio transcription using Sarvam API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.api_key = os.getenv('SARVAM_API_KEY')
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable is required")
            
        self.model = config.get('asr_model', 'saarika:v2.5')
        self.language_code = config.get('asr_language_code', 'ta-IN')
        self.batch_enabled = config.get('asr_batch_enabled', True)
        self.batch_max_files = config.get('asr_batch_max_files_per_job', 20)
        self.concurrency_limit = config.get('asr_concurrency_limit', 3)
        
        # Retry configuration
        retry_config = config.get('asr_retry', {})
        self.max_attempts = retry_config.get('max_attempts', 5)
        self.backoff_factor = retry_config.get('backoff_factor', 2)
        
        # Cache configuration
        cache_enabled = config.get('transcript_cache_enabled', True)
        cache_ttl = config.get('transcript_cache_ttl_days', 30)
        if cache_enabled:
            cache_dir = Path(config.get('temp_dir', './data/tmp')) / 'transcript_cache'
            self.cache = TranscriptCache(str(cache_dir), cache_ttl)
        else:
            self.cache = None
            
        # API endpoints (updated to correct Sarvam API endpoints)
        self.sync_endpoint = "https://api.sarvam.ai/speech-to-text"
        # Note: Batch endpoints are currently returning 404, using sync API as fallback
        self.batch_create_endpoint = "https://api.sarvam.ai/v1/batch_jobs"
        self.batch_upload_endpoint = "https://api.sarvam.ai/v1/batch_uploads"
        
        # Configure requests session with retry logic
        self.session = self._configure_session()
        
    def _configure_session(self) -> requests.Session:
        """Configure requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_attempts,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
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
    
    def _upload_file(self, audio_path: Path) -> str:
        """Upload audio file to Sarvam and return file URL."""
        headers = {"api-subscription-key": self.api_key}
        
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path.name, f, 'audio/wav')}
            response = self.session.post(
                self.batch_upload_endpoint,
                files=files,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
        result = response.json()
        return result.get('file_url') or result.get('url')
    
    def _create_batch_job(self, file_urls: List[str]) -> str:
        """Create a batch transcription job."""
        payload = {
            "model": self.model,
            "language_code": self.language_code,
            "files": file_urls,
            "with_timestamps": True,
            "with_diarization": True
        }
        
        headers = {"api-subscription-key": self.api_key}
        
        response = self.session.post(
            self.batch_create_endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result['id']
    
    def _poll_batch_job(self, job_id: str, max_wait_minutes: int = 60) -> Dict[str, Any]:
        """Poll batch job status until completion."""
        headers = {"api-subscription-key": self.api_key}
        
        for attempt in range(max_wait_minutes * 2):  # Poll every 30 seconds
            try:
                response = self.session.get(
                    f"{self.batch_create_endpoint}/{job_id}",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                status = response.json()
                
                if status['state'] in ("completed", "failed"):
                    return status
                    
                # Exponential backoff with jitter
                sleep_time = min(2 ** (attempt // 10), 30) + (attempt % 5)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.warning(f"Failed to poll job {job_id} (attempt {attempt}): {e}")
                time.sleep(5)
        
        raise TimeoutError(f"Batch job {job_id} did not complete within {max_wait_minutes} minutes")
    
    def _download_batch_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Download results from a completed batch job."""
        headers = {"api-subscription-key": self.api_key}
        
        response = self.session.get(
            f"{self.batch_create_endpoint}/{job_id}/results",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('results', [])
    
    def _transcribe_sync(self, audio_path: Path) -> SarvamTranscriptionResult:
        """Transcribe a single file using synchronous API (for short files)."""
        headers = {"api-subscription-key": self.api_key}
        
        # Check file size and basic validation
        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Audio file {audio_path.name} is empty")
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Audio file {audio_path.name} is too large ({file_size} bytes)")
        
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path.name, f, 'audio/wav')}
            data = {
                'model': self.model,
                'language_code': self.language_code
            }
            
            try:
                response = self.session.post(
                    self.sync_endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120
                )
                response.raise_for_status()
                
            except requests.exceptions.HTTPError as e:
                # Log detailed error information
                error_detail = "Unknown error"
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get('error', {}).get('message', str(error_response))
                except:
                    error_detail = e.response.text if e.response else str(e)
                
                logger.error(f"Sarvam API error for {audio_path.name}: {e.response.status_code} - {error_detail}")
                raise ValueError(f"Sarvam API error: {error_detail}")
            
        result = response.json()
        return self._parse_transcription_result(audio_path, result)
    
    def _parse_transcription_result(self, audio_path: Path, api_result: Dict[str, Any]) -> SarvamTranscriptionResult:
        """Parse Sarvam API result into our format."""
        metadata = self._extract_metadata_from_filename(audio_path.name)
        result = SarvamTranscriptionResult(audio_path, metadata)
        
        # Debug: Log the API response structure
        logger.info(f"API response for {audio_path.name}: {api_result}")
        
        # Parse segments from API response
        # Note: This structure may need adjustment based on actual Sarvam API response format
        segments = api_result.get('segments', [])
        
        # If no segments, try alternative response formats
        if not segments:
            # Try different possible response structures
            if 'text' in api_result:
                # Single text response
                text = api_result.get('text', '').strip()
                if text:
                    result.add_segment(0.0, 1.0, text, 0.8, None)
            elif 'transcript' in api_result:
                # Transcript field
                text = api_result.get('transcript', '').strip()
                if text:
                    result.add_segment(0.0, 1.0, text, 0.8, None)
            elif 'result' in api_result:
                # Nested result structure
                result_data = api_result.get('result', {})
                if isinstance(result_data, dict):
                    text = result_data.get('text', '').strip()
                    if text:
                        result.add_segment(0.0, 1.0, text, 0.8, None)
        
        for segment in segments:
            start = segment.get('start', 0.0)
            end = segment.get('end', start + 1.0)
            text = segment.get('text', '').strip()
            confidence = segment.get('confidence', 0.8)
            speaker = segment.get('speaker', None)
            
            if text:  # Only add non-empty segments
                result.add_segment(start, end, text, confidence, speaker)
        
        logger.info(f"Parsed {len(result.segments)} segments for {audio_path.name}")
        return result
    
    def transcribe(self, audio_path: Path) -> SarvamTranscriptionResult:
        """Transcribe a single audio file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(audio_path, self.model, self.language_code)
            if cached_result:
                logger.info(f"Using cached transcript for {audio_path.name}")
                result = SarvamTranscriptionResult(audio_path, cached_result.get('metadata', {}))
                for segment in cached_result.get('segments', []):
                    result.add_segment(
                        segment['start'],
                        segment['end'],
                        segment['text'],
                        segment.get('confidence', 0.8),
                        segment.get('speaker')
                    )
                return result
        
        # Determine duration (best-effort)
        duration = None
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
        except Exception as e:
            logger.warning(f"Could not determine audio duration for {audio_path.name}: {e}")
        
        # Choose path based on duration
        if duration is not None and duration <= 30:
            logger.info(f"File {audio_path.name} is {duration:.1f}s long, using sync API")
            return self._transcribe_sync(audio_path)
        
        # For long or unknown durations, prefer chunked sync transcription
        if duration is not None:
            logger.info(f"File {audio_path.name} is {duration:.1f}s long, using chunked sync transcription")
        else:
            logger.info(f"File {audio_path.name} duration unknown, using chunked sync transcription")
        try:
            return self._transcribe_with_splitting(audio_path)
        except Exception as split_error:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Audio splitting transcription failed for {audio_path.name}: {split_error}\n{tb}")
            # Final fallback: transcribe only first 30 seconds to ensure some output
            logger.warning(f"Falling back to first 30 seconds for {audio_path.name}")
            return self._transcribe_first_30_seconds(audio_path)
    
    def transcribe_batch(self, audio_paths: List[Path]) -> List[SarvamTranscriptionResult]:
        """Transcribe multiple audio files using batch API."""
        if not audio_paths:
            return []
        
        # Check cache for all files first
        uncached_paths = []
        cached_results = []
        
        for audio_path in audio_paths:
            if self.cache:
                cached_result = self.cache.get(audio_path, self.model, self.language_code)
                if cached_result:
                    logger.info(f"Using cached transcript for {audio_path.name}")
                    result = SarvamTranscriptionResult(audio_path, cached_result.get('metadata', {}))
                    for segment in cached_result.get('segments', []):
                        result.add_segment(
                            segment['start'],
                            segment['end'],
                            segment['text'],
                            segment.get('confidence', 0.8),
                            segment.get('speaker')
                        )
                    cached_results.append(result)
                    continue
            
            uncached_paths.append(audio_path)
        
        if not uncached_paths:
            return cached_results
        
        # Process uncached files in batches
        all_results = cached_results
        
        for i in range(0, len(uncached_paths), self.batch_max_files):
            batch_paths = uncached_paths[i:i + self.batch_max_files]
            batch_results = self._transcribe_batch(batch_paths)
            all_results.extend(batch_results)
        
        return all_results
    
    def _transcribe_batch_single(self, audio_path: Path) -> SarvamTranscriptionResult:
        """Transcribe a single file using batch API (for files longer than 30 seconds)."""
        logger.info(f"Using batch API for single file: {audio_path.name}")
        
        try:
            # Upload the file
            file_url = self._upload_file(audio_path)
            
            # Create batch job
            job_id = self._create_batch_job([file_url])
            
            # Poll for completion
            status = self._poll_batch_job(job_id)
            
            if status['state'] == 'completed':
                # Download results
                results = self._download_batch_results(job_id)
                
                # Find our file's result
                for result in results:
                    if result.get('file_name') == audio_path.name:
                        return self._parse_transcription_result(audio_path, result)
                
                raise ValueError(f"No result found for {audio_path.name} in batch response")
            else:
                raise ValueError(f"Batch job failed with state: {status['state']}")
                
        except Exception as e:
            logger.error(f"Batch transcription failed for {audio_path.name}: {e}")
            # Fallback to audio splitting if batch API is not available
            if "404" in str(e) or "Not Found" in str(e):
                logger.info(f"Batch API not available, falling back to audio splitting for {audio_path.name}")
                try:
                    return self._transcribe_with_splitting(audio_path)
                except Exception as split_error:
                    logger.error(f"Audio splitting also failed for {audio_path.name}: {split_error}")
                    # Final fallback: try to process the first 30 seconds only
                    logger.warning(f"Attempting to process only first 30 seconds of {audio_path.name}")
                    return self._transcribe_first_30_seconds(audio_path)
            else:
                raise
    
    def _transcribe_with_splitting(self, audio_path: Path) -> SarvamTranscriptionResult:
        """Transcribe a long audio file by splitting it into chunks."""
        try:
            from audio_splitter import AudioSplitter
            
            # Create temporary directory for chunks
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Split the audio file with optimized parameters for longer files
                # Use smaller chunks (20s) with overlap for better accuracy
                splitter = AudioSplitter(max_chunk_duration=20.0, overlap_duration=2.0)
                chunks = splitter.split_audio_file(audio_path, temp_path)
                
                if not chunks:
                    raise ValueError(f"No chunks created for {audio_path.name}")
                
                logger.info(f"Splitting {audio_path.name} ({len(chunks)} chunks)")
                
                # Transcribe each chunk with retry logic
                all_segments = []
                successful_chunks = 0
                
                for i, (chunk_path, start_time, end_time) in enumerate(chunks):
                    try:
                        logger.info(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk_path.name} ({start_time:.1f}s - {end_time:.1f}s)")
                        chunk_result = self._transcribe_sync(chunk_path)
                        
                        # Adjust segment timestamps to account for chunk offset
                        for segment in chunk_result.segments:
                            segment['start'] += start_time
                            segment['end'] += start_time
                            all_segments.append(segment)
                        
                        successful_chunks += 1
                        
                    except Exception as e:
                        import traceback as _tb
                        logger.warning(f"Failed to transcribe chunk {i+1}/{len(chunks)} {chunk_path.name}: {e}\n{_tb.format_exc()}")
                        continue
                
                if successful_chunks == 0:
                    raise ValueError(f"No chunks successfully transcribed for {audio_path.name}")
                
                # Create combined result
                metadata = self._extract_metadata_from_filename(audio_path.name)
                result = SarvamTranscriptionResult(audio_path, metadata)
                
                # Add all segments with adjusted timestamps
                for segment in all_segments:
                    result.add_segment(
                        segment['start'],
                        segment['end'],
                        segment['text'],
                        segment.get('confidence', 0.8),
                        segment.get('speaker')
                    )
                
                logger.info(f"Successfully transcribed {audio_path.name} using {successful_chunks}/{len(chunks)} chunks")
                return result
                
        except Exception as e:
            import traceback
            logger.error(f"Audio splitting transcription failed for {audio_path.name}: {e}\n{traceback.format_exc()}")
            raise
    
    def _transcribe_first_30_seconds(self, audio_path: Path) -> SarvamTranscriptionResult:
        """Transcribe only the first 30 seconds of a long audio file as a final fallback."""
        try:
            import tempfile
            from audio_splitter import AudioSplitter
            
            # Create a temporary chunk with only the first 30 seconds
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Determine original duration for clearer logging
                try:
                    import wave as _wave_for_len
                    with _wave_for_len.open(str(audio_path), 'rb') as _wf:
                        _frames = _wf.getnframes()
                        _rate = _wf.getframerate()
                        _orig_duration = _frames / float(_rate)
                except Exception:
                    _orig_duration = None

                # Create a chunk with first 30 seconds
                splitter = AudioSplitter(max_chunk_duration=30.0, overlap_duration=0.0)
                chunks = splitter.split_audio_file(audio_path, temp_path)
                
                if not chunks:
                    raise ValueError("Failed to create first 30-second chunk")
                
                # Take only the first chunk
                chunk_path, start_time, end_time = chunks[0]
                
                # Transcribe the first chunk
                chunk_result = self._transcribe_sync(chunk_path)
                
                # Create result with warning about partial transcription
                metadata = self._extract_metadata_from_filename(audio_path.name)
                result = SarvamTranscriptionResult(audio_path, metadata)
                
                # Add segments from first chunk
                for segment in chunk_result.segments:
                    result.add_segment(
                        segment['start'],
                        segment['end'],
                        segment['text'],
                        segment.get('confidence', 0.8),
                        segment.get('speaker')
                    )
                
                if _orig_duration is not None:
                    logger.warning(f"Transcribed only first 30 seconds of {audio_path.name} (original length { _orig_duration:.1f}s)")
                else:
                    logger.warning(f"Transcribed only first 30 seconds of {audio_path.name}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to transcribe first 30 seconds of {audio_path.name}: {e}")
            # Create empty result with error message
            metadata = self._extract_metadata_from_filename(audio_path.name)
            result = SarvamTranscriptionResult(audio_path, metadata)
            result.add_segment(0.0, 1.0, f"[TRANSCRIPTION FAILED: {str(e)}]", 0.0, None)
            return result
    
    def _transcribe_batch(self, audio_paths: List[Path]) -> List[SarvamTranscriptionResult]:
        """Internal method to transcribe a batch of files."""
        if not audio_paths:
            return []
        
        logger.info(f"Processing batch of {len(audio_paths)} files")
        
        # Upload files with concurrency limit
        file_urls = []
        with ThreadPoolExecutor(max_workers=self.concurrency_limit) as executor:
            future_to_path = {
                executor.submit(self._upload_file, path): path 
                for path in audio_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    file_url = future.result()
                    file_urls.append(file_url)
                    logger.info(f"Uploaded: {path.name}")
                except Exception as e:
                    logger.error(f"Failed to upload {path.name}: {e}")
                    raise
        
        if not file_urls:
            raise RuntimeError("No files were successfully uploaded")
        
        # Create batch job
        job_id = self._create_batch_job(file_urls)
        logger.info(f"Created batch job: {job_id}")
        
        # Poll for completion
        status = self._poll_batch_job(job_id)
        
        if status['state'] != 'completed':
            raise RuntimeError(f"Batch job failed: {status}")
        
        # Parse results
        results = []
        outputs = status.get('outputs', [])
        
        for i, output in enumerate(outputs):
            if i < len(audio_paths):
                audio_path = audio_paths[i]
                result = self._parse_transcription_result(audio_path, output)
                
                # Cache the result
                if self.cache:
                    cache_data = {
                        'metadata': result.metadata,
                        'segments': result.get_segments()
                    }
                    self.cache.set(audio_path, self.model, self.language_code, cache_data)
                
                results.append(result)
        
        logger.info(f"Batch transcription completed: {len(results)} results")
        return results


def main():
    """Standalone execution for testing."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe audio files using Sarvam')
    parser.add_argument('--config', default='../config.yaml', help='Configuration file path')
    parser.add_argument('--audio-file', required=True, help='Audio file to transcribe')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run transcription
    transcriber = SarvamTranscriber(config)
    audio_path = Path(args.audio_file)
    
    try:
        result = transcriber.transcribe(audio_path)
        print(f"Transcription complete for {audio_path.name}")
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
