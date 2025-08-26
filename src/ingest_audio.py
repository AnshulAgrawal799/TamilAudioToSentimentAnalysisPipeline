#!/usr/bin/env python3
"""
Audio ingestion module for converting MP3 files to WAV format.
Handles file discovery and audio format conversion with optimizations.
"""

import subprocess
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import os

logger = logging.getLogger(__name__)


class AudioIngester:
    """Handles audio file ingestion and conversion with optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.audio_dir = Path(config['audio_dir'])
        self.temp_dir = Path(config['temp_dir'])
        
        # Audio processing configuration
        self.audio_config = config.get('audio_processing', {})
        self.convert_mp3_to_wav = self.audio_config.get('convert_mp3_to_wav', True)
        self.wav_sample_rate = self.audio_config.get('wav_sample_rate', 16000)
        self.wav_channels = self.audio_config.get('wav_channels', 1)
        self.out_dir = Path(self.audio_config.get('out_dir', self.temp_dir))
        self.delete_mp3_after_success = self.audio_config.get('delete_mp3_after_success', False)
        self.workers = self.audio_config.get('workers')
        self.overwrite_existing_wav = self.audio_config.get('overwrite_existing_wav', False)
        
        # Set default workers if not specified
        if self.workers is None:
            self.workers = max(1, mp.cpu_count() // 2)
        
        logger.info(f"Audio processing config: convert={self.convert_mp3_to_wav}, "
                   f"workers={self.workers}, sample_rate={self.wav_sample_rate}, "
                   f"channels={self.wav_channels}")
        
    def discover_audio_files(self) -> List[Path]:
        """Discover all MP3 files in the audio directory."""
        mp3_files = list(self.audio_dir.glob("*.mp3"))
        logger.info(f"Found {len(mp3_files)} MP3 files in {self.audio_dir}")
        return mp3_files
    
    def should_convert_file(self, mp3_path: Path) -> bool:
        """Check if file should be converted based on config and existing files."""
        if not self.convert_mp3_to_wav:
            return False
            
        wav_path = self.out_dir / f"{mp3_path.stem}.wav"
        
        # Skip if WAV exists and we're not forcing overwrite
        if wav_path.exists() and not self.overwrite_existing_wav:
            logger.debug(f"Skipping {mp3_path.name} - WAV already exists")
            return False
            
        return True
    
    def convert_mp3_to_wav_single(self, mp3_path: Path) -> Optional[Path]:
        """Convert single MP3 file to WAV format using FFmpeg with atomic writes."""
        wav_path = self.out_dir / f"{mp3_path.stem}.wav"
        
        # Ensure output directory exists
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary file for atomic write
        temp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.wav', 
                dir=self.out_dir, 
                delete=False
            ) as temp_file:
                temp_wav_path = Path(temp_file.name)
            
            # FFmpeg command for conversion
            cmd = [
                "ffmpeg", "-y",  # Overwrite output files
                "-i", str(mp3_path),  # Input file
                "-ar", str(self.wav_sample_rate),  # Sample rate from config
                "-ac", str(self.wav_channels),     # Channels from config
                "-vn",           # No video
                "-f", "wav",     # Output format: WAV
                str(temp_wav_path)    # Output to temp file
            ]
            
            logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Atomic move from temp to final location
            shutil.move(str(temp_wav_path), str(wav_path))
            
            logger.info(f"Converted {mp3_path.name} → {wav_path.name}")
            
            # Optionally delete MP3 after successful conversion
            if self.delete_mp3_after_success:
                try:
                    mp3_path.unlink()
                    logger.debug(f"Deleted source MP3: {mp3_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {mp3_path.name}: {e}")
            
            return wav_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed for {mp3_path}: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg first.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error converting {mp3_path}: {e}")
            return None
        finally:
            # Clean up temp file if it still exists
            if temp_wav_path and temp_wav_path.exists():
                try:
                    temp_wav_path.unlink()
                except Exception:
                    pass
    
    def convert_mp3_to_wav_parallel(self, mp3_files: List[Path]) -> List[Path]:
        """Convert multiple MP3 files in parallel."""
        if not mp3_files:
            return []
        
        # Filter files that need conversion
        files_to_convert = [f for f in mp3_files if self.should_convert_file(f)]
        
        if not files_to_convert:
            logger.info("No files need conversion - all WAVs already exist")
            # Return existing WAV files
            return [self.out_dir / f"{f.stem}.wav" for f in mp3_files 
                   if (self.out_dir / f"{f.stem}.wav").exists()]
        
        logger.info(f"Converting {len(files_to_convert)} files using {self.workers} workers")
        
        # Use multiprocessing for parallel conversion
        with mp.Pool(processes=self.workers) as pool:
            results = pool.map(self.convert_mp3_to_wav_single, files_to_convert)
        
        # Filter out None results (failed conversions)
        converted_files = [r for r in results if r is not None]
        
        # Add existing WAV files that weren't converted
        existing_wavs = []
        for mp3_file in mp3_files:
            wav_path = self.out_dir / f"{mp3_file.stem}.wav"
            if wav_path.exists() and wav_path not in converted_files:
                existing_wavs.append(wav_path)
        
        all_wav_files = converted_files + existing_wavs
        logger.info(f"Successfully converted {len(converted_files)} files, "
                   f"found {len(existing_wavs)} existing WAVs")
        
        return all_wav_files
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract metadata from filename if it follows the convention."""
        # Expected format: seller_<SELLERID>_STOP<STOPID>_<YYYYMMDD>_<HHMMSS>.mp3
        parts = filename.replace('.mp3', '').split('_')
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
    
    def process(self) -> List[Path]:
        """Main processing method: discover and convert all MP3 files."""
        mp3_files = self.discover_audio_files()
        
        if not mp3_files:
            logger.warning(f"No MP3 files found in {self.audio_dir}")
            return []
        
        if not self.convert_mp3_to_wav:
            logger.info("MP3 to WAV conversion disabled - looking for existing WAV files")
            wav_files = [self.out_dir / f"{f.stem}.wav" for f in mp3_files 
                        if (self.out_dir / f"{f.stem}.wav").exists()]
            logger.info(f"Found {len(wav_files)} existing WAV files")
            return wav_files
        
        # Process files in parallel
        wav_files = self.convert_mp3_to_wav_parallel(mp3_files)
        
        logger.info(f"Processing complete: {len(wav_files)} WAV files available")
        return wav_files


def main():
    """Standalone execution for testing."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MP3 files to WAV format')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--audio-dir', help='Override audio directory')
    parser.add_argument('--temp-dir', help='Override temp directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config if specified
    if args.audio_dir:
        config['audio_dir'] = args.audio_dir
    if args.temp_dir:
        config['temp_dir'] = args.temp_dir
    
    # Run conversion
    ingester = AudioIngester(config)
    converted = ingester.process()
    
    print(f"Conversion complete. {len(converted)} files available.")


if __name__ == "__main__":
    main()
