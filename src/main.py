
"""
Main pipeline script for Tamil audio to sentiment analysis.
Runs the complete workflow: audio ingestion → transcription → analysis → aggregation.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
from datetime import timedelta
from dateutil import parser as dateparser

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set API key directly for testing (remove this in production)
if not os.getenv('SARVAM_API_KEY'):
    os.environ['SARVAM_API_KEY'] = 'sk_ttjsa620_dbdCrZsL8KDdf0qs0JtiEOJr'

from ingest_audio import AudioIngester
from transcriber_factory import UnifiedTranscriber
from analyze import NLUAnalyzer
from aggregate import Aggregator
# Ensure project root is on sys.path to import db_mysql when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from db_mysql import (
    get_connection as db_get_connection,
    insert_segment as db_insert_segment,
    get_audio_file_id_by_filename as db_get_audio_file_id_by_filename,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def setup_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories if they don't exist."""
    dirs = [
        config['temp_dir'],
        config['output_dir']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


def _normalize_products_field(raw_products) -> list:
    """Normalize the products field to a list of lowercase, deduped tokens.

    Accepts list, comma-separated string, empty string, or None.
    Returns a list with first-seen order preserved.
    """
    tokens = []
    if raw_products is None:
        return tokens
    if isinstance(raw_products, list):
        candidate_tokens = raw_products
    elif isinstance(raw_products, str):
        # Allow comma-separated strings
        candidate_tokens = raw_products.split(',') if raw_products else []
    else:
        # Unsupported type → treat as no products
        candidate_tokens = []

    seen = set()
    for tok in candidate_tokens:
        if tok is None:
            continue
        norm = str(tok).strip().lower()
        if not norm:
            continue
        if norm not in seen:
            seen.add(norm)
            tokens.append(norm)
    return tokens


def _annotate_product_intents(segments_data: list, product_intent_map: Dict[str, str]) -> None:
    """Attach product_intents to segments in-place based on PRODUCT_INTENT_MAP.

    - Case-insensitive lookup
    - Only add product_intents key if at least one product maps
    - Do not modify segments with no matches
    """
    if not segments_data or not isinstance(product_intent_map, dict):
        return

    # Build a case-insensitive map
    ci_map = {str(k).strip().lower(): v for k, v in product_intent_map.items() if k is not None}

    for seg in segments_data:
        raw_products = seg.get('products')
        norm_products = _normalize_products_field(raw_products)
        if not norm_products:
            continue

        mapped: Dict[str, str] = {}
        for p in norm_products:
            if p in ci_map:
                mapped[p] = ci_map[p]

        if mapped:
            seg['product_intents'] = mapped


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Tamil Audio to Sentiment Analysis Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--audio-dir', help='Override audio directory from config')
    parser.add_argument('--output-dir', help='Override output directory from config')
    parser.add_argument('--skip-audio-conversion', action='store_true', 
                       help='Skip MP3 to WAV conversion (assume WAV files exist)')
    parser.add_argument('--provider', choices=['sarvam', 'whisper'], 
                       help='Override ASR provider from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args if provided
    if args.audio_dir:
        config['audio_dir'] = args.audio_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.provider:
        config['asr_provider'] = args.provider
    
    logger.info("Starting Tamil Audio to Sentiment Analysis Pipeline")
    logger.info(f"Audio directory: {config['audio_dir']}")
    logger.info(f"Output directory: {config['output_dir']}")
    logger.info(f"ASR Provider: {config.get('asr_provider', 'sarvam')}")
    
    # Log audio processing configuration
    audio_config = config.get('audio_processing', {})
    if audio_config:
        logger.info("Audio processing configuration:")
        logger.info(f"  Convert MP3 to WAV: {audio_config.get('convert_mp3_to_wav', True)}")
        logger.info(f"  Sample rate: {audio_config.get('wav_sample_rate', 16000)} Hz")
        logger.info(f"  Channels: {audio_config.get('wav_channels', 1)}")
        logger.info(f"  Workers: {audio_config.get('workers', 'auto')}")
        logger.info(f"  Overwrite existing: {audio_config.get('overwrite_existing_wav', False)}")
    
    try:
        # Setup directories
        setup_directories(config)
        
        # Step 1: Audio Ingestion (MP3 → WAV)
        if not args.skip_audio_conversion:
            logger.info("Step 1: Processing audio files...")
            ingester = AudioIngester(config)
            wav_files = ingester.process()
            logger.info(f"Audio processing complete: {len(wav_files)} WAV files available")
        else:
            logger.info("Skipping audio conversion, using existing WAV files")
            wav_files = list(Path(config['temp_dir']).glob("*.wav"))
            logger.info(f"Found {len(wav_files)} existing WAV files")
        
        if not wav_files:
            logger.error("No WAV files found for processing")
            sys.exit(1)
        
        # Step 2: Transcription
        logger.info("Step 2: Transcribing audio files...")
        transcriber = UnifiedTranscriber(config)
        
        # Use individual transcription for now since batch API endpoints are returning 404
        logger.info("Using individual transcription with Sarvam")
        transcript_results = []
        failed_files = []
        
        for wav_file in wav_files:
            try:
                result = transcriber.transcribe(wav_file)
                # Resolve DB audio_file.id from filename and attach to metadata for downstream use
                try:
                    resolved_id = db_get_audio_file_id_by_filename(wav_file.name)
                    if resolved_id is not None:
                        if not isinstance(result.metadata, dict):
                            result.metadata = {}
                        result.metadata['audio_file_id'] = int(resolved_id)
                except Exception as _e:
                    logger.warning(f"Could not resolve audio_file_id for {wav_file.name}: {_e}")
                transcript_results.append(result)
                logger.info(f"Transcribed: {wav_file.name} ({result.provider}/{result.model_used})")
            except Exception as e:
                logger.error(f"Failed to transcribe {wav_file}: {e}")
                failed_files.append(wav_file.name)
                continue
        
        if failed_files:
            logger.warning(f"Failed to transcribe {len(failed_files)} files: {failed_files}")
            logger.info(f"Continuing with {len(transcript_results)} successfully transcribed files")
        
        if not transcript_results:
            logger.error("No transcriptions completed successfully")
            sys.exit(1)
        
        # Step 3: NLU Analysis with utterance-level segmentation
        logger.info("Step 3: Running NLU analysis with utterance-level segmentation...")
        analyzer = NLUAnalyzer(config)
        analyzed_segments = []
        # Create a deterministic run id for this pipeline execution
        from datetime import datetime as _dt
        import uuid as _uuid
        pipeline_run_id = f"run-{_dt.now().strftime('%Y%m%d%H%M%S')}-{str(_uuid.uuid4())[:6]}"
        
        for result in transcript_results:
            # Process each transcription result and create utterance-level segments
            segments = analyzer.analyze(result)
            # Inject run-level and provenance metadata
            for seg in segments:
                try:
                    seg.pipeline_run_id = pipeline_run_id
                    seg.source_provider = getattr(result, 'provider', None)
                    seg.source_model = getattr(result, 'model_used', None)
                except Exception:
                    pass
            
            # Add utterance indices and ensure proper timing
            # Normalize and align timings per audio
            # Sort by start_ms to compute monotonic, non-overlapping boundaries
            segments.sort(key=lambda s: (s.start_ms, s.end_ms))
            # Determine true audio duration in milliseconds for clamping
            audio_duration_ms = None
            try:
                import wave as _wave_len
                with _wave_len.open(str(result.audio_file), 'rb') as _wf_len:
                    _frames = _wf_len.getnframes()
                    _rate = _wf_len.getframerate()
                    audio_duration_ms = int((_frames / float(_rate)) * 1000.0)
            except Exception:
                audio_duration_ms = None

            last_end = 0
            min_gap_ms = 20
            for i, segment in enumerate(segments):
                # Ensure indices stable after sort
                segment.utterance_index = i
                # Clamp to non-negative
                if segment.start_ms is None:
                    segment.start_ms = 0
                if segment.end_ms is None:
                    segment.end_ms = max(segment.start_ms + 1, last_end + 1)
                segment.start_ms = max(0, int(segment.start_ms))
                segment.end_ms = max(int(segment.end_ms), segment.start_ms + 1)
                # Enforce monotonic non-overlap
                if segment.start_ms < last_end + min_gap_ms:
                    segment.start_ms = last_end + min_gap_ms
                if segment.end_ms <= segment.start_ms:
                    segment.end_ms = segment.start_ms + max(200, min_gap_ms)
                # Clamp to audio duration if known
                if audio_duration_ms is not None:
                    if segment.end_ms > audio_duration_ms:
                        segment.end_ms = audio_duration_ms
                    if segment.start_ms >= audio_duration_ms:
                        # Move start just before end to keep minimal positive duration
                        segment.start_ms = max(0, audio_duration_ms - max(200, min_gap_ms))
                # Update duration
                segment.duration_ms = segment.end_ms - segment.start_ms
                last_end = segment.end_ms
                # Flag placeholder timings
                if segment.start_ms == 0 and segment.end_ms in (1, 1000):
                    logger.warning(f"Segment {segment.segment_id} has placeholder timing - marking for human review")
                    segment.needs_human_review = True
                    segment.asr_confidence = min(segment.asr_confidence or 0.5, 0.5)
                # Generate proper ISO timestamp
                try:
                    base_time = dateparser.isoparse(result.metadata.get('recording_start', '2025-08-19T00:00:00Z'))
                    segment_start_time = base_time + timedelta(milliseconds=segment.start_ms)
                    segment.timestamp = segment_start_time.isoformat().replace('+00:00', 'Z')
                except Exception as e:
                    logger.warning(f"Failed to generate timestamp for segment {segment.segment_id}: {e}")
                    segment.needs_human_review = True
            
            analyzed_segments.extend(segments)
        
        logger.info(f"Analyzed {len(analyzed_segments)} utterance-level segments")
        
        # Step 4: Persist segments to database (no local JSON writes)
        logger.info("Step 4: Persisting segments to database...")

        # Sort segments by audio_file_id and utterance_index for proper ordering
        analyzed_segments.sort(key=lambda x: (x.audio_file_id, x.utterance_index))

        # Prepare segment dicts and annotate product intents deterministically
        segments_data = [segment.to_dict() for segment in analyzed_segments]
        _annotate_product_intents(segments_data, config.get('PRODUCT_INTENT_MAP', {}))

        # Insert into DB using a single connection
        conn = None
        inserted = 0
        try:
            conn = db_get_connection()
            for seg in segments_data:
                # Validate audio_file_id presence and type (FK constraint)
                afid = seg.get('audio_file_id')
                try:
                    _ = int(afid)
                except Exception:
                    logger.warning(f"Skipping segment_id={seg.get('segment_id')} due to invalid audio_file_id={afid}")
                    continue
                try:
                    db_insert_segment(seg, conn=conn)
                    inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert segment_id={seg.get('segment_id')} audio_file_id={seg.get('audio_file_id')}: {e}")
            logger.info(f"Inserted {inserted}/{len(segments_data)} segments into DB")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        
        # Generate aggregations
        aggregator = Aggregator(config)
        
        # Stop-level aggregation
        stop_aggregates = aggregator.aggregate_by_stop(analyzed_segments)
        for stop_id, data in stop_aggregates.items():
            output_file = Path(config['output_dir']) / f"aggregate_stop_{stop_id}_{data['date']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                import json
                # Add file-level metadata envelope
                from datetime import datetime as _dt
                audio_sample_rate = config.get('audio_processing', {}).get('wav_sample_rate', 16000)
                total_audio_ms = 0
                try:
                    for af in (data.get('audio_files') or {}).values():
                        total_audio_ms += int(af.get('total_duration_ms', 0) or 0)
                except Exception:
                    total_audio_ms = 0
                envelope = {
                    'schema_version': '1.1.0',
                    'pipeline_run_id': pipeline_run_id,
                    'processed_at': _dt.utcnow().isoformat() + 'Z',
                    'source_uri': str(Path(config['audio_dir']).resolve()),
                    'audio_duration_ms': total_audio_ms,
                    'audio_sample_rate': audio_sample_rate,
                    'data': data,
                }
                json.dump(envelope, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote stop aggregate: {output_file}")
        
        # Day-level aggregation
        day_aggregates = aggregator.aggregate_by_day(analyzed_segments)
        for seller_id, data in day_aggregates.items():
            output_file = Path(config['output_dir']) / f"aggregate_day_{seller_id}_{data['date']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                import json
                # Add file-level metadata envelope
                from datetime import datetime as _dt
                audio_sample_rate = config.get('audio_processing', {}).get('wav_sample_rate', 16000)
                total_audio_ms = 0
                try:
                    for af in (data.get('audio_files') or {}).values():
                        total_audio_ms += int(af.get('total_duration_ms', 0) or 0)
                except Exception:
                    total_audio_ms = 0
                envelope = {
                    'schema_version': '1.1.0',
                    'pipeline_run_id': pipeline_run_id,
                    'processed_at': _dt.utcnow().isoformat() + 'Z',
                    'source_uri': str(Path(config['audio_dir']).resolve()),
                    'audio_duration_ms': total_audio_ms,
                    'audio_sample_rate': audio_sample_rate,
                    'data': data,
                }
                json.dump(envelope, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote day aggregate: {output_file}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
