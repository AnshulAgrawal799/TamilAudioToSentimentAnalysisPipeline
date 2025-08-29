#!/usr/bin/env python3
"""
Main pipeline script for Tamil audio to sentiment analysis.
Runs the complete workflow: audio ingestion → transcription → analysis → aggregation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from ingest_audio import AudioIngester
from transcribe import Transcriber
from analyze import NLUAnalyzer
from aggregate import Aggregator

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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args if provided
    if args.audio_dir:
        config['audio_dir'] = args.audio_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    logger.info("Starting Tamil Audio to Sentiment Analysis Pipeline")
    logger.info(f"Audio directory: {config['audio_dir']}")
    logger.info(f"Output directory: {config['output_dir']}")
    
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
        transcriber = Transcriber(config)
        transcript_results = []
        for wav_file in wav_files:
            try:
                result = transcriber.transcribe(wav_file)
                transcript_results.append(result)
                logger.info(f"Transcribed: {wav_file.name}")
            except Exception as e:
                logger.error(f"Failed to transcribe {wav_file}: {e}")
        
        if not transcript_results:
            logger.error("No transcriptions completed successfully")
            sys.exit(1)
        
        # Step 3: NLU Analysis
        logger.info("Step 3: Running NLU analysis...")
        analyzer = NLUAnalyzer(config)
        analyzed_segments = []
        for result in transcript_results:
            segments = analyzer.analyze(result)
            analyzed_segments.extend(segments)
        
        logger.info(f"Analyzed {len(analyzed_segments)} segments")
        
        # Step 4: Generate outputs
        logger.info("Step 4: Generating output files...")
        
        # Sort segments by audio_file_id and timestamp for proper ordering
        analyzed_segments.sort(key=lambda x: (x.audio_file_id, x.start_ms))
        
        # Write segments to JSON with proper ordering
        segments_output = Path(config['output_dir']) / "segments.json"
        with open(segments_output, 'w', encoding='utf-8') as f:
            import json
            segments_data = [segment.to_dict() for segment in analyzed_segments]
            # Deterministic post-processing: map products to product_intents
            _annotate_product_intents(segments_data, config.get('PRODUCT_INTENT_MAP', {}))
            json.dump(segments_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {len(segments_data)} segments to: {segments_output}")
        
        # Also write segments grouped by audio file for better organization
        audio_grouped_segments = {}
        for segment in analyzed_segments:
            audio_id = segment.audio_file_id
            if audio_id not in audio_grouped_segments:
                audio_grouped_segments[audio_id] = []
            audio_grouped_segments[audio_id].append(segment.to_dict())
        
        # Write audio-grouped segments
        for audio_id, segments in audio_grouped_segments.items():
            audio_output = Path(config['output_dir']) / f"segments_{audio_id.replace('.wav', '')}.json"
            with open(audio_output, 'w', encoding='utf-8') as f:
                # Apply the same deterministic post-processing per file
                _annotate_product_intents(segments, config.get('PRODUCT_INTENT_MAP', {}))
                json.dump(segments, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote {len(segments)} segments for {audio_id} to: {audio_output}")
        
        # Generate aggregations
        aggregator = Aggregator(config)
        
        # Stop-level aggregation
        stop_aggregates = aggregator.aggregate_by_stop(analyzed_segments)
        for stop_id, data in stop_aggregates.items():
            output_file = Path(config['output_dir']) / f"aggregate_stop_{stop_id}_{data['date']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote stop aggregate: {output_file}")
        
        # Day-level aggregation
        day_aggregates = aggregator.aggregate_by_day(analyzed_segments)
        for seller_id, data in day_aggregates.items():
            output_file = Path(config['output_dir']) / f"aggregate_day_{seller_id}_{data['date']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote day aggregate: {output_file}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
