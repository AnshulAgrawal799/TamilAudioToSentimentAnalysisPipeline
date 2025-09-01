# Tamil Audio to Sentiment Analysis Pipeline

A minimal viable prototype pipeline that processes Tamil MP3 audio files and outputs sentiment analysis results in JSON format.

**Recent Updates:** Fixed data schema inconsistencies, improved sentiment labeling, enhanced intent classification with Tamil patterns, **IMPROVED TRANSLATION QUALITY**, **SARVAM SPEECH-TO-TEXT INTEGRATION**

## Overview

This pipeline:

1. **Ingests** MP3 files from an `audio/` directory
2. **Transcribes** Tamil audio using Sarvam Speech-to-Text (Saarika model) or Whisper ASR (fallback)
3. **Cleans** ASR output to remove noise and artifacts
4. **Translates** to English using Google Cloud Translate API (with fallback)
5. **Analyzes** each segment for intent, sentiment, and emotion
6. **Aggregates** results into stop-level and day-level summaries
7. **Outputs** JSON files matching the corrected schemas

## ASR Provider Options

### Sarvam Speech-to-Text (Recommended)

- **Model**: Saarika v2.5 (optimized for Tamil)
- **Features**: Timestamps, speaker diarization, code-mix handling, language detection
- **Benefits**: Faster processing, better Tamil accuracy, managed infrastructure
- **Requirements**: SARVAM_API_KEY environment variable

### Whisper ASR (Fallback)

- **Model**: Medium (smaller than previous large-v3 for faster processing)
- **Features**: Local processing, offline capability
- **Use case**: Fallback when Sarvam is unavailable or for offline processing

## Translation Quality Improvements

### Problem

The previous pipeline produced poor English translations with:

- Gibberish and nonsense text
- Wrong/unrelated content
- Transliteration artifacts
- Truncated phrases
- Inconsistent quality

### Solutions Implemented

1. **Upgraded Whisper Model**: Changed from `small` to `large-v3` for better Tamil transcription
2. **Improved ASR Post-processing**: Added comprehensive text cleaning to remove noise
3. **Better Translation Service**: Integrated Google Cloud Translate API with fallback
4. **Quality Filtering**: Added confidence thresholds and segment filtering
5. **Enhanced Text Cleaning**: Removes repeated tokens, transliteration artifacts, and ASR noise

### Configuration

```yaml
# ASR Provider Settings
asr_provider: "sarvam" # "sarvam" or "whisper"
asr_model: "saarika:v2.5" # Sarvam model for Tamil
asr_language_code: "ta-IN" # Tamil language code
asr_batch_enabled: true # Use batch API for longer files
asr_concurrency_limit: 3 # Max concurrent uploads
transcript_cache_enabled: true # Cache transcripts to avoid re-processing

# Whisper Fallback Settings
whisper_model: "medium" # Fallback Whisper model
language: "ta" # Tamil language for Whisper

# Translation settings
translation:
  use_google_cloud: true # Use Google Cloud Translate API
  fallback_to_googletrans: true # Fallback option
  batch_size: 10
  retry_attempts: 3

# Quality thresholds
quality_thresholds:
  min_segment_confidence: 0.5
  min_translation_quality: 0.7
  max_segment_length: 200
```

### Setup for Better Quality

1. **Set up Sarvam API Key** (required for Sarvam provider):

   ```bash
   export SARVAM_API_KEY="your-sarvam-api-key"
   ```

2. **Install Google Cloud Translate** (recommended):

   ```bash
   pip install google-cloud-translate
   python setup_google_translate.py
   ```

3. **Test Integration**:

   ```bash
   python test_sarvam_integration.py
   ```

4. **Test Translation Quality**:

   ```bash
   python test_translation_quality.py
   ```

5. **Monitor Quality Metrics**:
   - Noise removal percentage
   - Text preservation rate
   - Translation availability
   - ASR provider performance

## Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for audio conversion)
- 4GB+ RAM for ASR processing

### Installation

1. **Install FFmpeg:**

   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Install Python dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python src/main.py --help
   ```

### Usage

1. **Place your MP3 files in the `audio/` directory**

2. **Run the complete pipeline:**

   ```bash
   # Using Sarvam (default)
   python src/main.py --config config.yaml

   # Using Whisper fallback
   python src/main.py --config config.yaml --provider whisper

   # Override provider from command line
   python src/main.py --config config.yaml --provider sarvam
   ```

3. **Expected outputs:**

   - `data/outputs/segments.json` - Per-segment JSON array
   - `data/outputs/aggregate_stop_<STOPID>_<DATE>.json` - Stop-level summary
   - `data/outputs/aggregate_day_<SELLERID>_<DATE>.json` - Day-level summary

4. **Example outputs:**
   - `data/outputs/segments.json`
   - `data/outputs/aggregate_stop_STOP45_2025-08-19.json`
   - `data/outputs/aggregate_day_S123_2025-08-19.json`

## Configuration

Edit `config.yaml` to customize:

- Audio directories
- ASR provider and model settings
- Field definitions and enumerations
- Sentiment analysis thresholds
- Placeholder values for missing metadata

```yaml
# Audio processing
audio_dir: "./audio"
output_dir: "./data/outputs"
temp_dir: "./data/tmp"

# ASR Provider Settings
asr_provider: "sarvam" # "sarvam" or "whisper"
asr_model: "saarika:v2.5" # Sarvam model for Tamil
asr_language_code: "ta-IN" # Tamil language code
asr_batch_enabled: true # Use batch API for longer files
asr_concurrency_limit: 3 # Max concurrent uploads
transcript_cache_enabled: true # Cache transcripts to avoid re-processing

# Whisper Fallback Settings
whisper_model: "medium" # Fallback Whisper model
language: "ta" # Tamil language for Whisper

# Field definitions
fields:
  textTamil: "Tamil text from transcription"
  textEnglish: "English translation text"
  speaker_roles: ["buyer", "seller", "other"]
  intent_categories:
    ["purchase", "inquiry", "complaint", "bargain", "greeting", "other"]
  sentiment_labels: ["positive", "neutral", "negative"]
  emotion_categories: ["happy", "neutral", "disappointed"]

# Sentiment analysis thresholds
sentiment_thresholds:
  positive: 0.1
  negative: -0.1
  neutral_range: [-0.1, 0.1]

# Placeholder values for missing metadata
placeholders:
  seller_id: "S123"
  stop_id: "STOP45"
  anchor_time: "2025-08-19T00:00:00Z"
  sales_at_stop:
    tomato: 36
    onion: 15
    potato: 10
```

## File Naming Convention

For automatic metadata extraction, use this filename format:

```
seller_<SELLERID>_STOP<STOPID>_<YYYYMMDD>_<HHMMSS>.mp3
```

Example: `seller_S123_STOP45_20250819_191110.mp3`

If filenames don't follow this pattern, placeholder values from config will be used.

## Pipeline Architecture

```
MP3 Files → WAV Conversion → [Sarvam/Whisper] ASR → NLU Analysis → Aggregation → JSON Outputs
```

### Components

- **`ingest_audio.py`** - MP3 to WAV conversion using FFmpeg
- **`sarvam_transcribe.py`** - Tamil audio transcription using Sarvam API
- **`transcribe.py`** - Tamil audio transcription using Whisper (fallback)
- **`transcriber_factory.py`** - Provider selection and unified interface
- **`analyze.py`** - Enhanced NLU analysis with Tamil + English patterns
- **`aggregate.py`** - Generate stop/day level summaries
- **`main.py`** - Orchestrates the complete pipeline

## Output Schemas

### Per-Segment Output

```json
{
  "seller_id": "S123",
  "stop_id": "STOP45",
  "segment_id": "SEG001",
  "audio_file_id": "tmp/Audio1.wav",
  "timestamp": "2025-08-19T19:11:12Z",
  "start_ms": 5220,
  "end_ms": 7240,
  "duration_ms": 2020,
  "speaker_role": "buyer",
  "textTamil": "தக்காளி எவ்வளவு விலை",
  "textEnglish": "What is the price of tomatoes?",
  "intent": "inquiry",
  "sentiment_score": 0.2,
  "sentiment_label": "positive",
  "emotion": "neutral",
  "confidence": 0.85,
  "translation_confidence": 0.8,
  "role_confidence": 0.6,
  "churn_risk": "low",
  "asr_confidence": 0.9,
  "needs_human_review": false
}
```

**New Audio Mapping Fields:**

- `audio_file_id`: Path to the source audio file (e.g., "tmp/Audio1.wav")
- `start_ms`: Segment start time in milliseconds from audio file beginning
- `end_ms`: Segment end time in milliseconds from audio file beginning
- `duration_ms`: Segment duration in milliseconds (end_ms - start_ms)

These fields enable precise audio replay and debugging by providing deterministic pointers back to the raw audio source.

**New Quality Review Flag:**

- `needs_human_review` (boolean, default `false`)
  - Set to `true` when any of the following are met:
    - `asr_confidence < 0.65`
    - `translation_confidence < 0.65`
    - `sentiment_label == "negative"` AND `churn_risk == "high"` AND `role_confidence >= 0.5`

**Key Fixes Applied:**

- ✅ Added `textTamil` and `textEnglish` fields
- ✅ Fixed sentiment labeling: `score >= 0.1` → positive, `score <= -0.1` → negative
- ✅ Enhanced intent classification with Tamil patterns
- ✅ Proper speaker role enumeration: `["buyer", "seller", "other"]`

### Aggregated Stop Output

```json
{
  "seller_id": "S123",
  "stop_id": "STOP45",
  "date": "2025-08-19",
  "n_segments": 15,
  "n_calls": 5,
  "avg_sentiment_score": -0.01,
  "sentiment_distribution": { "positive": 3, "neutral": 9, "negative": 3 },
  "dominant_emotion": "neutral",
  "top_intents": [
    "other",
    "purchase",
    "bargain",
    "inquiry",
    "complaint",
    "greeting"
  ],
  "intent_distribution": {
    "other": 6,
    "purchase": 3,
    "bargain": 3,
    "inquiry": 1,
    "complaint": 1,
    "greeting": 1
  },
  "sales_at_stop": { "tomato": 36, "onion": 15, "potato": 10 },
  "inventory_after_sale": { "tomato": 22, "onion": 5, "potato": 8 },
  "review_statistics": {
    "total_segments": 15,
    "needs_review": 3,
    "review_percentage": 20.0,
    "review_reasons": {
      "asr_low_confidence": 1,
      "translation_low_confidence": 1,
      "high_risk_negative": 1
    }
  }
}
```

**New Review Statistics Field:**

- `review_statistics`: Aggregated human review metrics
  - `total_segments`: Total number of segments analyzed
  - `needs_review`: Number of segments flagged for human review
  - `review_percentage`: Percentage of segments needing review (0.0-100.0)
  - `review_reasons`: Breakdown of why segments need review:
    - `asr_low_confidence`: Segments with ASR confidence < 0.65
    - `translation_low_confidence`: Segments with translation confidence < 0.65
    - `high_risk_negative`: High-risk negative segments (negative sentiment + high churn + role confidence ≥ 0.5)

## Enhanced NLU Analysis

### Intent Classification

The pipeline now includes Tamil-specific patterns for better intent recognition:

- **Purchase**: வாங்கு, வாங்குவோம், எடுத்துக்கொள், தேவை
- **Inquiry**: எவ்வளவு, விலை என்ன, உள்ளதா, செலவு
- **Complaint**: மோசம், பிரச்சினை, தவறு, புகார்
- **Bargain**: தள்ளுபடி, குறைவு, பேரம், மலிவு
- **Greeting**: வணக்கம், காலை வணக்கம், நமஸ்காரம்

### Sentiment Analysis

Fixed sentiment labeling with proper thresholds:

- **Positive**: score ≥ 0.1
- **Negative**: score ≤ -0.1
- **Neutral**: -0.1 < score < 0.1

### Speaker Role Detection

Enhanced pattern matching for Tamil text to identify:

- **Seller**: விலை, கிலோ, கொடுக்கலாம், தரமான
- **Buyer**: வாங்குவோம், கொடுங்கள், எவ்வளவு, தேவை
- **Other**: Unidentified speakers

## Testing Individual Components

### Test Audio Ingestion

```bash
python src/ingest_audio.py --config config.yaml
```

### Test Transcription

```bash
python src/transcribe.py --config config.yaml --audio-file data/tmp/example.wav
```

### Test NLU Analysis

```bash
python src/analyze.py --config config.yaml --text "நான் தக்காளி வாங்க விரும்புகிறேன்"
```

### Test Aggregation

```bash
python src/aggregate.py --config config.yaml
```

## Performance Notes

### Sarvam Speech-to-Text

- **Processing time**: ~0.5-1x audio duration (much faster than local Whisper)
- **Memory usage**: Minimal (no local model loading)
- **Network dependency**: Requires internet connection
- **Batch processing**: Supports concurrent file processing

### Whisper ASR (Fallback)

- **Whisper medium model**: ~2GB RAM, ~3x real-time processing
- **Processing time**: ~2-3x audio duration
- **Memory usage**: Scales with audio length
- **First run**: Model download (~1.5GB for medium model)
- **Offline capability**: Works without internet connection

## Troubleshooting

### Common Issues

1. **Sarvam API Key not set**

   - Set environment variable: `export SARVAM_API_KEY="sk_ttjsa620_dbdCrZsL8KDdf0qs0JtiEOJr"`
   - Verify with: `echo $SARVAM_API_KEY`

2. **Sarvam API errors**

   - Check API key validity
   - Verify network connectivity
   - Check rate limits and quotas
   - Use Whisper fallback: `--provider whisper`

3. **FFmpeg not found**

   - Install FFmpeg and ensure it's in your PATH
   - Verify with: `ffmpeg -version`

4. **Out of memory (Whisper)**

   - Use smaller Whisper model (tiny, base, small)
   - Process shorter audio files
   - Close other applications

5. **No segments generated**

   - Check audio file quality
   - Verify Tamil language detection
   - Check minimum segment word count in config

6. **Import errors**

   - Ensure all dependencies are installed
   - Check Python version (3.8+ required)
   - Verify virtual environment activation

7. **Sentiment labels incorrect**
   - Check sentiment thresholds in config.yaml
   - Verify sentiment scoring logic in analyze.py
   - Ensure proper thresholding: positive ≥ 0.1, negative ≤ -0.1

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH=src
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from main import main
main()
"
```

## Limitations

This is a prototype with the following limitations:

- **Rule-based NLU**: No machine learning models for intent/sentiment (IMPROVED with Tamil patterns)
- **Basic segmentation**: Segments by ASR blocks only
- **Placeholder data**: Sales/inventory data uses config values
- **Batch processing**: No real-time streaming capability
- **Network dependency**: Sarvam provider requires internet connection
- **API costs**: Sarvam usage incurs per-minute billing

## Future Enhancements

- ML-based intent/sentiment/emotion models
- Speaker diarization for better segmentation (partially available via Sarvam)
- Real-time processing capability
- Integration with external metadata sources
- Web UI for manual review and correction
- **NEW:** Google Translate API integration for textEnglish field
- **NEW:** Fine-tuned Tamil domain vocabulary patterns
- **NEW:** Sarvam Speech-to-Text integration for faster processing

## Schema Validation

To validate your outputs against the corrected schemas:

1. **Check field names**: Ensure `textTamil` and `textEnglish` are present
2. **Verify sentiment labels**: Negative scores should be labeled "negative"
3. **Validate intents**: Should show meaningful distribution beyond "other"
4. **Confirm speaker roles**: Should use values from `["buyer", "seller", "other"]`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for prototype purposes.
