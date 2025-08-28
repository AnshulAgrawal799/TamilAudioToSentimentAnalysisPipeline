# Tamil Audio to Sentiment Analysis Pipeline

A minimal viable prototype pipeline that processes Tamil MP3 audio files and outputs sentiment analysis results in JSON format.

**Recent Updates:** Fixed data schema inconsistencies, improved sentiment labeling, enhanced intent classification with Tamil patterns, **IMPROVED TRANSLATION QUALITY**

## Overview

This pipeline:
1. **Ingests** MP3 files from an `audio/` directory
2. **Transcribes** Tamil audio using Whisper ASR (upgraded to large-v3 model)
3. **Cleans** ASR output to remove noise and artifacts
4. **Translates** to English using Google Cloud Translate API (with fallback)
5. **Analyzes** each segment for intent, sentiment, and emotion
6. **Aggregates** results into stop-level and day-level summaries
7. **Outputs** JSON files matching the corrected schemas

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
# ASR settings (improved)
asr_model: "large-v3"  # Better Tamil transcription
language: "ta"

# Translation settings
translation:
  use_google_cloud: true  # Use Google Cloud Translate API
  fallback_to_googletrans: true  # Fallback option
  batch_size: 10
  retry_attempts: 3

# Quality thresholds
quality_thresholds:
  min_segment_confidence: 0.5
  min_translation_quality: 0.7
  max_segment_length: 200
```

### Setup for Better Quality
1. **Install Google Cloud Translate** (recommended):
   ```bash
   pip install google-cloud-translate
   python setup_google_translate.py
   ```

2. **Test Translation Quality**:
   ```bash
   python test_translation_quality.py
   ```

3. **Monitor Quality Metrics**:
   - Noise removal percentage
   - Text preservation rate
   - Translation availability

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
   python src/main.py --config config.yaml
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
- ASR model size
- Field definitions and enumerations
- Sentiment analysis thresholds
- Placeholder values for missing metadata

```yaml
# Audio processing
audio_dir: "./audio"
output_dir: "./data/outputs"
temp_dir: "./data/tmp"

# ASR settings
asr_model: "small"  # Options: tiny, base, small, medium, large
language: "ta"  # Tamil

# Field definitions
fields:
  textTamil: "Tamil text from transcription"
  textEnglish: "English translation text"
  speaker_roles: ["buyer", "seller", "other"]
  intent_categories: ["purchase", "inquiry", "complaint", "bargain", "greeting", "other"]
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
MP3 Files → WAV Conversion → Whisper ASR → NLU Analysis → Aggregation → JSON Outputs
```

### Components

- **`ingest_audio.py`** - MP3 to WAV conversion using FFmpeg
- **`transcribe.py`** - Tamil audio transcription using Whisper
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
  "confidence": 0.85
}
```

**New Audio Mapping Fields:**
- `audio_file_id`: Path to the source audio file (e.g., "tmp/Audio1.wav")
- `start_ms`: Segment start time in milliseconds from audio file beginning
- `end_ms`: Segment end time in milliseconds from audio file beginning  
- `duration_ms`: Segment duration in milliseconds (end_ms - start_ms)

These fields enable precise audio replay and debugging by providing deterministic pointers back to the raw audio source.

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
  "sentiment_distribution": {"positive": 3, "neutral": 9, "negative": 3},
  "dominant_emotion": "neutral",
  "top_intents": ["other", "purchase", "bargain", "inquiry", "complaint", "greeting"],
  "intent_distribution": {
    "other": 6, "purchase": 3, "bargain": 3, 
    "inquiry": 1, "complaint": 1, "greeting": 1
  },
  "sales_at_stop": {"tomato": 36, "onion": 15, "potato": 10},
  "inventory_after_sale": {"tomato": 22, "onion": 5, "potato": 8}
}
```

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

- **Whisper small model**: ~1GB RAM, ~2x real-time processing
- **Processing time**: ~2-3x audio duration
- **Memory usage**: Scales with audio length
- **First run**: Model download (~500MB for small model)

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg and ensure it's in your PATH
   - Verify with: `ffmpeg -version`

2. **Out of memory**
   - Use smaller Whisper model (tiny, base, small)
   - Process shorter audio files
   - Close other applications

3. **No segments generated**
   - Check audio file quality
   - Verify Tamil language detection
   - Check minimum segment word count in config

4. **Import errors**
   - Ensure all dependencies are installed
   - Check Python version (3.8+ required)
   - Verify virtual environment activation

5. **Sentiment labels incorrect**
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
- **English translation**: textEnglish field empty (needs translation service)

## Future Enhancements

- ML-based intent/sentiment/emotion models
- Speaker diarization for better segmentation
- Real-time processing capability
- Integration with external metadata sources
- Web UI for manual review and correction
- **NEW:** Google Translate API integration for textEnglish field
- **NEW:** Fine-tuned Tamil domain vocabulary patterns

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

