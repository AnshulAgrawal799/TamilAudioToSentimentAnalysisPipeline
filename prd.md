# PRD: Tamil Audio to Sentiment Analysis Pipeline

**Version:** 1.1  
**Date:** 2025-01-27  
**Type:** Minimal Viable Prototype (Updated with Schema Fixes)

---

## 1. Overview

Build a pipeline that processes Tamil MP3 audio files and outputs sentiment analysis results in JSON format. The pipeline transcribes audio, segments by speaker, and analyzes intent, sentiment, and emotion for each segment.

**Input:** MP3 files in `audio/` folder  
**Output:** Per-segment JSON + aggregated summaries  
**Scope:** Batch processing only, no real-time streaming

**Recent Updates:** Fixed data schema inconsistencies, improved sentiment labeling, enhanced intent classification with Tamil patterns

---

## 2. Core Requirements

### 2.1 Functional Requirements
- Process all MP3 files in `audio/` directory
- Transcribe Tamil audio to text with timestamps
- Segment audio by speaker turns
- Analyze each segment for:
  - Intent (purchase, inquiry, complaint, bargain, greeting, other)
  - Sentiment score (-1.0 to +1.0)
  - Sentiment label (positive, neutral, negative) - **FIXED: Proper thresholding**
  - Emotion (happy, disappointed, neutral)
  - Confidence score (0.0 to 1.0)
- Generate aggregated summaries per stop and per day
- Use placeholder values for missing metadata

### 2.2 Technical Requirements
- ASR accuracy: Target ≤20% WER for Tamil
- Support for Tamil language audio
- Batch processing capability
- JSON output matching corrected schemas
- Configurable placeholder values
- **NEW:** Tamil + English pattern matching for NLU analysis

---

## 3. Data Schemas

### 3.1 Per-Segment Output
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

**Audio Mapping Fields (NEW):**
- `audio_file_id`: Canonical identifier or relative path to source audio file
- `start_ms`: Segment start time in milliseconds from audio file beginning
- `end_ms`: Segment end time in milliseconds from audio file beginning
- `duration_ms`: Segment duration in milliseconds (end_ms - start_ms)

These fields provide deterministic pointers back to raw audio for replay, QA, and debugging purposes.

**Key Fixes Applied:**
- ✅ Added `textTamil` and `textEnglish` fields
- ✅ Fixed sentiment labeling: `score >= 0.1` → positive, `score <= -0.1` → negative
- ✅ Enhanced intent classification with Tamil patterns
- ✅ Proper speaker role enumeration: `["buyer", "seller", "other"]`
- ✅ **NEW:** Added audio mapping fields for precise audio replay

### 3.2 Aggregated Stop Output
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
  "audio_files": {
    "tmp/Audio1.wav": {
      "n_segments": 8,
      "total_duration_ms": 15600,
      "time_range": {"start_ms": 0, "end_ms": 15600}
    },
    "tmp/Audio2.wav": {
      "n_segments": 7,
      "total_duration_ms": 14200,
      "time_range": {"start_ms": 0, "end_ms": 14200}
    }
  },
  "sales_at_stop": {"tomato": 36, "onion": 15, "potato": 10},
  "inventory_after_sale": {"tomato": 22, "onion": 5, "potato": 8}
}
```

**New Audio File Statistics:**
- `audio_files`: Per-audio file breakdown with segment counts, durations, and time ranges
- Enables tracking of which audio files contributed to each stop's analysis

### 3.3 Aggregated Day Output
```json
{
  "seller_id": "S123",
  "date": "2025-08-19",
  "total_stops": 7,
  "total_segments": 15,
  "total_calls": 5,
  "avg_sentiment_score": -0.01,
  "sentiment_distribution": {"positive": 3, "neutral": 9, "negative": 3},
  "dominant_emotion": "neutral",
  "intent_distribution": {
    "other": 6, "purchase": 3, "bargain": 3, 
    "inquiry": 1, "complaint": 1, "greeting": 1
  },
  "audio_files": {
    "tmp/Audio1.wav": {
      "n_segments": 8,
      "total_duration_ms": 15600,
      "time_range": {"start_ms": 0, "end_ms": 15600}
    },
    "tmp/Audio2.wav": {
      "n_segments": 7,
      "total_duration_ms": 14200,
      "time_range": {"start_ms": 0, "end_ms": 14200}
    }
  },
  "total_sales": {"tomato": 36, "onion": 15, "potato": 10},
  "closing_inventory": {"tomato": 22, "onion": 5, "potato": 8}
}
```

**New Audio File Statistics:**
- `audio_files`: Per-audio file breakdown across all stops for the day
- Provides complete audio mapping overview for daily analysis

---

## 4. Pipeline Architecture

### 4.1 Processing Steps
1. **Audio Ingestion**: Discover MP3 files, convert to WAV format
2. **Transcription**: Use Whisper ASR to transcribe Tamil audio with timestamps
3. **Segmentation**: Split transcript into speaker segments
4. **NLU Analysis**: Analyze intent, sentiment, and emotion for each segment
5. **Aggregation**: Generate stop-level and day-level summaries
6. **Output**: Write JSON files to `data/outputs/`

### 4.2 File Structure
```
/
├── audio/                          # Input MP3 files
├── data/
│   ├── tmp/                       # Temporary WAV files
│   └── outputs/                   # Final JSON outputs
│       ├── segments.json          # Per-segment data (JSON array)
│       ├── aggregate_stop_*.json  # Stop-level summaries
│       └── aggregate_day_*.json   # Day-level summaries
├── src/                           # Pipeline scripts
│   ├── ingest_audio.py           # MP3 to WAV conversion
│   ├── transcribe.py             # ASR transcription
│   ├── analyze.py                # NLU analysis (UPDATED)
│   ├── aggregate.py              # Generate summaries
│   └── main.py                   # End-to-end pipeline
├── config.yaml                    # Configuration (UPDATED)
└── requirements.txt               # Dependencies
```

---

## 5. Configuration (UPDATED)

### 5.1 config.yaml
```yaml
# Audio processing
audio_dir: "./audio"
output_dir: "./data/outputs"
temp_dir: "./data/tmp"

# ASR settings
asr_model: "small"
language: "ta"

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
  inventory_after_sale:
    tomato: 22
    onion: 5
    potato: 8

# NLU thresholds
confidence_threshold: 0.3
min_segment_words: 4
```

---

## 6. Implementation Plan

### Phase 1: Core Pipeline (2-3 days) ✅ COMPLETED
- Set up project structure and dependencies
- Implement audio ingestion (MP3 → WAV)
- Integrate Whisper ASR for Tamil transcription
- Basic segmentation by ASR blocks

### Phase 2: NLU Analysis (2-3 days) ✅ COMPLETED + IMPROVED
- Implement intent classification (rule-based for prototype)
- Implement sentiment analysis (rule-based for prototype)
- Implement emotion detection (rule-based for prototype)
- Add confidence scoring
- **NEW:** Enhanced with Tamil + English patterns
- **NEW:** Fixed sentiment labeling thresholds

### Phase 3: Aggregation & Output (1-2 days) ✅ COMPLETED + ENHANCED
- Implement stop-level aggregation
- Implement day-level aggregation
- Generate JSON outputs matching schemas
- Add validation and error handling
- **NEW:** Enhanced with audio file statistics and mapping information
- **NEW:** Complete audio ↔ segment traceability for QA and debugging

### Phase 4: Testing & Refinement (1-2 days) ✅ COMPLETED
- Test with sample audio files
- Validate output schemas
- Performance optimization
- Documentation

---

## 7. Dependencies

### System Requirements
- Python 3.8+
- FFmpeg for audio conversion
- 4GB+ RAM for ASR processing

### Python Packages
```
torch
torchaudio
whisper
nltk
pyyaml
python-dateutil
```

---

## 8. Usage

### 8.1 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/main.py --config config.yaml
```

### 8.2 Expected Outputs
- `data/outputs/segments.json` - Per-segment JSON array with audio mapping fields
- `data/outputs/aggregate_stop_<STOPID>_<DATE>.json` - Stop-level summary with audio file statistics
- `data/outputs/aggregate_day_<SELLERID>_<DATE>.json` - Day-level summary with audio file statistics

**New Audio Mapping Features:**
- **Segment-Level**: Each segment now includes `audio_file_id`, `start_ms`, `end_ms`, and `duration_ms`
- **Aggregate-Level**: Each aggregate includes `audio_files` section with per-file statistics
- **Complete Traceability**: From segment back to exact audio file and time range

**New Aggregate Features:**
- **Audio File Statistics**: Each aggregate now includes `audio_files` section with:
  - Number of segments per audio file
  - Total duration of segments per audio file
  - Time range (start_ms to end_ms) covered by segments in each audio file
- **Enhanced Debugging**: Aggregates now provide complete audio mapping information for QA and troubleshooting

### 8.3 Example Output Files
- `data/outputs/segments.json`
- `data/outputs/aggregate_stop_STOP45_2025-08-19.json`
- `data/outputs/aggregate_day_S123_2025-08-19.json`

---

## 9. Success Criteria

### 9.1 Functional ✅ ACHIEVED
- Pipeline processes all MP3 files in audio/ directory
- Outputs match corrected JSON schemas exactly
- Placeholder values are used for missing metadata
- All required fields are populated
- **NEW:** Proper sentiment labeling based on score thresholds
- **NEW:** Meaningful intent distribution with Tamil patterns

### 9.2 Technical ✅ ACHIEVED
- ASR transcription works for Tamil audio
- Segmentation produces logical speaker turns
- NLU analysis provides reasonable intent/sentiment/emotion
- Pipeline runs without errors on sample data
- **NEW:** Tamil + English pattern matching for better classification

### 9.3 Quality ✅ ACHIEVED
- JSON outputs are valid and well-formed
- Timestamps are properly formatted (ISO 8601)
- Confidence scores are reasonable (0.0-1.0)
- Sentiment scores are normalized (-1.0 to +1.0)
- **NEW:** Sentiment labels correctly match their scores
- **NEW:** Intent classification shows meaningful distribution

---

## 10. Limitations & Future Work

### 10.1 Current Limitations
- Rule-based NLU (no ML models) - **IMPROVED with Tamil patterns**
- No speaker diarization (segments by ASR blocks only)
- Placeholder values for sales/inventory data
- Batch processing only
- **NEW:** English translation field empty (needs translation service)

### 10.2 Future Enhancements
- ML-based intent/sentiment/emotion models
- Speaker diarization for better segmentation
- Real-time processing capability
- Integration with external metadata sources
- Web UI for manual review and correction
- **NEW:** Google Translate API integration for textEnglish field
- **NEW:** Fine-tuned Tamil domain vocabulary patterns

---

## 11. Appendix

### 11.1 Filename Convention
Recommended: `seller_<SELLERID>_STOP<STOPID>_<YYYYMMDD>_<HHMMSS>.mp3`

Example: `seller_S123_STOP45_20250819_191110.mp3`

### 11.2 Error Handling
- Invalid audio files: Skip with warning
- ASR failures: Log error, continue with next file
- Missing metadata: Use placeholders from config
- Output errors: Log and retry once

### 11.3 Performance Notes
- Whisper small model: ~1GB RAM, ~2x real-time
- Processing time: ~2-3x audio duration
- Memory usage: Scales with audio length
- Parallel processing: Not implemented in prototype

### 11.4 Schema Fixes Applied
- ✅ Added `textTamil` and `textEnglish` fields to match expected schema
- ✅ Fixed sentiment labeling: negative scores now properly labeled as "negative"
- ✅ Enhanced intent classification with Tamil vocabulary patterns
- ✅ Improved speaker role detection with Tamil patterns
- ✅ Added proper field definitions and enumerations in config
- ✅ Created corrected example outputs for validation
- ✅ **NEW:** Added audio mapping fields (`audio_file_id`, `start_ms`, `end_ms`, `duration_ms`)
- ✅ **NEW:** Enhanced aggregates with audio file statistics for complete traceability

### 11.5 Example Output Format
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
