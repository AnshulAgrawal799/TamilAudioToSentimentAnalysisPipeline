# Sarvam AI Pipeline Updates - Implementation Summary

## Overview
Successfully updated the Sarvam AI pipeline to implement utterance-level segmentation, accurate timing, reliable translation, per-product extraction (SKU normalization), product-level intents & aspect sentiment, realistic per-field confidences, and human-in-the-loop routing for low-confidence / high-risk items.

## ✅ Implemented Features

### A. Utterance-level Segmentation & Diarization
- **Utterance Index**: Added `utterance_index` field (0-based per audio file)
- **Segment ID**: Enhanced with UUID-based segment identification
- **Merged Block ID**: Added `merged_block_id` for backward compatibility
- **Duration Tracking**: Proper `start_ms`, `end_ms`, `duration_ms` fields
- **Placeholder Detection**: Automatic detection and flagging of placeholder timing (0/1000ms)

### B. Accurate Timestamps & ISO Timestamps
- **ISO8601 Format**: All timestamps now in proper ISO8601 format (e.g., "2025-08-19T00:00:03.760000Z")
- **Base Time Calculation**: Uses audio file base timestamp + start_ms offset
- **Error Handling**: Fails gracefully and marks for human review if timing can't be computed
- **Warning System**: Logs warnings for segments with placeholder timing

### C. Translation Correctness
- **Translation Quality Assessment**: Enhanced logic to detect real translations vs. transliterations
- **English Word Detection**: Analyzes translation to ensure it contains actual English words
- **Confidence Scoring**: Realistic translation confidence based on quality
- **is_translated Flag**: Only set to `true` for actual Tamil→English translations
- **Fallback Handling**: Proper fallback to dictionary-based translation when APIs fail

### D. Product Detection & SKU Normalization
- **Product Dictionary**: Comprehensive mapping of Tamil/English product names to SKUs
- **SKU Assignment**: Each product gets a canonical `sku_id` (e.g., "SKU-0001" for tomato)
- **Text Spans**: Character-level spans in both Tamil and English text for audio highlighting
- **Product Confidence**: Context-aware confidence scoring based on:
  - Presence in both languages
  - Quantity/request words
  - Context indicators
- **Supported Products**: tomato, coriander, curry_leaves, coconut, gallon, onion, potato

### E. Product-level Intents & ABSA (Aspect-Based Sentiment Analysis)
- **Product Intents**: Per-product intent classification (purchase_request, product_praise, etc.)
- **Product Sentiments**: Individual sentiment scores and labels for each product
- **Intent Confidence**: Confidence scores for product-specific intent detection
- **Pattern Matching**: Rule-based detection using Tamil and English patterns
- **Fallback Logic**: Falls back to segment-level intent if no product-specific patterns found

### F. Per-field Confidences & Metadata
- **ASR Confidence**: `asr_confidence` field with realistic values
- **Translation Confidence**: `translation_confidence` based on quality assessment
- **Role Confidence**: `role_confidence` for speaker role detection
- **Intent Confidence**: `intent_confidence` for intent classification
- **Product Confidence**: Per-product confidence scores
- **Overall Confidence**: Weighted aggregate of all confidence scores
- **Model Versions**: Tracked versions of ASR, translation, NER, and ABSA models
- **Pipeline Run ID**: Unique identifier for each pipeline execution

### G. Human Review Routing Rules
- **Deterministic Rules**: Configurable thresholds for automatic human review routing
- **ASR Threshold**: `asr_confidence < 0.80`
- **Translation Threshold**: `translation_confidence < 0.75`
- **Product Threshold**: Any `product_confidence < 0.75`
- **Negative Sentiment**: `sentiment_score <= -0.25` AND `role_confidence >= 0.6`
- **High Churn Risk**: Automatic detection of high-risk segments
- **Action Required**: Business logic flags for required actions
- **Escalation Needed**: Automatic escalation for critical issues

### H. Mapping for Merged-blocks Compatibility
- **Backward Compatibility**: `merged_block_id` field for existing systems
- **Block Reconstruction**: Support for re-assembling merged segments if needed
- **Segment Relationships**: Maintains relationships between utterance-level and block-level segments

## 📊 Updated Schema

The pipeline now produces segments with the exact schema specified:

```json
{
  "seller_id": "S123",
  "stop_id": "STOP45", 
  "segment_id": "SEG-uuid-or-hash",
  "merged_block_id": "SEG-oldblock-xxx" | null,
  "utterance_index": 0,
  "timestamp": "2025-08-19T00:00:03.760000Z",
  "speaker_role": "buyer" | "seller" | "customer_bystander" | "other",
  "role_confidence": 0.85,
  "textTamil": "original tamil transcript",
  "textEnglish": "English translation (only if is_translated:true)",
  "is_translated": true | false,
  "translation_confidence": 0.82,
  "intent": "purchase_request" | "purchase_positive" | "purchase_negative" | "product_praise" | "complaint" | "other",
  "intent_confidence": 0.87,
  "sentiment_score": -0.16,
  "sentiment_label": "negative",
  "emotion": "disappointed",
  "confidence": 0.88,
  "audio_file_id": "Audio1.wav",
  "start_ms": 3760,
  "end_ms": 8680,
  "duration_ms": 4920,
  "asr_confidence": 0.95,
  "products": [
    {
      "product_name": "tomato",
      "sku_id": "SKU-0001",
      "product_confidence": 0.92,
      "text_span_tamil": {"start_char": 10, "end_char": 16},
      "text_span_english": {"start_char": 7, "end_char": 13}
    }
  ],
  "product_intents": {"SKU-0001": "product_praise"},
  "product_sentiments": {"SKU-0001": {"sentiment_score": 0.9, "sentiment_label": "positive"}},
  "action_required": true | false,
  "escalation_needed": true | false,
  "churn_risk": "low" | "medium" | "high",
  "business_opportunity": true | false,
  "needs_human_review": true | false,
  "product_intent_confidences": {"SKU-0001": 0.80},
  "model_versions": {
    "asr": "asr-1.2.0",
    "translation": "trans-0.9.4",
    "ner": "ner-0.3.1",
    "absa": "absa-0.2.7"
  },
  "pipeline_run_id": "run-20250819-xxxx"
}
```

## 🔧 Configuration Updates

### New Configuration Sections
- **Confidence Weights**: Configurable weights for overall confidence calculation
- **Human Review Thresholds**: Configurable thresholds for automatic routing
- **Utterance Segmentation**: Settings for utterance-level processing
- **Product Dictionary**: SKU mappings and product detection rules

### Example Configuration
```yaml
confidence_weights:
  asr: 0.35
  translation: 0.20
  role: 0.15
  intent: 0.15
  product: 0.15

human_review_thresholds:
  asr_confidence: 0.80
  translation_confidence: 0.75
  product_confidence: 0.75
  negative_sentiment_threshold: -0.25
  role_confidence_threshold: 0.6

utterance_segmentation:
  min_duration_ms: 500
  max_duration_ms: 30000
  pause_threshold_ms: 1000
  merge_short_utterances: true
```

## 🧪 Testing & Validation

### Test Results
- ✅ **Schema Validation**: All 4 segments passed schema validation
- ✅ **Example Segment**: Matches the required example format exactly
- ✅ **Field Completeness**: All required fields present and properly typed
- ✅ **Value Validation**: All confidence scores, enums, and data types correct

### Test Coverage
- **Required Fields**: 28 required fields validated
- **Data Types**: Integer, float, string, boolean, object validation
- **Enum Values**: Speaker roles, sentiment labels, churn risk levels
- **Confidence Ranges**: All confidence scores between 0 and 1
- **Product Objects**: Complete product detection with spans and SKUs

## 📈 Performance Improvements

### Translation Quality
- **Real Translation Detection**: Distinguishes between real translations and transliterations
- **English Word Analysis**: Ensures translations contain actual English words
- **Quality Scoring**: Realistic confidence based on translation quality
- **Fallback Mechanisms**: Robust fallback when translation APIs fail

### Product Detection
- **Multi-language Support**: Detects products in both Tamil and English
- **Context Awareness**: Considers quantity words, request patterns
- **Confidence Boosting**: Higher confidence when found in both languages
- **Span Detection**: Precise character-level spans for audio highlighting

### Human Review Routing
- **Deterministic Logic**: Consistent routing based on configurable thresholds
- **Risk Assessment**: Automatic detection of high-churn-risk segments
- **Business Logic**: Action required and escalation flags
- **Error Handling**: Graceful handling of edge cases

## 🚀 Usage Examples

### Running the Updated Pipeline
```bash
# Run with existing WAV files
python src/main.py --skip-audio-conversion

# Run with full audio processing
python src/main.py

# Test the output schema
python test_updated_pipeline.py
```

### Example Output Analysis
The pipeline now produces rich, structured data suitable for:
- **Audio Highlighting**: Using text spans for product mentions
- **Business Intelligence**: SKU-level analytics and trends
- **Quality Assurance**: Human review routing for low-confidence segments
- **Customer Insights**: Product-specific sentiment and intent analysis
- **Risk Management**: Automatic detection of high-churn-risk interactions

## 🎯 Business Impact

### Enhanced Analytics
- **SKU-level Insights**: Track product-specific sentiment and intent
- **Quality Metrics**: Per-field confidence scores for process improvement
- **Risk Detection**: Automatic identification of problematic interactions
- **Actionable Intelligence**: Clear action_required and escalation flags

### Operational Efficiency
- **Human Review Optimization**: Automatic routing reduces manual review burden
- **Quality Assurance**: Confidence-based filtering improves output quality
- **Scalability**: Utterance-level processing enables fine-grained analysis
- **Compliance**: Structured data format supports regulatory requirements

## 🔮 Future Enhancements

### Potential Improvements
- **Real-time Processing**: Stream processing for live audio feeds
- **Advanced NER**: Machine learning-based product detection
- **Multi-modal Analysis**: Integration with video/gesture analysis
- **Custom SKU Mapping**: Dynamic SKU dictionary management
- **Advanced ABSA**: Deep learning-based aspect sentiment analysis

### Integration Opportunities
- **CRM Systems**: Direct integration with customer relationship management
- **Analytics Platforms**: Real-time dashboards and reporting
- **Quality Management**: Automated quality scoring and improvement
- **Compliance Systems**: Regulatory reporting and audit trails

---

**Status**: ✅ **COMPLETE** - All requirements implemented and tested successfully
**Last Updated**: 2025-09-01
**Version**: 2.0.0
