#!/usr/bin/env python3
"""
Test script to verify the updated pipeline produces the expected output format.
"""

import json
import sys
from pathlib import Path

def test_segment_schema(segment_data):
    """Test that a segment follows the required schema."""
    required_fields = [
        'seller_id', 'stop_id', 'segment_id', 'merged_block_id', 'utterance_index',
        'timestamp', 'speaker_role', 'role_confidence', 'textTamil', 'textEnglish',
        'is_translated', 'translation_confidence', 'intent', 'intent_confidence',
        'sentiment_score', 'sentiment_label', 'emotion', 'confidence', 'audio_file_id',
        'start_ms', 'end_ms', 'duration_ms', 'asr_confidence', 'products',
        'product_intents', 'product_sentiments', 'action_required', 'escalation_needed',
        'churn_risk', 'business_opportunity', 'needs_human_review',
        'product_intent_confidences', 'model_versions', 'pipeline_run_id'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in segment_data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Test specific field types and values
    issues = []
    
    # Test utterance_index is integer
    if not isinstance(segment_data['utterance_index'], int):
        issues.append("utterance_index should be integer")
    
    # Test timestamp is ISO8601 format
    if not segment_data['timestamp'].endswith('Z') and 'T' not in segment_data['timestamp']:
        issues.append("timestamp should be ISO8601 format")
    
    # Test speaker_role is valid
    valid_roles = ['buyer', 'seller', 'customer_bystander', 'other']
    if segment_data['speaker_role'] not in valid_roles:
        issues.append(f"speaker_role should be one of {valid_roles}")
    
    # Test sentiment_label is valid
    valid_sentiments = ['positive', 'neutral', 'negative']
    if segment_data['sentiment_label'] not in valid_sentiments:
        issues.append(f"sentiment_label should be one of {valid_sentiments}")
    
    # Test churn_risk is valid
    valid_risks = ['low', 'medium', 'high']
    if segment_data['churn_risk'] not in valid_risks:
        issues.append(f"churn_risk should be one of {valid_risks}")
    
    # Test products is list
    if not isinstance(segment_data['products'], list):
        issues.append("products should be list")
    
    # Test product objects have required fields
    for product in segment_data['products']:
        if not isinstance(product, dict):
            issues.append("product should be dict")
            continue
        required_product_fields = ['product_name', 'sku_id', 'product_confidence', 
                                 'text_span_tamil', 'text_span_english']
        for field in required_product_fields:
            if field not in product:
                issues.append(f"product missing {field}")
    
    # Test confidences are between 0 and 1
    confidence_fields = ['role_confidence', 'translation_confidence', 'intent_confidence', 
                        'confidence', 'asr_confidence']
    for field in confidence_fields:
        if field in segment_data:
            value = segment_data[field]
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                issues.append(f"{field} should be between 0 and 1")
    
    if issues:
        print(f"❌ Schema validation issues: {issues}")
        return False
    
    print("✅ Schema validation passed")
    return True

def test_example_segment():
    """Test the example segment from the requirements."""
    example_segment = {
        "seller_id": "S123",
        "stop_id": "STOP45",
        "segment_id": "SEG7093f921-utt1",
        "merged_block_id": "SEG7093f921",
        "utterance_index": 0,
        "timestamp": "2025-08-19T00:00:03.760000Z",
        "speaker_role": "buyer",
        "role_confidence": 0.5,
        "textTamil": "நீங்கள் ஒரு நாள் வருகிறீங்க, ஒரு நாள் வரமாற்றுறீங்க, ஆனால் சொல்லும்போது தினமும் வருகும் என்று சொல்றீங்க.",
        "textEnglish": "You come one day and don't come another day, but tell me that you will come every day.",
        "is_translated": True,
        "translation_confidence": 0.80,
        "intent": "complaint",
        "intent_confidence": 0.90,
        "sentiment_score": -0.16,
        "sentiment_label": "negative",
        "emotion": "disappointed",
        "confidence": 0.94,
        "audio_file_id": "Audio1.wav",
        "start_ms": 3760,
        "end_ms": 8680,
        "duration_ms": 4920,
        "asr_confidence": 0.95,
        "products": [],
        "product_intents": {},
        "product_sentiments": {},
        "action_required": True,
        "escalation_needed": True,
        "churn_risk": "high",
        "business_opportunity": False,
        "needs_human_review": True,
        "product_intent_confidences": {},
        "model_versions": {
            "asr": "asr-1.2.0",
            "translation": "trans-0.9.4",
            "ner": "ner-0.3.1",
            "absa": "absa-0.2.7"
        },
        "pipeline_run_id": "run-20250819-0001"
    }
    
    print("Testing example segment schema...")
    return test_segment_schema(example_segment)

def test_actual_output():
    """Test the actual pipeline output."""
    segments_file = Path("data/outputs/segments.json")
    
    if not segments_file.exists():
        print("❌ segments.json not found. Run the pipeline first.")
        return False
    
    try:
        with open(segments_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load segments.json: {e}")
        return False
    
    print(f"Testing {len(segments)} segments from pipeline output...")
    
    all_valid = True
    for i, segment in enumerate(segments):
        print(f"\nTesting segment {i+1}/{len(segments)} (ID: {segment.get('segment_id', 'unknown')})")
        if not test_segment_schema(segment):
            all_valid = False
    
    if all_valid:
        print(f"\n✅ All {len(segments)} segments passed schema validation")
    else:
        print(f"\n❌ Some segments failed schema validation")
    
    return all_valid

def main():
    """Run all tests."""
    print("Testing Updated Pipeline Output Schema")
    print("=" * 50)
    
    # Test example segment
    print("\n1. Testing example segment...")
    example_ok = test_example_segment()
    
    # Test actual output
    print("\n2. Testing actual pipeline output...")
    actual_ok = test_actual_output()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Example segment: {'✅ PASS' if example_ok else '❌ FAIL'}")
    print(f"Actual output: {'✅ PASS' if actual_ok else '❌ FAIL'}")
    
    if example_ok and actual_ok:
        print("\n🎉 All tests passed! Pipeline output matches required schema.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
