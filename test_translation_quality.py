#!/usr/bin/env python3
"""
Test script to evaluate translation quality improvements.
Compares different translation methods and shows quality metrics.
"""

import json
import logging
from pathlib import Path
from src.analyze import NLUAnalyzer
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_translation_quality():
    """Test translation quality with sample Tamil text."""
    
    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found. Please run from the project root.")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create analyzer
    analyzer = NLUAnalyzer(config)
    
    # Test cases from the problematic examples
    test_cases = [
        {
            "original": "நேது ஓர் உண்மைத் திடைக்கிற இது அடிக்கடிப் பெண்ணிருக்க nya, ஒன்றுமென்னால் வரப்ப்பாடுங்கள் ஆனாம் திடைப்பன்போது நிஷக்கமாக வரோ அவனுடைய விளைவில்ல.ோதனை ஒரு ஒரு நாள் வர மாட்டிறீ Garrett்சேன் ஆனால் கொண்டிருவன் திடைப்ப அழுபக்கும்ல என்று சத்தம் வர்க்கும்', நேற்று உணல்",
            "expected": "Yesterday no one came, but today you are here, so we are not buying vegetables. You say you will come everyday, but you don't come often.",
            "description": "Audio1 - Complex Tamil text with ASR artifacts"
        },
        {
            "original": "நன்றித்துள் சிலக்கக் கூறுங்கு வழக்கம இருக்கிறது நான் உறுத்திய சிலக்க மிகவும் ந.",
            "expected": "Thank you, I am very good.",
            "description": "Audio2 - Simple greeting with ASR noise"
        },
        {
            "original": "நாளைக்கு வரும்பும் milk, காலாந்து எடுத்துக்கொள்கிறியும் ஆக்கு ஏறு முடியும் அங்கே நம்மா காலாந்தால் இங்கே நbbepaidieszanas",
            "expected": "I bought all the vegetables, can you give me some coriander leaves?",
            "description": "Audio4 - Mixed Tamil-English with ASR artifacts"
        }
    ]
    
    print("Testing Translation Quality Improvements")
    print("=" * 60)
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print("-" * 50)
        
        # Clean the ASR text
        cleaned_text = analyzer._clean_asr_text(test_case['original'])
        print(f"Original (with ASR artifacts): {test_case['original'][:100]}...")
        print(f"Cleaned text: {cleaned_text}")
        
        # Translate to English
        english_translation = analyzer._translate_to_english(cleaned_text)
        print(f"English translation: {english_translation}")
        print(f"Expected: {test_case['expected']}")
        
        # Quality metrics
        original_length = len(test_case['original'])
        cleaned_length = len(cleaned_text)
        noise_removed = original_length - cleaned_length
        
        print(f"Quality metrics:")
        print(f"  - Noise removed: {noise_removed} characters ({noise_removed/original_length*100:.1f}%)")
        print(f"  - Text preserved: {cleaned_length/original_length*100:.1f}%")
        print(f"  - Translation available: {'Yes' if english_translation else 'No'}")
        print()
    
    print("Translation Quality Test Complete!")
    print()
    print("Recommendations:")
    print("1. Use Google Cloud Translate API for production (better quality)")
    print("2. Run setup_google_translate.py to configure Google Cloud")
    print("3. Consider using Whisper large-v3 model for better ASR")
    print("4. Review cleaned text before translation to ensure quality")

if __name__ == "__main__":
    test_translation_quality()
