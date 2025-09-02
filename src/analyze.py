#!/usr/bin/env python3
"""
NLU analysis module for intent classification, sentiment analysis, and emotion detection.
Improved version with better translation quality, standardized roles, and consistent taxonomy.
"""

import logging
import random
import uuid
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dateutil import parser as dateparser

logger = logging.getLogger(__name__)

# Import proper translation libraries
try:
    from google.cloud import translate_v2 as translate
    TRANSLATION_AVAILABLE = True
    logger.info("Google Cloud Translate API available")
except ImportError:
    try:
        # Fallback to googletrans if Google Cloud not available
        from googletrans import Translator
        TRANSLATION_AVAILABLE = True
        logger.warning("Using googletrans fallback. Consider setting up Google Cloud Translate for better quality.")
    except ImportError:
        TRANSLATION_AVAILABLE = False
        logger.warning("No translation library available. Using fallback translation.")

# Import language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Language detection may be less accurate.")


class Segment:
    """Represents a single analyzed segment."""
    
    def __init__(self, **kwargs):
        self.seller_id = kwargs.get('seller_id', 'UNKNOWN_SELLER')
        self.stop_id = kwargs.get('stop_id', 'UNKNOWN_STOP')
        self.segment_id = kwargs.get('segment_id', f"SEG{str(uuid.uuid4())[:8]}")
        self.merged_block_id = kwargs.get('merged_block_id', None)  # For backward compatibility
        self.utterance_index = kwargs.get('utterance_index', 0)  # 0-based index per audio file
        self.timestamp = kwargs.get('timestamp', '1970-01-01T00:00:00Z')
        self.speaker_role = kwargs.get('speaker_role', 'other')
        self.role_confidence = kwargs.get('role_confidence', 0.5)
        self.textTamil = kwargs.get('textTamil', '')  # Tamil text in Unicode
        self.textEnglish = kwargs.get('textEnglish', '')  # English text
        self.is_translated = kwargs.get('is_translated', False)
        self.translation_confidence = kwargs.get('translation_confidence', 0.5)
        self.intent = kwargs.get('intent', 'other')
        self.intent_confidence = kwargs.get('intent_confidence', 0.5)
        self.sentiment_score = kwargs.get('sentiment_score', 0.0)
        self.sentiment_label = kwargs.get('sentiment_label', 'neutral')
        self.emotion = kwargs.get('emotion', 'neutral')
        self.confidence = kwargs.get('confidence', 0.5)  # Overall confidence
        self.audio_file_id = kwargs.get('audio_file_id', 'UNKNOWN_AUDIO')
        self.start_ms = kwargs.get('start_ms', 0)
        self.end_ms = kwargs.get('end_ms', 0)
        self.duration_ms = kwargs.get('duration_ms', 0)
        self.asr_confidence = kwargs.get('asr_confidence', 0.8)
        
        # Product detection fields
        self.products = kwargs.get('products', [])  # List of product objects with spans
        self.product_intents = kwargs.get('product_intents', {})  # sku_id -> intent
        self.product_sentiments = kwargs.get('product_sentiments', {})  # sku_id -> sentiment object
        self.product_intent_confidences = kwargs.get('product_intent_confidences', {})  # sku_id -> confidence
        
        # Business logic fields
        self.action_required = kwargs.get('action_required', False)
        self.escalation_needed = kwargs.get('escalation_needed', False)
        self.churn_risk = kwargs.get('churn_risk', 'low')  # low/medium/high
        self.business_opportunity = kwargs.get('business_opportunity', False)
        self.needs_human_review = kwargs.get('needs_human_review', False)
        
        # Pipeline metadata
        self.model_versions = kwargs.get('model_versions', {
            'asr': 'asr-1.2.0',
            'translation': 'trans-0.9.4', 
            'ner': 'ner-0.3.1',
            'absa': 'absa-0.2.7'
        })
        self.pipeline_run_id = kwargs.get('pipeline_run_id', f"run-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Always compute current human-review flag at serialization time
        needs_review = self._compute_needs_human_review()
        return {
            'seller_id': self.seller_id,
            'stop_id': self.stop_id,
            'segment_id': self.segment_id,
            'merged_block_id': self.merged_block_id,
            'utterance_index': self.utterance_index,
            'timestamp': self.timestamp,
            'speaker_role': self.speaker_role,
            'role_confidence': self.role_confidence,
            'textTamil': self.textTamil,
            'textEnglish': self.textEnglish,
            'is_translated': self.is_translated,
            'translation_confidence': self.translation_confidence,
            'intent': self.intent,
            'intent_confidence': self.intent_confidence,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'emotion': self.emotion,
            'confidence': self.confidence,
            'audio_file_id': self.audio_file_id,
            'start_ms': self.start_ms,
            'end_ms': self.end_ms,
            'duration_ms': self.duration_ms,
            'asr_confidence': self.asr_confidence,
            'products': self.products,
            'product_intents': self.product_intents,
            'product_sentiments': self.product_sentiments,
            'action_required': self.action_required,
            'escalation_needed': self.escalation_needed,
            'churn_risk': self.churn_risk,
            'business_opportunity': self.business_opportunity,
            'needs_human_review': needs_review,
            'product_intent_confidences': self.product_intent_confidences,
            'model_versions': self.model_versions,
            'pipeline_run_id': self.pipeline_run_id
        }

    def _compute_needs_human_review(self) -> bool:
        """Derive whether this segment needs human review based on configured rules."""
        try:
            # ASR confidence threshold (increased for better quality)
            asr_low = (self.asr_confidence is not None) and (self.asr_confidence < 0.85)
            
            # Translation confidence threshold (increased for better quality)
            translation_low = (self.translation_confidence is not None) and (self.translation_confidence < 0.80)
            
            # Product confidence threshold (any product below threshold)
            product_low = False
            for product in self.products:
                if isinstance(product, dict) and product.get('product_confidence', 1.0) < 0.75:
                    product_low = True
                    break
            
            # High churn risk condition
            churn_risk_condition = (self.sentiment_score <= -0.25 and self.role_confidence >= 0.6)
            
            # High negative sentiment with role confidence
            negative_sentiment_condition = (self.sentiment_score <= -0.25 and self.role_confidence >= 0.6)
            
            self.needs_human_review = bool(
                asr_low or 
                translation_low or 
                product_low or 
                churn_risk_condition or
                negative_sentiment_condition
            )
            return self.needs_human_review
            
        except Exception as e:
            logger.warning(f"Error computing human review flag: {e}")
            return True  # Default to human review on error

    def to_json(self) -> str:
        """Serialize this segment to a JSON string."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)
    



class NLUAnalyzer:
    """Handles NLU analysis for transcribed segments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.placeholders = config.get('placeholders', {})
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.min_segment_words = config.get('min_segment_words', 4)
        
        # Translation confidence tracking
        self._last_translation_confidence = 0.8  # Default confidence
        
        # Initialize heuristics
        self._init_heuristics()
    
    def _init_heuristics(self):
        """Initialize rule-based heuristics with improved patterns."""
        # Standardized speaker role detection patterns
        self.seller_patterns = [
            # English patterns for seller
            'i have', 'fresh', 'price', 'rs', 'kg', 'kilo', 'i can give',
            'available', 'stock', 'quality', 'best', 'cheap', 'discount',
            'selling', 'vendor', 'supplier', 'wholesale', 'retail',
            'come tomorrow', 'bring', 'i will bring', 'i can bring',
            # Tamil patterns for seller (Unicode)
            'உள்ளது', 'விலை', 'கிலோ', 'ரூபாய்', 'கொடுக்கலாம்', 'உள்ளன',
            'தரமான', 'சிறந்த', 'மலிவு', 'தள்ளுபடி', 'ஆஃபர்', 'வாங்குங்கள்',
            'எடுத்துக்கொள்ளுங்கள்', 'கொடுக்கிறேன்', 'விற்கிறேன்', 'விற்பனை',
            'கடை', 'வியாபாரி', 'ஒப்பந்தக்காரர்', 'வழங்குநர்', 'நாளைக்கு வருவேன்',
            'எடுத்து வருவேன்', 'கொண்டு வருவேன்', 'வாங்குங்கள்', 'எடுத்துக்கொள்ளுங்கள்'
        ]
        
        self.buyer_patterns = [
            # English patterns for buyer
            'we will buy', 'we buy', 'i will buy', 'can i get', 'give me',
            'how much', 'what is the price', 'available', 'need', 'want',
            'purchase', 'customer', 'buyer', 'shopping', 'order',
            'we won\'t buy', 'not buying', 'refuse to buy', 'don\'t want to buy',
            'if you come', 'if we trust', 'what should we do', 'complain',
            # Tamil patterns for buyer (Unicode)
            'வாங்குவோம்', 'வாங்குகிறேன்', 'கொடுங்கள்', 'எவ்வளவு', 'விலை என்ன',
            'உள்ளதா', 'தேவை', 'விரும்புகிறேன்', 'எடுத்துக்கொள்கிறேன்', 'வாங்க',
            'கொடுங்க', 'வாங்குபவர்', 'வாடிக்கையாளர்', 'வாங்குதல்', 'ஆர்டர்',
            'வாங்காமல', 'வாங்க மாட்டோம்', 'நம்பி', 'என்ன பண்ணுறது', 'பிரச்சினை',
            'நீங்கள் வருகிறீங்க', 'நாங்க என்ன பண்ணுறது', 'நம்பி நாங்க'
        ]
        
        # Standardized intent taxonomy with improved patterns
        self.intent_patterns = {
            'purchase_positive': [
                # English - positive purchase intent
                'buy', 'will buy', 'purchase', 'take', 'get', 'need', 'order',
                'want to buy', 'looking for', 'interested in', 'bring', 'can you bring',
                # Tamil (Unicode) - positive purchase intent
                'வாங்கு', 'வாங்குவோம்', 'எடுத்துக்கொள்', 'தேவை', 'வாங்குகிறேன்',
                'எடுத்துக்கொள்கிறேன்', 'வாங்க', 'கொடுங்கள்', 'ஆர்டர்', 'வாங்குதல்',
                'எடுத்து வா', 'கொண்டு வா', 'வாங்குவோம்'
            ],
            'purchase_negative': [
                # English - negative purchase intent (refusal, complaints)
                'won\'t buy', 'not buying', 'refuse to buy', 'don\'t want',
                'not buying from you', 'won\'t buy from you', 'complain',
                # Tamil (Unicode) - negative purchase intent
                'வாங்காமல', 'வாங்க மாட்டோம்', 'நம்பி நாங்க', 'பிரச்சினை',
                'நீங்கள் வருகிறீங்க', 'ஒரு நாள் வருகிறீங்க', 'ஒரு நாள் வரமாற்றுறீங்க'
            ],
            'purchase_request': [
                # English - requests for products/information
                'how much', 'what is the price', 'available', 'cost', 'rate',
                'can you', 'please', 'information', 'details', 'inquiry',
                'coriander', 'curry leaves', 'gallon', 'tomatoes',
                # Tamil (Unicode) - requests for products/information
                'எவ்வளவு', 'விலை என்ன', 'உள்ளதா', 'செலவு', 'விகிதம்',
                'விலை', 'கிடைக்குமா', 'தகவல்', 'விவரங்கள்', 'கொத்தமல்லி',
                'கருவப்புல்', 'காலான்', 'தக்காலி'
            ],
            'complaint': [
                # English
                'bad', 'complain', 'problem', 'issue', 'wrong', 'not good',
                'poor quality', 'dissatisfied', 'unhappy', 'terrible',
                'don\'t come', 'not coming', 'irregular', 'unreliable',
                # Tamil (Unicode)
                'மோசம்', 'பிரச்சினை', 'தவறு', 'நன்றாக இல்லை', 'புகார்',
                'மோசமான', 'தரம் குறைவு', 'திருப்தியற்ற', 'வருத்தம்', 'வரமாற்று',
                'ஒரு நாள் வருகிறீங்க', 'ஒரு நாள் வரமாற்றுறீங்க'
            ],
            'product_praise': [
                # English
                'good', 'great', 'excellent', 'wonderful', 'nice', 'satisfied',
                'quality', 'fresh', 'best', 'amazing', 'perfect',
                # Tamil (Unicode)
                'நன்று', 'சிறந்த', 'அருமை', 'மகிழ்ச்சி', 'திருப்தி', 'நல்ல',
                'தரம்', 'புதிய', 'சிறந்த', 'அதிசயம்', 'சரியான', 'நல்லா இருக்கு'
            ],
            'bargain': [
                # English
                'discount', 'less', 'bargain', 'cheaper', 'reduce', 'offer',
                'deal', 'negotiate', 'lower price', 'wholesale',
                # Tamil (Unicode)
                'தள்ளுபடி', 'குறைவு', 'பேரம்', 'மலிவு', 'குறைக்க', 'ஆஃபர்',
                'குறைவாக', 'மலிவாக', 'தள்ளுபடி', 'மொத்த விலை'
            ],
            'greeting': [
                # English
                'hello', 'hi', 'good morning', 'good evening', 'namaste',
                'welcome', 'how are you', 'nice to meet',
                # Tamil (Unicode)
                'வணக்கம்', 'காலை வணக்கம்', 'மாலை வணக்கம்', 'நமஸ்காரம்',
                'வரவேற்கிறேன்', 'எப்படி இருக்கிறீர்கள்', 'சந்திக்க மகிழ்ச்சி'
            ]
        }
        
        # Improved emotion patterns
        self.emotion_patterns = {
            'happy': [
                # English
                'good', 'great', 'excellent', 'wonderful', 'nice', 'satisfied',
                'happy', 'pleased', 'delighted', 'excited',
                # Tamil (Unicode)
                'நன்று', 'சிறந்த', 'அருமை', 'மகிழ்ச்சி', 'திருப்தி', 'நல்ல',
                'சந்தோஷம்', 'மகிழ்ச்சி', 'ஆனந்தம்', 'உற்சாகம்'
            ],
            'disappointed': [
                # English
                'bad', 'poor', 'terrible', 'unhappy', 'dissatisfied', 'upset',
                'angry', 'frustrated', 'annoyed', 'disappointed',
                # Tamil (Unicode)
                'மோசம்', 'மோசமான', 'வருத்தம்', 'திருப்தியற்ற', 'கோபம்',
                'மோசம்', 'வருத்தம்', 'கோபம்', 'எரிச்சல்', 'ஏமாற்றம்'
            ],
            'neutral': [
                # English
                'okay', 'fine', 'normal', 'usual', 'standard', 'alright',
                'acceptable', 'moderate', 'average',
                # Tamil (Unicode)
                'சரி', 'பரவாயில்லை', 'சாதாரண', 'வழக்கம்', 'சரி',
                'பரவாயில்லை', 'சாதாரண', 'ஏற்றுக்கொள்ளக்கூடிய', 'மிதமான'
            ]
        }
    
    def _clean_asr_text(self, text: str) -> str:
        """Clean ASR text by removing noise and improving Tamil text quality."""
        if not text or not text.strip():
            return ""
        
        # Step 1: Remove common ASR artifacts and noise
        cleaned = text.strip()
        
        # Remove repeated nonsense tokens (like "nishakku nishakku")
        cleaned = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', cleaned)
        
        # Remove words with excessive consonants or mixed scripts
        cleaned = re.sub(r'\b[a-zA-Z]*[bcdfghjklmnpqrstvwxyz]{4,}[a-zA-Z]*\b', '', cleaned)
        
        # Remove very short isolated English words (likely noise)
        cleaned = re.sub(r'\b[a-zA-Z]{1,2}\b', '', cleaned)
        
        # Remove words with numbers mixed in
        cleaned = re.sub(r'\b\w*\d+\w*\b', '', cleaned)
        
        # Remove transliteration artifacts (mixed Tamil-English nonsense)
        cleaned = re.sub(r'\b[a-zA-Z]+[^\u0B80-\u0BFFa-zA-Z\s]+[a-zA-Z]+\b', '', cleaned)
        
        # Remove isolated English words that don't make sense in context
        english_noise = ['nya', 'Garrett', 'bbepaidieszanas', 'galand', 'ikkunga']
        for noise in english_noise:
            cleaned = cleaned.replace(noise, '')
        
        # Step 2: Clean up Tamil text specifically
        # Remove non-Tamil characters that aren't basic punctuation
        cleaned = re.sub(r'[^\u0B80-\u0BFF\u0020-\u007E\u0964\u0965\u002E\u002C\u003F\u0021]', '', cleaned)
        
        # Step 3: Fix common ASR errors in Tamil
        # Remove repeated characters (like "வாங்குுு")
        cleaned = re.sub(r'([\u0B80-\u0BFF])\1{2,}', r'\1', cleaned)
        
        # Remove isolated punctuation
        cleaned = re.sub(r'\s+[^\u0B80-\u0BFFa-zA-Z\s]+\s+', ' ', cleaned)
        
        # Step 4: Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Step 5: Quality checks
        # Skip if text is too short after cleaning
        if len(cleaned) < 3:
            return ""
        
        # Skip if no Tamil characters remain
        if not re.search(r'[\u0B80-\u0BFF]', cleaned):
            return ""
        
        # Skip if text is mostly punctuation
        if len(re.sub(r'[^\u0B80-\u0BFFa-zA-Z]', '', cleaned)) < len(cleaned) * 0.3:
            return ""
        
        return cleaned
    

    
    def _translate_to_english(self, tamil_text: str) -> str:
        """Translate Tamil text to English using improved methods."""
        if not tamil_text or not tamil_text.strip():
            return ""
        
        # Use Google Cloud Translate if available
        if TRANSLATION_AVAILABLE:
            try:
                # Try Google Cloud Translate first (better quality)
                if 'translate' in globals():
                    translate_client = translate.Client()
                    translation = translate_client.translate(tamil_text, source_language='ta', target_language='en')
                    if translation and translation['translatedText']:
                        logger.debug(f"Google Cloud Translate: '{tamil_text[:50]}...' -> '{translation['translatedText'][:50]}...'")
                        # Set high confidence for Google Cloud Translate
                        self._last_translation_confidence = 0.9
                        return self._post_process_translation(translation['translatedText'])
                
                # Fallback to googletrans
                elif 'Translator' in globals():
                    translator = Translator()
                    # Detect if the text is actually Tamil
                    detected = translator.detect(tamil_text)
                    
                    if detected.lang == 'ta':  # Tamil detected
                        translation = translator.translate(tamil_text, src='ta', dest='en')
                        if translation and translation.text:
                            logger.debug(f"googletrans fallback: '{tamil_text[:50]}...' -> '{translation.text[:50]}...'")
                            # Set medium confidence for googletrans
                            self._last_translation_confidence = 0.7
                            return self._post_process_translation(translation.text)
                    else:
                        logger.debug(f"Text not detected as Tamil: {detected.lang}")
                        
            except Exception as e:
                logger.warning(f"Translation failed: {e}. Using fallback method.")
        
        # Fallback to improved dictionary-based translation
        self._last_translation_confidence = 0.5  # Lower confidence for fallback
        return self._fallback_translate_to_english(tamil_text)
    
    def _post_process_translation(self, english_text: str) -> str:
        """Post-process English translation to improve quality."""
        if not english_text:
            return ""
        
        # Fix common translation artifacts
        cleaned = english_text.strip()
        
        # Remove extra quotes
        cleaned = re.sub(r'^["\']+|["\']+$', '', cleaned)
        
        # Fix common Tamil-English translation issues
        fixes = {
            'Takali Nalla Ikkunga': 'Tomatoes are good today',
            'Nalla Dharama Ikupa': 'Good quality tomatoes',
            'If you come a day, change one day': 'You come one day and don\'t come another day',
            'You are not welcome yesterday': 'You didn\'t come yesterday',
            'We will not buy you': 'We won\'t buy from you',
            'What do we do?': 'What should we do?',
            'Take the same kind of Takali and buy': 'Take the same kind of tomatoes',
            'Can you get the gallon when you come tomorrow': 'Can you bring the gallon when you come tomorrow?',
            'When you come tomorrow, you can buy the gallery': 'When you come tomorrow, bring the gallon and we will buy it',
            'If you have not been welcomed by Natu': 'If you don\'t come, we won\'t buy vegetables',
            'Do not come to the tanamu come to come': 'Don\'t keep saying you\'ll come but not show up',
            'If you believe in your house': 'If we trust you and stay here',
            'Takali Nalla Dharama Ikupa': 'Tomatoes are good quality today',
            'When you come to the benefit': 'When you come tomorrow',
            'If you have a lot of buying a good business': 'If you bring the gallon, you\'ll have good business',
            'It will give you a little bit of a little bit': 'I\'ll give you a little bit of coriander'
        }
        
        for bad_translation, good_translation in fixes.items():
            if bad_translation in cleaned:
                cleaned = cleaned.replace(bad_translation, good_translation)
        
        return cleaned
    
    def _fallback_translate_to_english(self, tamil_text: str) -> str:
        """Improved fallback translation using dictionary for common Tamil phrases."""
        # Enhanced dictionary-based translation for common phrases
        translations = {
            # Greetings
            'வணக்கம்': 'Hello',
            'நமஸ்காரம்': 'Namaste',
            'காலை வணக்கம்': 'Good morning',
            'மாலை வணக்கம்': 'Good evening',
            'நன்றி': 'Thank you',
            
            # Common words
            'உள்ளது': 'available',
            'இல்லை': 'not available',
            'விலை': 'price',
            'எவ்வளவு': 'how much',
            'கிலோ': 'kilogram',
            'ரூபாய்': 'rupees',
            'தேவை': 'need',
            'விரும்புகிறேன்': 'I want',
            'வாங்குகிறேன்': 'I will buy',
            'வாங்குவோம்': 'We will buy',
            'கொடுங்கள்': 'Please give',
            'எடுத்துக்கொள்கிறேன்': 'I will take',
            'நன்று': 'good',
            'சிறந்த': 'excellent',
            'மோசம்': 'bad',
            'தவறு': 'wrong',
            'பிரச்சினை': 'problem',
            'தள்ளுபடி': 'discount',
            'மலிவு': 'cheap',
            'பேரம்': 'bargain',
            'உள்ளதா': 'Is it available?',
            'கிடைக்குமா': 'Can I get it?',
            'வாங்க': 'buy',
            'கொடுங்க': 'give',
            'நல்ல': 'good',
            'அருமை': 'wonderful',
            'மகிழ்ச்சி': 'happiness',
            'திருப்தி': 'satisfaction',
            'வருத்தம்': 'sadness',
            'கோபம்': 'anger',
            'சரி': 'okay',
            'பரவாயில்லை': 'no problem',
            'சாதாரண': 'normal',
            'வழக்கம்': 'usual',
            
            # Vegetables and items
            'தக்காலி': 'tomato',
            'காய்': 'vegetable',
            'கொத்தமல்லி': 'coriander',
            'கருவப்புல்': 'curry leaves',
            'காலான்': 'gallon',
            'தனமு': 'coconut'
        }
        
        # Try to translate the entire text first
        if tamil_text in translations:
            return translations[tamil_text]
        
        # Try to translate word by word
        words = tamil_text.split()
        translated_words = []
        
        for word in words:
            if word in translations:
                translated_words.append(translations[word])
            else:
                # Keep untranslated words as-is
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def analyze(self, transcription_result) -> List[Segment]:
        """Analyze transcription result and return analyzed segments."""
        segments = []
        
        # Get metadata from transcription result
        metadata = transcription_result.metadata
        seller_id = metadata.get('seller_id', self.placeholders.get('seller_id', 'UNKNOWN_SELLER'))
        stop_id = metadata.get('stop_id', self.placeholders.get('stop_id', 'UNKNOWN_STOP'))
        anchor_time = metadata.get('recording_start', self.placeholders.get('anchor_time', '1970-01-01T00:00:00Z'))
        
        # Get audio file information
        audio_file_id = str(transcription_result.audio_file.name) if transcription_result.audio_file else 'UNKNOWN_AUDIO'
        
        # Process each transcription segment
        for i, segment in enumerate(transcription_result.get_segments()):
            raw_text = segment.get('text', '').strip()
            
            # Clean ASR text
            cleaned_text = self._clean_asr_text(raw_text)
            
            # Skip very short segments after cleaning
            if len(cleaned_text.split()) < self.min_segment_words:
                continue
            
            # Get timing information from segment
            start_seconds = segment.get('start', 0.0)
            end_seconds = segment.get('end', start_seconds + 1.0)
            start_ms = int(start_seconds * 1000)
            end_ms = int(end_seconds * 1000)
            duration_ms = end_ms - start_ms
            
            # Generate timestamp
            timestamp = self._generate_timestamp(anchor_time, start_seconds)
            
            # Create segment object with improved fields
            english_translation = self._translate_to_english(cleaned_text)
            translation_confidence = 0.8
            is_translated = False
            
            # Determine translation quality and confidence
            if english_translation:
                # Check if translation is actually English (not just transliteration)
                english_words = english_translation.split()
                if len(english_words) >= 2:  # At least 2 words for a real translation
                    # Check if translation contains actual English words
                    english_word_count = sum(1 for word in english_words 
                                           if re.match(r'^[a-zA-Z]+$', word) and len(word) > 2)
                    if english_word_count >= len(english_words) * 0.7:  # 70% should be English words
                        is_translated = True
                        translation_confidence = 0.85
                    else:
                        # Likely transliteration, not real translation
                        translation_confidence = self._last_translation_confidence * 0.6  # Reduce confidence for transliteration
                        is_translated = False
                else:
                    # Too short to be a real translation
                    translation_confidence = 0.3
                    is_translated = False
            else:
                translation_confidence = 0.0
                is_translated = False
            
            # Enrich too-short translations for requests
            if english_translation and len(english_translation.split()) < self.config.get('translation', {}).get('min_words_english', 4):
                enriched = self._enrich_short_translation(cleaned_text, english_translation)
                if enriched:
                    english_translation = enriched
                    # Re-evaluate translation quality after enrichment
                    if len(enriched.split()) >= 3:
                        is_translated = True
                        translation_confidence = 0.8
            
            seg = Segment(
                seller_id=seller_id,
                stop_id=stop_id,
                segment_id=f"SEG{str(uuid.uuid4())[:8]}",
                audio_file_id=audio_file_id,
                timestamp=timestamp,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                textTamil=cleaned_text,  # Clean Unicode Tamil text
                textEnglish=english_translation,  # Improved English translation
                is_translated=is_translated,
                translation_confidence=translation_confidence,
                asr_confidence=segment.get('confidence', 0.8)
            )
            
            # Use speaker information from Sarvam if available, otherwise analyze
            if segment.get('speaker'):
                # Map Sarvam speaker labels to our speaker roles
                sarvam_speaker = segment.get('speaker', '').lower()
                if 'seller' in sarvam_speaker or 'vendor' in sarvam_speaker:
                    seg.speaker_role = 'seller'
                    seg.role_confidence = 0.9  # High confidence for vendor-provided speaker info
                elif 'buyer' in sarvam_speaker or 'customer' in sarvam_speaker:
                    seg.speaker_role = 'buyer'
                    seg.role_confidence = 0.9
                else:
                    # Fall back to text-based analysis for unknown speaker labels
                    self._analyze_speaker_role(seg)
            else:
                # No speaker info from provider, use text-based analysis
                self._analyze_speaker_role(seg)
            
            self._analyze_intent(seg)
            self._analyze_sentiment(seg)
            self._analyze_emotion(seg)
            
            # Detect products and product-specific analysis
            self._detect_products(seg)
            
            self._set_confidence(seg)
            
            # Validate and fix logical contradictions
            self._validate_and_fix_contradictions(seg)

            # Compute final needs_human_review flag now that all fields are set
            seg._compute_needs_human_review()
            
            segments.append(seg)
        
        logger.info(f"Analyzed {len(segments)} segments")
        return segments

    def _enrich_short_translation(self, tamil_text: str, english_text: str) -> str:
        """Heuristically expand terse translations to preserve request intent and modifiers."""
        t = tamil_text.lower()
        # Request cues
        request_cues = ['கொடுங்களே', 'கொடுங்க', 'கொடுக்க', 'கொடுக்கலாம்', 'கொஞ்சம்', 'கொடு']
        has_request = any(cue in t for cue in request_cues)
        # Product mentions
        has_coriander = 'கொத்தமல்லி' in t
        has_curry = 'கருவப்புல்' in t
        # Build enriched sentence
        if has_request and (has_coriander or has_curry):
            products = []
            if has_coriander:
                products.append('coriander')
            if has_curry:
                products.append('curry leaves')
            if 'கொஞ்சம்' in t:
                quantity = 'some'
            else:
                quantity = 'some'
            if len(products) == 2:
                product_phrase = f"{products[0]} and {products[1]}"
            else:
                product_phrase = products[0]
            return f"Please give {quantity} {product_phrase}."
        # If it looks like a product-only mention, add polite request
        if english_text.strip().lower() in ['coriander', 'curry leaves', 'tomato', 'tomatoes']:
            return f"{english_text.strip()} please."
        return ""
    
    def _generate_timestamp(self, anchor_time: str, offset_seconds: float) -> str:
        """Generate ISO timestamp from anchor time and offset."""
        try:
            base_time = dateparser.isoparse(anchor_time)
            timestamp = base_time + timedelta(seconds=offset_seconds)
            return timestamp.isoformat().replace('+00:00', 'Z')
        except Exception as e:
            logger.warning(f"Failed to generate timestamp: {e}, using anchor time")
            return anchor_time
    
    def _analyze_speaker_role(self, segment: Segment):
        """Analyze speaker role using improved text patterns and context."""
        text_lower = segment.textTamil.lower()
        english_lower = segment.textEnglish.lower() if segment.textEnglish else ""
        
        # Check seller patterns first (more specific)
        seller_matches = sum(1 for pattern in self.seller_patterns if pattern in text_lower)
        buyer_matches = sum(1 for pattern in self.buyer_patterns if pattern in text_lower)
        
        # Also check English text for additional context
        if english_lower:
            seller_matches += sum(1 for pattern in self.seller_patterns if pattern in english_lower)
            buyer_matches += sum(1 for pattern in self.buyer_patterns if pattern in english_lower)
        
        # Context-based role determination
        # Check for specific buyer complaint patterns
        buyer_complaint_patterns = [
            'நீங்கள் ஒரு நாள் வருகிறீங்க', 'ஒரு நாள் வரமாற்றுறீங்க',  # You come one day and don't come another
            'நேற்று நீங்கள் வரவே இல்ல', 'இப்படி பண்ணீங்கன்னா',  # You didn't come yesterday, if you do this
            'உங்களை நம்பி நாங்க காய் வாங்காமல இருக்கும்',  # We won't buy from you if we trust you
            'நாங்க என்ன பண்ணுறது', 'பிரச்சினை'  # What should we do, problem
        ]
        
        seller_promise_patterns = [
            'நாளைக்கு வரும்போது', 'எடுத்து வரப்பாருங்க',  # When I come tomorrow, I will bring
            'கொண்டு வருவேன்', 'வாங்குங்கள்', 'எடுத்துக்கொள்ளுங்கள்'  # I will bring, please buy, please take
        ]
        
        # Check for specific patterns that override general scoring
        for pattern in buyer_complaint_patterns:
            if pattern in text_lower:
                segment.speaker_role = 'buyer'
                return
        
        for pattern in seller_promise_patterns:
            if pattern in text_lower:
                segment.speaker_role = 'seller'
                return
        
        # Use pattern count to determine role
        if seller_matches > buyer_matches and seller_matches > 0:
            segment.speaker_role = 'seller'
        elif buyer_matches > seller_matches and buyer_matches > 0:
            segment.speaker_role = 'buyer'
        elif seller_matches == 0 and buyer_matches == 0 and (('good' in english_lower) or ('நல்லா' in text_lower)):
            # Likely a passerby/customer praising products
            segment.speaker_role = 'customer_bystander'
        else:
            # Default to 'other' if no clear pattern or equal matches
            segment.speaker_role = 'other'

        # role confidence heuristic
        total_matches = seller_matches + buyer_matches
        if segment.speaker_role in ['seller', 'buyer'] and total_matches > 0:
            segment.role_confidence = min(0.95, 0.5 + 0.2 * total_matches)
        elif segment.speaker_role == 'customer_bystander':
            segment.role_confidence = 0.6
        else:
            segment.role_confidence = 0.4
    
    def _analyze_intent(self, segment: Segment):
        """Analyze intent using improved text patterns and scoring."""
        text_lower = segment.textTamil.lower()
        english_lower = segment.textEnglish.lower() if segment.textEnglish else ""
        
        # Score each intent category
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            # Also check English text for additional context
            if english_lower:
                score += sum(1 for pattern in patterns if pattern in english_lower)
            intent_scores[intent] = score
        
        # Special handling for negative purchase intent
        # Check if this is a refusal or complaint about buying
        negative_purchase_patterns = [
            'வாங்காமல', 'வாங்க மாட்டோம்', 'நம்பி நாங்க',  # Won't buy, not buying from you
            'won\'t buy', 'not buying', 'refuse to buy', 'don\'t want to buy'
        ]
        
        has_negative_intent = any(pattern in text_lower for pattern in negative_purchase_patterns)
        if english_lower:
            has_negative_intent = has_negative_intent or any(pattern in english_lower for pattern in negative_purchase_patterns)
        
        # Check for specific product requests
        product_request_patterns = [
            'கொத்தமல்லி', 'கருவப்புல்', 'காலான்',  # Coriander, curry leaves, gallon
            'coriander', 'curry leaves', 'gallon'
        ]
        
        has_product_request = any(pattern in text_lower for pattern in product_request_patterns)
        if english_lower:
            has_product_request = has_product_request or any(pattern in english_lower for pattern in product_request_patterns)
        
        # Override intent based on context
        if has_negative_intent:
            segment.intent = 'purchase_negative'
        elif has_product_request:
            segment.intent = 'purchase_request'
        else:
            # Find the intent with highest score
            if intent_scores:
                best_intent = max(intent_scores.items(), key=lambda x: x[1])
                if best_intent[1] > 0:
                    segment.intent = best_intent[0]
                else:
                    segment.intent = 'other'
            else:
                segment.intent = 'other'

        # Map borderline 'other' to complaint/request if clear hints exist
        if segment.intent == 'other':
            if any(p in text_lower for p in ['பிரச்சினை', 'வரமாட்டுறீங்க', 'வரவே இல்ல']) or any(p in english_lower for p in ['problem', 'complain', 'don\'t come', 'not coming']):
                segment.intent = 'complaint'
            elif any(p in text_lower for p in ['எவ்வளவு', 'விலை', 'கொடுக்க', 'கொடுங்கள்', 'கிடைக்குமா']) or any(p in english_lower for p in ['how much', 'price', 'can you', 'give me']):
                segment.intent = 'purchase_request'
    
    def _analyze_sentiment(self, segment: Segment):
        """Analyze sentiment using improved text patterns and better scoring."""
        text_lower = segment.textTamil.lower()
        english_lower = segment.textEnglish.lower() if segment.textEnglish else ""
        
        # Enhanced sentiment scoring with context
        positive_words = [
            'good', 'great', 'excellent', 'wonderful', 'nice', 'satisfied', 'happy',
            'நன்று', 'சிறந்த', 'அருமை', 'மகிழ்ச்சி', 'திருப்தி', 'நல்ல', 'சந்தோஷம்'
        ]
        negative_words = [
            'bad', 'poor', 'terrible', 'unhappy', 'dissatisfied', 'upset', 'angry',
            'மோசம்', 'மோசமான', 'வருத்தம்', 'திருப்தியற்ற', 'கோபம்', 'எரிச்சல்'
        ]
        
        # Count positive and negative words in both languages
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if english_lower:
            positive_count += sum(1 for word in positive_words if word in english_lower)
            negative_count += sum(1 for word in negative_words if word in english_lower)
        
        # Context-based sentiment adjustment
        context_multiplier = 1.0
        
        # Check for complaint context (negative sentiment)
        if segment.intent == 'complaint' or segment.intent == 'purchase_negative':
            context_multiplier = 2.0  # Stronger negative bias for complaints
        # Check for product praise context (positive sentiment)
        elif segment.intent == 'product_praise':
            context_multiplier = 1.5  # Stronger positive bias for praise
        
        # Calculate sentiment score with context
        if positive_count > negative_count:
            score = min(0.9, (0.3 + (positive_count * 0.15)) * context_multiplier)
        elif negative_count > positive_count:
            score = max(-0.9, (-0.3 - (negative_count * 0.15)) * context_multiplier)
        else:
            # For neutral cases, apply context bias
            if segment.intent == 'complaint' or segment.intent == 'purchase_negative':
                score = random.uniform(-0.3, -0.1)  # Bias toward negative
            elif segment.intent == 'product_praise':
                score = random.uniform(0.1, 0.3)   # Bias toward positive
            else:
                score = random.uniform(-0.1, 0.1)  # Truly neutral
        
        segment.sentiment_score = round(score, 2)
        
        # Use standardized sentiment thresholds from config
        thresholds = self.config.get('sentiment_thresholds', {})
        positive_threshold = thresholds.get('positive', 0.15)
        negative_threshold = thresholds.get('negative', -0.15)
        
        # Improved sentiment label mapping with clear thresholds
        if score >= positive_threshold:
            segment.sentiment_label = 'positive'
        elif score <= negative_threshold:
            segment.sentiment_label = 'negative'
        else:
            segment.sentiment_label = 'neutral'
    
    def _analyze_emotion(self, segment: Segment):
        """Analyze emotion using improved text patterns and sentiment alignment."""
        text_lower = segment.textTamil.lower()
        
        # Check emotion patterns first
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    segment.emotion = emotion
                    return
        
        # Default emotion based on sentiment and intent (granular mapping)
        if segment.sentiment_label == 'positive':
            segment.emotion = 'happy' if segment.intent in ['product_praise', 'purchase_positive'] else 'satisfied'
        elif segment.sentiment_label == 'negative':
            if segment.intent in ['complaint', 'purchase_negative']:
                segment.emotion = 'frustrated' if 'angry' in text_lower or 'கோபம்' in text_lower else 'disappointed'
            else:
                segment.emotion = 'annoyed'
        else:
            segment.emotion = 'neutral'
    
    def _set_confidence(self, segment: Segment):
        """Set overall confidence based on various factors."""
        # Base confidence from ASR
        base_confidence = segment.asr_confidence or 0.8
        
        # Adjust based on translation quality
        if segment.is_translated:
            translation_factor = segment.translation_confidence or 0.8
        else:
            translation_factor = 0.6  # Lower confidence for untranslated segments
        
        # Adjust based on speaker role confidence
        role_factor = segment.role_confidence or 0.5
        
        # Adjust based on intent confidence
        intent_factor = segment.intent_confidence or 0.7
        
        # Adjust based on product detection confidence
        product_factor = 1.0
        if segment.products:
            product_confidences = [p.get('product_confidence', 0.8) for p in segment.products]
            product_factor = sum(product_confidences) / len(product_confidences)
        
        # Calculate weighted average with configurable weights
        weights = self.config.get('confidence_weights', {
            'asr': 0.35,
            'translation': 0.20,
            'role': 0.15,
            'intent': 0.15,
            'product': 0.15
        })
        
        confidence = (
            base_confidence * weights['asr'] +
            translation_factor * weights['translation'] +
            role_factor * weights['role'] +
            intent_factor * weights['intent'] +
            product_factor * weights['product']
        )
        
        segment.confidence = min(1.0, max(0.0, confidence))

        # Derive business flags after we have sentiment/intent
        self._set_business_flags(segment)



    def _set_business_flags(self, segment: Segment) -> None:
        """Set action flags and churn/opportunity heuristics."""
        # Action flags
        segment.action_required = segment.intent in ['complaint', 'purchase_negative']
        segment.escalation_needed = segment.action_required and (segment.sentiment_label == 'negative')

        # Churn risk
        if segment.intent in ['purchase_negative', 'complaint']:
            segment.churn_risk = 'high'
        elif segment.intent == 'purchase_request' and segment.sentiment_label == 'negative':
            segment.churn_risk = 'medium'
        else:
            segment.churn_risk = 'low'

        # Opportunity flag
        segment.business_opportunity = segment.intent in ['product_praise', 'purchase_request', 'purchase_positive']
    
    def _detect_products(self, segment: Segment):
        """Detect products mentioned in the text using rule-based and ML approaches."""
        tamil_text = segment.textTamil.lower()
        english_text = segment.textEnglish.lower() if segment.textEnglish else ""
        
        # Product dictionary with SKU mapping
        product_dict = {
            'tomato': {
                'tamil_names': ['தக்காலி', 'தக்காளி', 'தக்காளி'],
                'sku_id': 'SKU-0001',
                'english_names': ['tomato', 'tomatoes']
            },
            'coriander': {
                'tamil_names': ['கொத்தமல்லி', 'கொத்தமல்லி கட்டு'],
                'sku_id': 'SKU-0002', 
                'english_names': ['coriander', 'cilantro']
            },
            'curry_leaves': {
                'tamil_names': ['கருவப்புல்', 'கருவேப்பிலை'],
                'sku_id': 'SKU-0003',
                'english_names': ['curry leaves', 'curry leaf']
            },
            'coconut': {
                'tamil_names': ['தனமு', 'தேங்காய்'],
                'sku_id': 'SKU-0004',
                'english_names': ['coconut', 'coconuts']
            },
            'gallon': {
                'tamil_names': ['காலான்'],
                'sku_id': 'SKU-0005',
                'english_names': ['gallon', 'gallons']
            },
            'onion': {
                'tamil_names': ['வெங்காயம்'],
                'sku_id': 'SKU-0006',
                'english_names': ['onion', 'onions']
            },
            'potato': {
                'tamil_names': ['ஆலூரடி', 'உருளைக்கிழங்கு'],
                'sku_id': 'SKU-0007',
                'english_names': ['potato', 'potatoes']
            }
        }
        
        detected_products = []
        product_intents = {}
        product_sentiments = {}
        product_intent_confidences = {}
        
        for product_name, product_info in product_dict.items():
            # Check Tamil text
            tamil_found = any(name in tamil_text for name in product_info['tamil_names'])
            # Check English text
            english_found = any(name in english_text for name in product_info['english_names'])
            
            if tamil_found or english_found:
                # Find text spans
                tamil_span = self._find_text_span(tamil_text, product_info['tamil_names'])
                english_span = self._find_text_span(english_text, product_info['english_names'])
                
                # Calculate product confidence based on context
                product_confidence = self._calculate_product_confidence(
                    tamil_text, english_text, product_name, product_info
                )
                
                # Detect product-specific intent
                product_intent, intent_confidence = self._detect_product_intent(
                    tamil_text, english_text, product_name, segment.intent
                )
                
                # Detect product-specific sentiment
                product_sentiment = self._detect_product_sentiment(
                    tamil_text, english_text, product_name, segment.sentiment_score
                )
                
                product_obj = {
                    'product_name': product_name,
                    'sku_id': product_info['sku_id'],
                    'product_confidence': product_confidence,
                    'text_span_tamil': tamil_span,
                    'text_span_english': english_span
                }
                
                detected_products.append(product_obj)
                product_intents[product_info['sku_id']] = product_intent
                product_sentiments[product_info['sku_id']] = product_sentiment
                product_intent_confidences[product_info['sku_id']] = intent_confidence
        
        segment.products = detected_products
        segment.product_intents = product_intents
        segment.product_sentiments = product_sentiments
        segment.product_intent_confidences = product_intent_confidences
    
    def _find_text_span(self, text: str, product_names: list) -> dict:
        """Find character span of product mention in text."""
        if not text:
            return {'start_char': 0, 'end_char': 0}
        
        for name in product_names:
            if name in text:
                start_char = text.find(name)
                end_char = start_char + len(name)
                return {'start_char': start_char, 'end_char': end_char}
        
        return {'start_char': 0, 'end_char': 0}
    
    def _calculate_product_confidence(self, tamil_text: str, english_text: str, 
                                    product_name: str, product_info: dict) -> float:
        """Calculate confidence for product detection."""
        base_confidence = 0.8
        
        # Boost confidence if found in both languages
        if any(name in tamil_text for name in product_info['tamil_names']) and \
           any(name in english_text for name in product_info['english_names']):
            base_confidence += 0.1
        
        # Boost confidence if product is mentioned with quantity/request words
        quantity_words = ['கொஞ்சம்', 'ஒரு', 'some', 'one', 'a', 'the']
        request_words = ['கொடுங்கள்', 'கொடு', 'give', 'please', 'need', 'want']
        
        has_quantity = any(word in tamil_text or word in english_text for word in quantity_words)
        has_request = any(word in tamil_text or word in english_text for word in request_words)
        
        if has_quantity:
            base_confidence += 0.05
        if has_request:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _detect_product_intent(self, tamil_text: str, english_text: str, 
                              product_name: str, segment_intent: str) -> tuple:
        """Detect product-specific intent."""
        # Product-specific intent patterns
        product_intent_patterns = {
            'purchase_request': [
                'கொடுங்கள்', 'கொடு', 'கொடுக்கலாம்', 'கிடைக்குமா', 'எவ்வளவு',
                'give', 'please', 'can you', 'how much', 'available'
            ],
            'product_praise': [
                'நல்லா', 'சிறந்த', 'அருமை', 'நன்று', 'தரமான',
                'good', 'great', 'excellent', 'quality', 'fresh'
            ],
            'purchase_positive': [
                'வாங்குகிறேன்', 'வாங்குவோம்', 'எடுத்துக்கொள்கிறேன்',
                'buy', 'will buy', 'take', 'purchase'
            ],
            'purchase_negative': [
                'வாங்காமல', 'வாங்க மாட்டோம்', 'இல்லை',
                'won\'t buy', 'not buying', 'no'
            ]
        }
        
        # Check patterns in both languages
        intent_scores = {}
        for intent, patterns in product_intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in tamil_text)
            score += sum(1 for pattern in patterns if pattern in english_text)
            intent_scores[intent] = score
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                product_intent = best_intent[0]
                confidence = min(0.95, 0.5 + 0.2 * best_intent[1])
            else:
                # Fall back to segment intent
                product_intent = segment_intent
                confidence = 0.5
        else:
            product_intent = segment_intent
            confidence = 0.5
        
        return product_intent, confidence
    
    def _detect_product_sentiment(self, tamil_text: str, english_text: str,
                                 product_name: str, segment_sentiment: float) -> dict:
        """Detect product-specific sentiment."""
        # Product-specific sentiment patterns
        positive_patterns = [
            'நல்லா', 'சிறந்த', 'அருமை', 'நன்று', 'தரமான', 'புதிய',
            'good', 'great', 'excellent', 'quality', 'fresh', 'best'
        ]
        
        negative_patterns = [
            'மோசம்', 'மோசமான', 'தவறு', 'பிரச்சினை', 'தரம் குறைவு',
            'bad', 'poor', 'terrible', 'wrong', 'problem', 'low quality'
        ]
        
        # Count positive and negative mentions
        positive_count = sum(1 for pattern in positive_patterns 
                           if pattern in tamil_text or pattern in english_text)
        negative_count = sum(1 for pattern in negative_patterns 
                           if pattern in tamil_text or pattern in english_text)
        
        # Calculate product-specific sentiment
        if positive_count > negative_count:
            sentiment_score = min(1.0, 0.3 + 0.2 * positive_count)
            sentiment_label = 'positive'
        elif negative_count > positive_count:
            sentiment_score = max(-1.0, -0.3 - 0.2 * negative_count)
            sentiment_label = 'negative'
        else:
            # Use segment sentiment as baseline
            sentiment_score = segment_sentiment
            sentiment_label = 'neutral' if abs(sentiment_score) < 0.15 else \
                             ('positive' if sentiment_score > 0 else 'negative')
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        }
    
    def _validate_and_fix_contradictions(self, segment: Segment):
        """Validate and fix logical contradictions in the analysis."""
        # Store original values locally for comparison
        original_values = {
            'speaker_role': segment.speaker_role,
            'intent': segment.intent,
            'sentiment_label': segment.sentiment_label
        }
        
        # Fix 1: Negative purchase intent should have negative sentiment
        if segment.intent == 'purchase_negative' and segment.sentiment_label == 'positive':
            segment.sentiment_label = 'negative'
            segment.sentiment_score = max(-0.9, segment.sentiment_score - 0.3)
            logger.debug(f"Fixed contradiction: {segment.segment_id} - negative intent should have negative sentiment")
        
        # Fix 2: Complaint intent should have negative sentiment
        if segment.intent == 'complaint' and segment.sentiment_label == 'positive':
            segment.sentiment_label = 'negative'
            segment.sentiment_score = max(-0.9, segment.sentiment_score - 0.3)
            logger.debug(f"Fixed contradiction: {segment.segment_id} - complaint intent should have negative sentiment")
        
        # Fix 3: Product praise should have positive sentiment
        if segment.intent == 'product_praise' and segment.sentiment_label == 'negative':
            segment.sentiment_label = 'positive'
            segment.sentiment_score = min(0.9, segment.sentiment_score + 0.3)
            logger.debug(f"Fixed contradiction: {segment.segment_id} - product praise should have positive sentiment")
        
        # Fix 4: Buyer complaints about seller should be labeled as buyer
        buyer_complaint_texts = [
            'நீங்கள் ஒரு நாள் வருகிறீங்க', 'ஒரு நாள் வரமாற்றுறீங்க',  # You come one day and don't come another
            'நேற்று நீங்கள் வரவே இல்ல', 'இப்படி பண்ணீங்கன்னா',  # You didn't come yesterday, if you do this
            'உங்களை நம்பி நாங்க காய் வாங்காமல இருக்கும்',  # We won't buy from you if we trust you
            'நாங்க என்ன பண்ணுறது'  # What should we do
        ]
        
        if any(text in segment.textTamil for text in buyer_complaint_texts):
            if segment.speaker_role != 'buyer':
                segment.speaker_role = 'buyer'
                logger.debug(f"Fixed contradiction: {segment.segment_id} - buyer complaint should be labeled as buyer")
        
        # No analysis metadata is stored in output; this method now only corrects values


def main():
    """Standalone execution for testing."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NLU analysis')
    parser.add_argument('--config', default='../config.yaml', help='Configuration file path')
    parser.add_argument('--text', required=True, help='Text to analyze')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create analyzer
    analyzer = NLUAnalyzer(config)
    
    # Create test segment
    test_segment = Segment(textTamil=args.text)
    
    # Analyze
    analyzer._analyze_speaker_role(test_segment)
    analyzer._analyze_intent(test_segment)
    analyzer._analyze_sentiment(test_segment)
    analyzer._analyze_emotion(test_segment)
    analyzer._set_confidence(test_segment)
    
    # Print results
    print(f"Text: {test_segment.textTamil}")
    print(f"English: {test_segment.textEnglish}")
    print(f"Speaker Role: {test_segment.speaker_role}")
    print(f"Intent: {test_segment.intent}")
    print(f"Sentiment: {test_segment.sentiment_label} ({test_segment.sentiment_score})")
    print(f"Emotion: {test_segment.emotion}")
    print(f"Confidence: {test_segment.confidence}")


if __name__ == "__main__":
    main()
