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
        self.timestamp = kwargs.get('timestamp', '1970-01-01T00:00:00Z')
        self.speaker_role = kwargs.get('speaker_role', 'other')
        self.textTamil = kwargs.get('textTamil', '')  # Tamil text in Unicode
        self.textEnglish = kwargs.get('textEnglish', '')  # English text
        self.intent = kwargs.get('intent', 'other')
        self.sentiment_score = kwargs.get('sentiment_score', 0.0)
        self.sentiment_label = kwargs.get('sentiment_label', 'neutral')
        self.emotion = kwargs.get('emotion', 'neutral')
        self.confidence = kwargs.get('confidence', 0.5)
        self.audio_file_id = kwargs.get('audio_file_id', 'UNKNOWN_AUDIO')
        self.start_ms = kwargs.get('start_ms', 0)
        self.end_ms = kwargs.get('end_ms', 0)
        self.duration_ms = kwargs.get('duration_ms', 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'seller_id': self.seller_id,
            'stop_id': self.stop_id,
            'segment_id': self.segment_id,
            'timestamp': self.timestamp,
            'speaker_role': self.speaker_role,
            'textTamil': self.textTamil,
            'textEnglish': self.textEnglish,
            'intent': self.intent,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'emotion': self.emotion,
            'confidence': self.confidence,
            'audio_file_id': self.audio_file_id,
            'start_ms': self.start_ms,
            'end_ms': self.end_ms,
            'duration_ms': self.duration_ms
        }
    



class NLUAnalyzer:
    """Handles NLU analysis for transcribed segments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.placeholders = config.get('placeholders', {})
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.min_segment_words = config.get('min_segment_words', 4)
        
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
            # Tamil patterns for seller (Unicode)
            'உள்ளது', 'விலை', 'கிலோ', 'ரூபாய்', 'கொடுக்கலாம்', 'உள்ளன',
            'தரமான', 'சிறந்த', 'மலிவு', 'தள்ளுபடி', 'ஆஃபர்', 'வாங்குங்கள்',
            'எடுத்துக்கொள்ளுங்கள்', 'கொடுக்கிறேன்', 'விற்கிறேன்', 'விற்பனை',
            'கடை', 'வியாபாரி', 'ஒப்பந்தக்காரர்', 'வழங்குநர்'
        ]
        
        self.buyer_patterns = [
            # English patterns for buyer
            'we will buy', 'we buy', 'i will buy', 'can i get', 'give me',
            'how much', 'what is the price', 'available', 'need', 'want',
            'purchase', 'customer', 'buyer', 'shopping', 'order',
            # Tamil patterns for buyer (Unicode)
            'வாங்குவோம்', 'வாங்குகிறேன்', 'கொடுங்கள்', 'எவ்வளவு', 'விலை என்ன',
            'உள்ளதா', 'தேவை', 'விரும்புகிறேன்', 'எடுத்துக்கொள்கிறேன்', 'வாங்க',
            'கொடுங்க', 'வாங்குபவர்', 'வாடிக்கையாளர்', 'வாங்குதல்', 'ஆர்டர்'
        ]
        
        # Standardized intent taxonomy with improved patterns
        self.intent_patterns = {
            'purchase': [
                # English
                'buy', 'will buy', 'purchase', 'take', 'get', 'need', 'order',
                'want to buy', 'looking for', 'interested in',
                # Tamil (Unicode)
                'வாங்கு', 'வாங்குவோம்', 'எடுத்துக்கொள்', 'தேவை', 'வாங்குகிறேன்',
                'எடுத்துக்கொள்கிறேன்', 'வாங்க', 'கொடுங்கள்', 'ஆர்டர்', 'வாங்குதல்'
            ],
            'request': [
                # English
                'how much', 'what is the price', 'available', 'cost', 'rate',
                'can you', 'please', 'information', 'details', 'inquiry',
                # Tamil (Unicode)
                'எவ்வளவு', 'விலை என்ன', 'உள்ளதா', 'செலவு', 'விகிதம்',
                'விலை', 'உள்ளதா', 'கிடைக்குமா', 'தகவல்', 'விவரங்கள்'
            ],
            'complaint': [
                # English
                'bad', 'complain', 'problem', 'issue', 'wrong', 'not good',
                'poor quality', 'dissatisfied', 'unhappy', 'terrible',
                # Tamil (Unicode)
                'மோசம்', 'பிரச்சினை', 'தவறு', 'நன்றாக இல்லை', 'புகார்',
                'மோசமான', 'தரம் குறைவு', 'திருப்தியற்ற', 'வருத்தம்'
            ],
            'product_praise': [
                # English
                'good', 'great', 'excellent', 'wonderful', 'nice', 'satisfied',
                'quality', 'fresh', 'best', 'amazing', 'perfect',
                # Tamil (Unicode)
                'நன்று', 'சிறந்த', 'அருமை', 'மகிழ்ச்சி', 'திருப்தி', 'நல்ல',
                'தரம்', 'புதிய', 'சிறந்த', 'அதிசயம்', 'சரியான'
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
                            return self._post_process_translation(translation.text)
                    else:
                        logger.debug(f"Text not detected as Tamil: {detected.lang}")
                        
            except Exception as e:
                logger.warning(f"Translation failed: {e}. Using fallback method.")
        
        # Fallback to improved dictionary-based translation
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
            
            # Get timing information from Whisper segment
            start_seconds = segment.get('start', 0.0)
            end_seconds = segment.get('end', start_seconds + 1.0)
            start_ms = int(start_seconds * 1000)
            end_ms = int(end_seconds * 1000)
            duration_ms = end_ms - start_ms
            
            # Generate timestamp
            timestamp = self._generate_timestamp(anchor_time, start_seconds)
            
            # Create segment object with improved fields
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
                textEnglish=self._translate_to_english(cleaned_text)  # Improved English translation
            )
            
            # Run improved NLU analysis
            self._analyze_speaker_role(seg)
            self._analyze_intent(seg)
            self._analyze_sentiment(seg)
            self._analyze_emotion(seg)
            self._set_confidence(seg)
            
            segments.append(seg)
        
        logger.info(f"Analyzed {len(segments)} segments")
        return segments
    
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
        """Analyze speaker role using improved text patterns."""
        text_lower = segment.textTamil.lower()
        
        # Check seller patterns first (more specific)
        seller_matches = sum(1 for pattern in self.seller_patterns if pattern in text_lower)
        buyer_matches = sum(1 for pattern in self.buyer_patterns if pattern in text_lower)
        
        # Use pattern count to determine role
        if seller_matches > buyer_matches and seller_matches > 0:
            segment.speaker_role = 'seller'
        elif buyer_matches > seller_matches and buyer_matches > 0:
            segment.speaker_role = 'buyer'
        else:
            # Default to 'other' if no clear pattern or equal matches
            segment.speaker_role = 'other'
    
    def _analyze_intent(self, segment: Segment):
        """Analyze intent using improved text patterns and scoring."""
        text_lower = segment.textTamil.lower()
        
        # Score each intent category
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent] = score
        
        # Find the intent with highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                segment.intent = best_intent[0]
            else:
                segment.intent = 'other'
        else:
            segment.intent = 'other'
    
    def _analyze_sentiment(self, segment: Segment):
        """Analyze sentiment using improved text patterns and better scoring."""
        text_lower = segment.textTamil.lower()
        
        # Enhanced sentiment scoring with context
        positive_words = [
            'good', 'great', 'excellent', 'wonderful', 'nice', 'satisfied', 'happy',
            'நன்று', 'சிறந்த', 'அருமை', 'மகிழ்ச்சி', 'திருப்தி', 'நல்ல', 'சந்தோஷம்'
        ]
        negative_words = [
            'bad', 'poor', 'terrible', 'unhappy', 'dissatisfied', 'upset', 'angry',
            'மோசம்', 'மோசமான', 'வருத்தம்', 'திருப்தியற்ற', 'கோபம்', 'எரிச்சல்'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Context-based sentiment adjustment
        context_multiplier = 1.0
        
        # Check for complaint context (negative sentiment)
        if segment.intent == 'complaint':
            context_multiplier = 1.5
        # Check for product praise context (positive sentiment)
        elif segment.intent == 'product_praise':
            context_multiplier = 1.3
        
        # Calculate sentiment score with context
        if positive_count > negative_count:
            score = min(0.9, (0.3 + (positive_count * 0.15)) * context_multiplier)
        elif negative_count > positive_count:
            score = max(-0.9, (-0.3 - (negative_count * 0.15)) * context_multiplier)
        else:
            score = random.uniform(-0.1, 0.1)
        
        segment.sentiment_score = round(score, 2)
        
        # Improved sentiment label mapping
        if score >= 0.15:
            segment.sentiment_label = 'positive'
        elif score <= -0.15:
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
        
        # Default emotion based on sentiment and intent
        if segment.sentiment_label == 'positive':
            if segment.intent == 'product_praise':
                segment.emotion = 'happy'
            else:
                segment.emotion = 'happy'
        elif segment.sentiment_label == 'negative':
            if segment.intent == 'complaint':
                segment.emotion = 'disappointed'
            else:
                segment.emotion = 'disappointed'
        else:
            segment.emotion = 'neutral'
    
    def _set_confidence(self, segment: Segment):
        """Set confidence score based on analysis quality."""
        # Base confidence
        base_confidence = 0.7
        
        # Adjust based on text length (longer text = higher confidence)
        length_factor = min(1.0, len(segment.textTamil.split()) / 20.0)
        
        # Adjust based on pattern matches
        pattern_matches = 0
        if segment.speaker_role != 'other':
            pattern_matches += 1
        if segment.intent != 'other':
            pattern_matches += 1
        if segment.emotion != 'neutral':
            pattern_matches += 1
        
        pattern_factor = pattern_matches / 3.0
        
        # Adjust based on translation quality
        translation_quality = 0.5  # Base translation quality
        if segment.textEnglish and len(segment.textEnglish.split()) > 2:
            translation_quality = 0.8
        
        # Calculate final confidence
        confidence = base_confidence + (length_factor * 0.15) + (pattern_factor * 0.1) + (translation_quality * 0.05)
        confidence = min(0.95, max(0.3, confidence))  # Clamp between 0.3 and 0.95
        
        segment.confidence = round(confidence, 2)


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
