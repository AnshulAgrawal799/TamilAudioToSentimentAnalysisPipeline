from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from analyze import NLUAnalyzer  # type: ignore
from transcriber_factory import UnifiedTranscriptionResult  # type: ignore


def make_result(text: str):
    res = UnifiedTranscriptionResult(
        Path('data/tmp/Recording1.wav'),
        {'seller_id': 'S123', 'stop_id': 'STOP45', 'recording_start': '2025-08-19T00:00:00Z'},
        'sarvam', 'saarika:v2.5')
    res.add_segment(0.0, 2.0, text, confidence=0.9)
    return res


def test_analyze_purchase_request_intent():
    analyzer = NLUAnalyzer({'placeholders': {}})
    res = make_result('கொத்தமல்லி கொடுங்கள்')  # Please give coriander
    segs = analyzer.analyze(res)
    assert len(segs) >= 1
    s = segs[0]
    assert s.intent in ('purchase_request', 'purchase_positive')
    assert s.textEnglish and isinstance(s.textEnglish, str)
    assert s.is_translated is True


def test_analyze_sentiment_and_emotion_alignment():
    analyzer = NLUAnalyzer({'placeholders': {}})
    res = make_result('மோசம், வாங்க மாட்டோம்')  # bad, we will not buy
    segs = analyzer.analyze(res)
    assert len(segs) >= 1
    s = segs[0]
    assert s.sentiment_label in ('negative', 'neutral')
    assert s.intent in ('purchase_negative', 'complaint')


