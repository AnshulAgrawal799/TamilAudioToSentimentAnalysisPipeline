#!/usr/bin/env python3
"""
Tests for needs_human_review field logic on Segment objects.
"""

import sys
import os
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyze import Segment


def expect(condition: bool, message: str) -> bool:
    if not condition:
        print(f"❌ {message}")
        return False
    print(f"✅ {message}")
    return True


def test_default_false():
    seg = Segment(
        textTamil="வணக்கம்",
        textEnglish="Hello",
        asr_confidence=0.9,
        translation_confidence=0.9,
        sentiment_label='neutral',
        churn_risk='low',
        role_confidence=0.4,
    )
    d = seg.to_dict()
    return expect(d.get('needs_human_review') is False, "Default needs_human_review is False")


def test_asr_low_triggers():
    seg = Segment(
        asr_confidence=0.6,  # below 0.65
        translation_confidence=0.9,
        sentiment_label='neutral',
        churn_risk='low',
        role_confidence=0.4,
    )
    return expect(seg.to_dict().get('needs_human_review') is True, "ASR < 0.65 triggers review")


def test_translation_low_triggers():
    seg = Segment(
        asr_confidence=0.9,
        translation_confidence=0.5,  # below 0.65
        sentiment_label='neutral',
        churn_risk='low',
        role_confidence=0.4,
    )
    return expect(seg.to_dict().get('needs_human_review') is True, "Translation < 0.65 triggers review")


def test_risk_combo_triggers():
    seg = Segment(
        asr_confidence=0.9,
        translation_confidence=0.9,
        sentiment_label='negative',
        churn_risk='high',
        role_confidence=0.5,  # >= 0.5
    )
    return expect(seg.to_dict().get('needs_human_review') is True, "Negative+high churn+role_conf>=0.5 triggers review")


def test_risk_combo_not_trigger_with_low_role_conf():
    seg = Segment(
        asr_confidence=0.9,
        translation_confidence=0.9,
        sentiment_label='negative',
        churn_risk='high',
        role_confidence=0.49,  # below 0.5
    )
    return expect(seg.to_dict().get('needs_human_review') is False, "Role confidence < 0.5 does not trigger risk rule")


def main():
    tests = [
        test_default_false,
        test_asr_low_triggers,
        test_translation_low_triggers,
        test_risk_combo_triggers,
        test_risk_combo_not_trigger_with_low_role_conf,
    ]
    passed = 0
    for t in tests:
        if t():
            passed += 1
    print(f"Human-review flag tests: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())


