#!/usr/bin/env python3
"""
Aggregation module for generating stop-level and day-level summaries.
Combines analyzed segments into aggregated JSON outputs.
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class Aggregator:
    """Handles aggregation of analyzed segments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.placeholders = config.get('placeholders', {})
    
    def _calculate_audio_file_stats(self, segments: List) -> Dict[str, Any]:
        """Calculate audio file statistics for segments."""
        audio_files = {}
        
        for segment in segments:
            audio_file_id = getattr(segment, 'audio_file_id', 'UNKNOWN_AUDIO')
            if audio_file_id not in audio_files:
                audio_files[audio_file_id] = {
                    'n_segments': 0,
                    'total_duration_ms': 0,
                    'time_range': {'start_ms': float('inf'), 'end_ms': 0}
                }
            
            audio_files[audio_file_id]['n_segments'] += 1
            audio_files[audio_file_id]['total_duration_ms'] += getattr(segment, 'duration_ms', 0)
            
            start_ms = getattr(segment, 'start_ms', 0)
            end_ms = getattr(segment, 'end_ms', 0)
            
            if start_ms < audio_files[audio_file_id]['time_range']['start_ms']:
                audio_files[audio_file_id]['time_range']['start_ms'] = start_ms
            if end_ms > audio_files[audio_file_id]['time_range']['end_ms']:
                audio_files[audio_file_id]['time_range']['end_ms'] = end_ms
        
        # Convert inf values to 0 for JSON serialization
        for audio_file in audio_files.values():
            if audio_file['time_range']['start_ms'] == float('inf'):
                audio_file['time_range']['start_ms'] = 0
        
        return audio_files

    def aggregate_by_stop(self, segments: List) -> Dict[str, Dict[str, Any]]:
        """Aggregate segments by stop ID."""
        stop_aggregates = {}
        
        # Group segments by stop
        stops = defaultdict(list)
        for segment in segments:
            stop_id = segment.stop_id
            stops[stop_id].append(segment)
        
        # Generate aggregate for each stop
        for stop_id, stop_segments in stops.items():
            if not stop_segments:
                continue
                
            # Get date from first segment
            date_str = self._extract_date(stop_segments[0].timestamp)
            seller_id = stop_segments[0].seller_id
            
            # Calculate metrics
            n_segments = len(stop_segments)
            n_calls = self._estimate_call_count(stop_segments)
            avg_sentiment = self._calculate_avg_sentiment(stop_segments)
            sentiment_dist = self._calculate_sentiment_distribution(stop_segments)
            dominant_emotion = self._calculate_dominant_emotion(stop_segments)
            top_intents = self._calculate_top_intents(stop_segments)
            audio_file_stats = self._calculate_audio_file_stats(stop_segments)
            
            # Get placeholder data
            sales_data = self.placeholders.get('sales_at_stop', {})
            inventory_data = self.placeholders.get('inventory_after_sale', {})
            
            # Create aggregate
            stop_aggregate = {
                'seller_id': seller_id,
                'stop_id': stop_id,
                'date': date_str,
                'n_segments': n_segments,
                'n_calls': n_calls,
                'avg_sentiment_score': avg_sentiment,
                'sentiment_distribution': sentiment_dist,
                'dominant_emotion': dominant_emotion,
                'top_intents': top_intents,
                'audio_files': audio_file_stats,
                'sales_at_stop': sales_data,
                'inventory_after_sale': inventory_data
            }
            
            stop_aggregates[stop_id] = stop_aggregate
        
        logger.info(f"Generated {len(stop_aggregates)} stop-level aggregates")
        return stop_aggregates
    
    def aggregate_by_day(self, segments: List) -> Dict[str, Dict[str, Any]]:
        """Aggregate segments by seller and day."""
        day_aggregates = {}
        
        # Group segments by seller and day
        seller_days = defaultdict(list)
        for segment in segments:
            date_str = self._extract_date(segment.timestamp)
            seller_id = segment.seller_id
            key = f"{seller_id}_{date_str}"
            seller_days[key].append(segment)
        
        # Generate aggregate for each seller-day combination
        for key, day_segments in seller_days.items():
            if not day_segments:
                continue
            
            seller_id, date_str = key.split('_', 1)
            
            # Calculate metrics
            total_segments = len(day_segments)
            total_calls = self._estimate_call_count(day_segments)
            avg_sentiment = self._calculate_avg_sentiment(day_segments)
            sentiment_dist = self._calculate_sentiment_distribution(day_segments)
            dominant_emotion = self._calculate_dominant_emotion(day_segments)
            intent_dist = self._calculate_intent_distribution(day_segments)
            audio_file_stats = self._calculate_audio_file_stats(day_segments)
            
            # Get placeholder data
            total_sales = self.placeholders.get('sales_at_stop', {})
            closing_inventory = self.placeholders.get('inventory_after_sale', {})
            
            # Create aggregate
            day_aggregate = {
                'seller_id': seller_id,
                'date': date_str,
                'total_stops': self._estimate_stop_count(day_segments),
                'total_segments': total_segments,
                'total_calls': total_calls,
                'avg_sentiment_score': avg_sentiment,
                'sentiment_distribution': sentiment_dist,
                'dominant_emotion': dominant_emotion,
                'intent_distribution': intent_dist,
                'audio_files': audio_file_stats,
                'total_sales': total_sales,
                'closing_inventory': closing_inventory
            }
            
            day_aggregates[seller_id] = day_aggregate
        
        logger.info(f"Generated {len(day_aggregates)} day-level aggregates")
        return day_aggregates
    
    def _extract_date(self, timestamp: str) -> str:
        """Extract date string from ISO timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
            return "1970-01-01"
    
    def _estimate_call_count(self, segments: List) -> int:
        """Estimate number of calls from segments."""
        # Simple heuristic: assume 3-5 segments per call
        if not segments:
            return 0
        
        # Count segments and estimate calls
        n_segments = len(segments)
        estimated_calls = max(1, n_segments // 3)
        
        return estimated_calls
    
    def _estimate_stop_count(self, segments: List) -> int:
        """Estimate number of stops from segments."""
        if not segments:
            return 0
        
        # Count unique stop IDs
        stop_ids = set(seg.stop_id for seg in segments)
        return len(stop_ids)
    
    def _calculate_avg_sentiment(self, segments: List) -> float:
        """Calculate average sentiment score."""
        if not segments:
            return 0.0
        
        scores = [seg.sentiment_score for seg in segments]
        avg = statistics.mean(scores)
        return round(avg, 2)
    
    def _calculate_sentiment_distribution(self, segments: List) -> Dict[str, int]:
        """Calculate distribution of sentiment labels."""
        if not segments:
            return {'positive': 0, 'neutral': 0, 'negative': 0}
        
        labels = [seg.sentiment_label for seg in segments]
        counter = Counter(labels)
        
        return {
            'positive': counter.get('positive', 0),
            'neutral': counter.get('neutral', 0),
            'negative': counter.get('negative', 0)
        }
    
    def _calculate_dominant_emotion(self, segments: List) -> str:
        """Calculate dominant emotion."""
        if not segments:
            return 'neutral'
        
        emotions = [seg.emotion for seg in segments]
        counter = Counter(emotions)
        
        # Return most common emotion, default to neutral
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else 'neutral'
    
    def _calculate_top_intents(self, segments: List) -> List[str]:
        """Calculate top intents."""
        if not segments:
            return []
        
        intents = [seg.intent for seg in segments]
        counter = Counter(intents)
        
        # Return top 3 intents
        top_intents = [intent for intent, count in counter.most_common(3)]
        return top_intents
    
    def _calculate_intent_distribution(self, segments: List) -> Dict[str, int]:
        """Calculate distribution of intents."""
        if not segments:
            return {}
        
        intents = [seg.intent for seg in segments]
        counter = Counter(intents)
        
        return dict(counter)


def main():
    """Standalone execution for testing."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Test aggregation')
    parser.add_argument('--config', default='../config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create aggregator
    aggregator = Aggregator(config)
    
    # Test with mock segments
    from analyze import Segment
    
    mock_segments = [
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            timestamp="2025-08-19T19:11:12Z",
            text="Test text 1",
            intent="purchase",
            sentiment_score=0.5,
            sentiment_label="positive",
            emotion="happy",
            confidence=0.8
        ),
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            timestamp="2025-08-19T19:11:20Z",
            text="Test text 2",
            intent="inquiry",
            sentiment_score=-0.2,
            sentiment_label="negative",
            emotion="disappointed",
            confidence=0.7
        )
    ]
    
    # Test aggregation
    stop_aggs = aggregator.aggregate_by_stop(mock_segments)
    day_aggs = aggregator.aggregate_by_day(mock_segments)
    
    print("Stop-level aggregates:")
    for stop_id, data in stop_aggs.items():
        print(f"  {stop_id}: {data['n_segments']} segments, avg sentiment: {data['avg_sentiment_score']}")
    
    print("\nDay-level aggregates:")
    for seller_id, data in day_aggs.items():
        print(f"  {seller_id}: {data['total_segments']} segments, avg sentiment: {data['avg_sentiment_score']}")


if __name__ == "__main__":
    main()
