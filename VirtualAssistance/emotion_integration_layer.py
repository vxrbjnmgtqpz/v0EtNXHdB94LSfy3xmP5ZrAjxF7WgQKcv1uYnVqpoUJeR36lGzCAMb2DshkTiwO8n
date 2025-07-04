"""
Emotion Integration Layer

This module integrates the enhanced emotion parser with the existing emotion systems:
- Connects enhanced parsing with emotion interpolation engine
- Maps enhanced emotions to chord progressions 
- Provides unified interface for emotion-driven music generation
- Handles consistency between hierarchical emotions and music mapping
"""

import json
from typing import Dict, List, Tuple, Optional, Union
from enhanced_emotion_parser import EnhancedEmotionParser, EmotionState, EmotionDimensions
from emotion_interpolation_engine import EmotionInterpolationEngine
from contextual_progression_engine import ContextualProgressionEngine
import numpy as np

class EmotionIntegrationLayer:
    """
    Unified emotion processing system that connects enhanced parsing 
    with chord progression generation and interpolation.
    """
    
    def __init__(self):
        self.emotion_parser = EnhancedEmotionParser()
        self.interpolation_engine = EmotionInterpolationEngine()
        self.progression_engine = ContextualProgressionEngine()
        
        # Load existing emotion database
        self.emotion_database = self._load_emotion_database()
        
        # Create mapping between enhanced parser emotions and database emotions
        self.emotion_mapping = self._build_emotion_mapping()
        
    def process_emotion_input(self, text_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Complete emotion processing pipeline:
        1. Parse emotions from text using enhanced parser
        2. Map to database emotions
        3. Generate appropriate chord progressions
        4. Return comprehensive emotion and music data
        """
        
        # Step 1: Enhanced emotion parsing
        parsed_emotions = self.emotion_parser.parse_emotions(text_input)
        
        # Step 2: Create emotion state objects with full context
        emotion_states = self._create_emotion_states(parsed_emotions, text_input)
        
        # Step 3: Map to database emotions for chord generation
        database_emotions = self._map_to_database_emotions(parsed_emotions)
        
        # Step 4: Generate chord progressions using contextual engine
        chord_progressions = []
        for emotion, weight in database_emotions.items():
            if weight > 0.1:  # Only process significant emotions
                progression = self.progression_engine.generate_contextual_progression(emotion)
                progression['weight'] = weight
                chord_progressions.append(progression)
        
        # Step 5: Handle multi-emotion blending if needed
        if len([e for e in parsed_emotions.values() if e > 0.1]) > 1:
            blended_progression = self._blend_progressions(chord_progressions, parsed_emotions)
            chord_progressions.append(blended_progression)
        
        return {
            'parsed_emotions': parsed_emotions,
            'emotion_states': emotion_states,
            'database_emotions': database_emotions,
            'chord_progressions': chord_progressions,
            'primary_emotion': max(parsed_emotions.items(), key=lambda x: x[1])[0],
            'complexity': len([e for e in parsed_emotions.values() if e > 0.1]),
            'sarcasm_detected': self.emotion_parser._detect_sarcasm(text_input.lower()),
            'compound_emotions_detected': self._detect_compound_emotions_in_result(parsed_emotions)
        }
    
    def generate_music_from_emotions(self, emotion_result: Dict, length: int = 8) -> Dict:
        """
        Generate complete musical structure from processed emotions
        """
        primary_emotion = emotion_result['primary_emotion']
        chord_progressions = emotion_result['chord_progressions']
        
        if not chord_progressions:
            # Fallback progression
            fallback = self.progression_engine.generate_contextual_progression(primary_emotion, length)
            chord_progressions = [fallback]
        
        # Select best progression or blend multiple
        if len(chord_progressions) == 1:
            final_progression = chord_progressions[0]
        else:
            # Use the blended progression if available
            blended = [p for p in chord_progressions if p.get('type') == 'blended']
            final_progression = blended[0] if blended else chord_progressions[0]
        
        # Extend progression to desired length
        base_chords = final_progression['chords']
        extended_chords = (base_chords * ((length // len(base_chords)) + 1))[:length]
        
        return {
            'chords': extended_chords,
            'emotion': primary_emotion,
            'emotion_weights': emotion_result['parsed_emotions'],
            'cadence_type': final_progression.get('cadence_type', 'authentic'),
            'needs_resolution': final_progression.get('needs_resolution', True),
            'complexity_level': emotion_result['complexity'],
            'musical_context': self._determine_musical_context(emotion_result),
            'tempo_suggestion': self._suggest_tempo(emotion_result),
            'dynamics_suggestion': self._suggest_dynamics(emotion_result)
        }
    
    def _load_emotion_database(self) -> Dict:
        """Load the existing emotion progression database"""
        try:
            with open('emotion_progression_database.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return minimal structure if database not found
            return {'emotions': {}}
    
    def _build_emotion_mapping(self) -> Dict[str, str]:
        """
        Create mapping between enhanced parser emotions and database emotions
        This ensures compatibility with existing chord progression data
        """
        return {
            'Joy': 'Joy', 'Sadness': 'Sadness', 'Anger': 'Anger',
            'Fear': 'Fear', 'Trust': 'Trust', 'Love': 'Love',
            'Excitement': 'Joy', 'Melancholy': 'Sadness',
            'Awe': 'Wonder', 'Nostalgia': 'Sadness'
        }
    
    def _create_emotion_states(self, parsed_emotions: Dict[str, float], text: str) -> List[EmotionState]:
        """Create detailed emotion state objects with psychological dimensions"""
        emotion_states = []
        
        for emotion, weight in parsed_emotions.items():
            if weight > 0.05:  # Only create states for significant emotions
                dimensions = self.emotion_parser.dimension_map.get(emotion)
                if dimensions:
                    # Determine if this is a sub-emotion
                    primary_emotion, sub_emotion = self._determine_emotion_hierarchy(emotion)
                    
                    state = EmotionState(
                        primary_emotion=primary_emotion,
                        sub_emotion=sub_emotion,
                        family=self._get_emotion_family(primary_emotion),
                        intensity=weight,
                        dimensions=dimensions,
                        confidence=min(weight * 1.2, 1.0),  # Slight confidence boost
                        context_modifiers=self.emotion_parser._extract_context_modifiers(text.lower())
                    )
                    emotion_states.append(state)
        
        return emotion_states
    
    def _map_to_database_emotions(self, parsed_emotions: Dict[str, float]) -> Dict[str, float]:
        """Map enhanced parser emotions to database emotions"""
        database_emotions = {}
        for emotion, weight in parsed_emotions.items():
            mapped_emotion = self.emotion_mapping.get(emotion, emotion)
            database_emotions[mapped_emotion] = database_emotions.get(mapped_emotion, 0) + weight
        
        total = sum(database_emotions.values())
        if total > 0:
            database_emotions = {k: v/total for k, v in database_emotions.items()}
        
        return database_emotions
    
    def _blend_progressions(self, progressions: List[Dict], emotion_weights: Dict[str, float]) -> Dict:
        """Blend multiple chord progressions based on emotion weights"""
        if len(progressions) <= 1:
            return progressions[0] if progressions else {}
        
        # Simple blending: alternate between progressions based on weights
        primary_prog = max(progressions, key=lambda p: p.get('weight', 0))
        secondary_prog = [p for p in progressions if p != primary_prog][0]
        
        # Create a blended progression
        primary_chords = primary_prog['chords']
        secondary_chords = secondary_prog['chords']
        
        # Interleave chords based on relative weights
        blended_chords = []
        max_length = max(len(primary_chords), len(secondary_chords))
        
        for i in range(max_length):
            if i % 2 == 0 or len(secondary_chords) <= i:
                blended_chords.append(primary_chords[i % len(primary_chords)])
            else:
                blended_chords.append(secondary_chords[i % len(secondary_chords)])
        
        return {
            'chords': blended_chords,
            'emotion': f"{primary_prog['emotion']}+{secondary_prog['emotion']}",
            'type': 'blended',
            'cadence_type': primary_prog.get('cadence_type', 'authentic'),
            'needs_resolution': primary_prog.get('needs_resolution', True),
            'component_emotions': [p['emotion'] for p in progressions]
        }
    
    def _determine_emotion_hierarchy(self, emotion: str) -> Tuple[str, Optional[str]]:
        """Determine if emotion is primary or sub-emotion"""
        for primary, data in self.emotion_parser.emotion_hierarchy.items():
            if emotion == primary:
                return primary, None
            if emotion in data.get('sub_emotions', {}):
                return primary, emotion
        return emotion, None
    
    def _get_emotion_family(self, emotion: str):
        """Get emotion family for a given emotion"""
        emotion_data = self.emotion_parser.emotion_hierarchy.get(emotion, {})
        return emotion_data.get('family')
    
    def _detect_compound_emotions_in_result(self, emotions: Dict[str, float]) -> List[str]:
        """Detect if any compound emotions are present in the result"""
        detected = []
        for compound_name, compound_emotion in self.emotion_parser.compound_emotions.items():
            # Check if components are present with significant weights
            components_present = all(
                emotions.get(comp, 0) > 0.1 for comp in compound_emotion.components.keys()
            )
            if components_present:
                detected.append(compound_name)
        return detected
    
    def _determine_musical_context(self, emotion_result: Dict) -> Dict:
        """Determine musical context based on emotion analysis"""
        primary_emotion = emotion_result['primary_emotion']
        dimensions = self.emotion_parser.dimension_map.get(primary_emotion)
        
        if not dimensions:
            return {'mode': 'major', 'style': 'general'}
        
        # Determine mode based on valence
        mode = 'major' if dimensions.valence > 0 else 'minor'
        
        # Determine style based on arousal and dominance
        if dimensions.arousal > 0.7 and dimensions.dominance > 0.7:
            style = 'aggressive'
        elif dimensions.arousal < 0.3 and dimensions.valence > 0.5:
            style = 'peaceful'
        elif dimensions.valence < -0.5 and dimensions.arousal < 0.4:
            style = 'melancholic'
        else:
            style = 'general'
        
        return {'mode': mode, 'style': style, 'dimensions': dimensions}
    
    def _suggest_tempo(self, emotion_result: Dict) -> str:
        """Suggest tempo based on emotion arousal levels"""
        primary_emotion = emotion_result['primary_emotion']
        dimensions = self.emotion_parser.dimension_map.get(primary_emotion)
        
        if not dimensions:
            return 'moderate'
        
        if dimensions.arousal > 0.8:
            return 'fast'
        elif dimensions.arousal > 0.5:
            return 'moderate-fast'
        elif dimensions.arousal > 0.3:
            return 'moderate'
        else:
            return 'slow'
    
    def _suggest_dynamics(self, emotion_result: Dict) -> str:
        """Suggest dynamics based on emotion intensity and dominance"""
        primary_emotion = emotion_result['primary_emotion']
        dimensions = self.emotion_parser.dimension_map.get(primary_emotion)
        
        if not dimensions:
            return 'moderate'
        
        # Combine arousal and dominance for dynamics
        intensity = (dimensions.arousal + dimensions.dominance) / 2
        
        if intensity > 0.8:
            return 'loud'
        elif intensity > 0.6:
            return 'moderate-loud'
        elif intensity > 0.4:
            return 'moderate'
        else:
            return 'soft'

# Example usage and testing
if __name__ == "__main__":
    integration = EmotionIntegrationLayer()
    
    # Test with various emotional inputs
    test_inputs = [
        "I'm feeling bittersweet about this beautiful memory",
        "This makes me absolutely furious and disgusted!",
        "I'm so grateful and filled with peaceful joy",
        "Oh great, just what I needed... more problems", # Sarcasm test
        "I'm excited but also a bit nervous about tomorrow"
    ]
    
    for text in test_inputs:
        print(f"\nInput: {text}")
        result = integration.process_emotion_input(text)
        music = integration.generate_music_from_emotions(result)
        
        print(f"Primary emotion: {result['primary_emotion']}")
        print(f"Emotions detected: {list(result['parsed_emotions'].keys())}")
        print(f"Chord progression: {' - '.join(music['chords'][:4])}")
        print(f"Musical context: {music['musical_context']}")
        if result['compound_emotions_detected']:
            print(f"Compound emotions: {result['compound_emotions_detected']}") 