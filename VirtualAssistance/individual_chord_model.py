"""
Individual Chord Generation Model from Natural Language Prompts
Based on Chord-to-Emotion Mapping across Multiple Musical Contexts

This model generates single chords from natural language input by:
1. Parsing emotional content from text prompts
2. Mapping emotions to chord types within musical contexts (modes, jazz, blues)
3. Selecting appropriate individual chords based on emotional weights
4. Supporting extended harmony and modal chord colors

Architecture:
- Emotion Parser: Keyword-based text analysis → emotion weight vectors
- Chord Mapper: Weighted emotion blend → chord selection within context
- Context Handler: Mode/style-aware chord filtering and selection
- Chord Database: Comprehensive chord-to-emotion mappings
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class IndividualChord:
    """Represents a single chord with emotional and contextual metadata"""
    symbol: str  # e.g., "Cmaj7", "Am", "F#dim7"
    roman_numeral: str  # e.g., "I", "vi", "♯iv°"
    mode_context: str  # e.g., "Ionian", "Aeolian", "Dorian" - modal contexts
    style_context: str  # e.g., "Jazz", "Blues", "Classical" - genre/stylistic contexts
    emotion_weights: Dict[str, float]
    chord_id: str

class EmotionChordDatabase:
    """Database of chord-to-emotion mappings across different musical contexts"""
    
    def __init__(self):
        self.chord_emotion_map = self._load_chord_database()
        
    def _load_chord_database(self) -> List[IndividualChord]:
        """Load comprehensive chord-to-emotion mapping database"""
        try:
            with open('individual_chord_database.json', 'r') as f:
                data = json.load(f)
            
            chords = []
            for chord_data in data['chord_to_emotion_map']:
                # Default style_context if missing from database
                style_context = chord_data.get('style_context', 'Classical')
                chord = IndividualChord(
                    symbol=chord_data.get('symbol', chord_data['chord']),
                    roman_numeral=chord_data['chord'],
                    mode_context=chord_data['mode_context'],
                    style_context=style_context,
                    emotion_weights=chord_data['emotion_weights'],
                    chord_id=chord_data.get('chord_id', f"{style_context}_{chord_data['mode_context']}_{chord_data['chord']}")
                )
                chords.append(chord)
            
            return chords
            
        except FileNotFoundError:
            print("Warning: individual_chord_database.json not found. Creating sample data.")
            return self._create_sample_chord_database()
    
    def _create_sample_chord_database(self) -> List[IndividualChord]:
        """Create sample chord database based on the IndividChords.md mapping"""
        sample_chords = [
            # Ionian chords
            {
                "chord": "I",
                "symbol": "C",
                "mode_context": "Ionian",
                "style_context": "Classical",
                "emotion_weights": {
                    "Joy": 1.0, "Trust": 0.7, "Love": 0.6, "Surprise": 0.3,
                    "Sadness": 0.0, "Anger": 0.0, "Fear": 0.0, "Disgust": 0.0,
                    "Anticipation": 0.2, "Shame": 0.0, "Envy": 0.0, "Aesthetic Awe": 0.4
                }
            },
            {
                "chord": "IV",
                "symbol": "F",
                "mode_context": "Ionian",
                "style_context": "Classical",
                "emotion_weights": {
                    "Joy": 0.8, "Trust": 0.6, "Love": 0.5, "Surprise": 0.2,
                    "Sadness": 0.1, "Anticipation": 0.3, "Aesthetic Awe": 0.4,
                    "Fear": 0.0, "Disgust": 0.0, "Shame": 0.0, "Anger": 0.0, "Envy": 0.0
                }
            },
            {
                "chord": "V",
                "symbol": "G",
                "mode_context": "Ionian",
                "style_context": "Classical",
                "emotion_weights": {
                    "Anticipation": 0.9, "Joy": 0.4, "Love": 0.3, "Trust": 0.2,
                    "Surprise": 0.4, "Fear": 0.2, "Sadness": 0.1, "Anger": 0.1,
                    "Shame": 0.0, "Disgust": 0.0, "Aesthetic Awe": 0.3, "Envy": 0.0
                }
            },
            {
                "chord": "vi",
                "symbol": "Am",
                "mode_context": "Ionian",
                "style_context": "Classical",
                "emotion_weights": {
                    "Sadness": 0.7, "Trust": 0.5, "Love": 0.5, "Shame": 0.3,
                    "Anticipation": 0.2, "Joy": 0.2, "Fear": 0.1, "Aesthetic Awe": 0.1,
                    "Anger": 0.0, "Disgust": 0.0, "Surprise": 0.0, "Envy": 0.0
                }
            },
            # Aeolian chords
            {
                "chord": "i",
                "symbol": "Am",
                "mode_context": "Aeolian",
                "style_context": "Classical",
                "emotion_weights": {
                    "Sadness": 1.0, "Shame": 0.6, "Trust": 0.4, "Love": 0.3,
                    "Fear": 0.2, "Anticipation": 0.3, "Envy": 0.2,
                    "Joy": 0.0, "Anger": 0.0, "Disgust": 0.0, "Surprise": 0.0, "Aesthetic Awe": 0.0
                }
            },
            {
                "chord": "iv",
                "symbol": "Dm",
                "mode_context": "Aeolian",
                "style_context": "Classical",
                "emotion_weights": {
                    "Sadness": 0.8, "Shame": 0.6, "Fear": 0.3, "Trust": 0.3,
                    "Anger": 0.1, "Love": 0.2, "Joy": 0.0, "Disgust": 0.0,
                    "Surprise": 0.0, "Anticipation": 0.0, "Envy": 0.0, "Aesthetic Awe": 0.0
                }
            },
            # Jazz chords
            {
                "chord": "maj7",
                "symbol": "Cmaj7",
                "mode_context": "Ionian",
                "style_context": "Jazz",
                "emotion_weights": {
                    "Joy": 0.8, "Love": 0.7, "Trust": 0.6, "Aesthetic Awe": 0.6,
                    "Sadness": 0.3, "Surprise": 0.2, "Fear": 0.0, "Anger": 0.0,
                    "Disgust": 0.0, "Anticipation": 0.0, "Shame": 0.0, "Envy": 0.0
                }
            },
            {
                "chord": "min7",
                "symbol": "Am7",
                "mode_context": "Dorian",
                "style_context": "Jazz",
                "emotion_weights": {
                    "Sadness": 0.6, "Trust": 0.5, "Love": 0.4, "Shame": 0.3,
                    "Joy": 0.3, "Fear": 0.0, "Anger": 0.0, "Disgust": 0.0,
                    "Surprise": 0.0, "Anticipation": 0.0, "Envy": 0.0, "Aesthetic Awe": 0.0
                }
            },
            {
                "chord": "7",
                "symbol": "G7",
                "mode_context": "Mixolydian",
                "style_context": "Jazz",
                "emotion_weights": {
                    "Anticipation": 0.9, "Surprise": 0.6, "Trust": 0.4, "Aesthetic Awe": 0.3,
                    "Anger": 0.2, "Fear": 0.3, "Joy": 0.0, "Sadness": 0.0,
                    "Disgust": 0.0, "Shame": 0.0, "Love": 0.0, "Envy": 0.0
                }
            },
            {
                "chord": "dim7",
                "symbol": "G#dim7",
                "mode_context": "Locrian",
                "style_context": "Jazz",
                "emotion_weights": {
                    "Fear": 1.0, "Disgust": 0.6, "Shame": 0.6, "Anticipation": 0.7,
                    "Surprise": 0.5, "Joy": 0.0, "Sadness": 0.0, "Trust": 0.0,
                    "Love": 0.0, "Anger": 0.0, "Envy": 0.0, "Aesthetic Awe": 0.0
                }
            },
            # Blues chords
            {
                "chord": "I7",
                "symbol": "C7",
                "mode_context": "Mixolydian",
                "style_context": "Blues",
                "emotion_weights": {
                    "Joy": 0.8, "Trust": 0.6, "Sadness": 0.4, "Love": 0.4,
                    "Shame": 0.2, "Aesthetic Awe": 0.3, "Surprise": 0.4,
                    "Fear": 0.0, "Anger": 0.0, "Disgust": 0.0, "Anticipation": 0.0, "Envy": 0.0
                }
            }
        ]
        
        chords = []
        for i, chord_data in enumerate(sample_chords):
            chord = IndividualChord(
                symbol=chord_data['symbol'],
                roman_numeral=chord_data['chord'],
                mode_context=chord_data['mode_context'],
                style_context=chord_data['style_context'],
                emotion_weights=chord_data['emotion_weights'],
                chord_id=f"sample_{i:03d}"
            )
            chords.append(chord)
        
        return chords

class IndividualChordEmotionParser:
    """Parse emotional content from text for individual chord generation"""
    
    def __init__(self):
        # Complete 22-emotion system emotion labels
        self.emotion_labels = ["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
                              "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe", "Malice",
                              "Arousal", "Guilt", "Reverence", "Wonder", "Dissociation", 
                              "Empowerment", "Belonging", "Ideology", "Gratitude"]
    
    def parse_emotion_weights(self, text: str) -> Dict[str, float]:
        """Parse text and return emotion weights using keyword matching"""
        text_lower = text.lower()
        
        # Initialize weights
        emotion_weights = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Load keyword mapping from database or use fallback
        emotion_keywords = {
            "Joy": ["happy", "joy", "joyful", "excited", "cheerful", "uplifted", "bright", "celebratory", "elated", "playful"],
            "Sadness": ["sad", "depressed", "grieving", "blue", "mournful", "melancholy", "sorrowful", "down", "bluesy"],
            "Fear": ["afraid", "scared", "anxious", "nervous", "terrified", "worried", "tense", "fearful", "mysterious", "dark"],
            "Anger": ["angry", "furious", "frustrated", "mad", "irritated", "rage", "aggressive", "hostile"],
            "Disgust": ["disgusted", "grossed out", "repulsed", "nauseated", "revolted", "sickened"],
            "Surprise": ["surprised", "shocked", "amazed", "startled", "unexpected", "wonder", "astonished"],
            "Trust": ["trust", "safe", "secure", "supported", "bonded", "intimate", "comfortable", "confident"],
            "Anticipation": ["anticipation", "expectation", "eager", "hopeful", "building", "yearning", "excited"],
            "Shame": ["guilt", "shame", "regret", "embarrassed", "remorseful", "ashamed", "humiliated"],
            "Love": ["love", "romantic", "affection", "caring", "warm", "tender", "devoted", "passionate"],
            "Envy": ["jealous", "envious", "spiteful", "competitive", "bitter", "possessive", "resentful"],
            "Aesthetic Awe": ["awe", "wonder", "sublime", "inspired", "majestic", "transcendent", "beautiful"],
            "Malice": ["malicious", "evil", "wicked", "cruel", "vicious", "sinister", "vindictive", "sadistic", "callous", "manipulative"]
        }
        
        # Count keyword matches
        matches = {emotion: 0 for emotion in self.emotion_labels}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matches[emotion] += 1
        
        # Convert to weights
        total_matches = sum(matches.values())
        if total_matches > 0:
            for emotion in self.emotion_labels:
                emotion_weights[emotion] = matches[emotion] / total_matches
        else:
            # Default fallback based on basic sentiment
            if any(word in text_lower for word in ["happy", "joy", "good", "great", "bright", "playful"]):
                emotion_weights["Joy"] = 1.0
            elif any(word in text_lower for word in ["sad", "down", "depressed", "melancholy", "blue", "bluesy"]):
                emotion_weights["Sadness"] = 1.0
            elif any(word in text_lower for word in ["angry", "mad", "frustrated", "furious"]):
                emotion_weights["Anger"] = 1.0
            elif any(word in text_lower for word in ["scared", "afraid", "nervous", "anxious", "dark", "mysterious"]):
                emotion_weights["Fear"] = 1.0
            elif any(word in text_lower for word in ["love", "romantic", "tender", "caring", "warm"]):
                emotion_weights["Love"] = 1.0
            elif any(word in text_lower for word in ["beautiful", "awe", "amazing", "sublime", "jazz"]):
                emotion_weights["Aesthetic Awe"] = 1.0
            else:
                # Very basic fallback
                emotion_weights["Joy"] = 1.0
        
        return emotion_weights

class IndividualChordModel:
    """
    Complete pipeline: Text → Emotions → Individual Chord Selection
    """
    
    def __init__(self):
        self.database = EmotionChordDatabase()
        self.emotion_parser = IndividualChordEmotionParser()
        
    def generate_chord_from_prompt(self, text_prompt: str, 
                                  mode_preference: str = "Any",
                                  style_preference: str = "Any",
                                  key: str = "C",
                                  num_options: int = 1) -> List[Dict]:
        """
        Main interface: Generate individual chords from natural language
        
        Args:
            text_prompt: Natural language description (e.g., "melancholy but hopeful")
            mode_preference: Modal context ("Ionian", "Aeolian", "Dorian", "Any")
            style_preference: Style context ("Jazz", "Blues", "Classical", "Any")
            key: Root key for chord symbols (default: C)
            num_options: Number of chord options to return
            
        Returns:
            List of chord dictionaries with metadata
        """
        # 1. Parse emotions from text
        emotion_weights = self.emotion_parser.parse_emotion_weights(text_prompt)
        
        # 2. Find matching chords
        candidate_chords = self._find_matching_chords(emotion_weights, mode_preference, style_preference)
        
        # 3. Select best options
        selected_chords = self._select_chords(candidate_chords, num_options)
        
        # 4. Format results
        results = []
        for chord, score in selected_chords:
            # Transpose to requested key if needed
            chord_symbol = self._transpose_chord(chord.symbol, "C", key)
            
            result = {
                "chord_id": chord.chord_id,
                "prompt": text_prompt,
                "emotion_weights": emotion_weights,
                "chord_symbol": chord_symbol,
                "roman_numeral": chord.roman_numeral,
                "mode_context": chord.mode_context,
                "style_context": chord.style_context,
                "emotional_score": score,
                "key": key,
                "metadata": {
                    "generation_method": "emotion_weighted_selection",
                    "timestamp": datetime.now().isoformat(),
                    "mode_filter": mode_preference,
                    "style_filter": style_preference
                }
            }
            results.append(result)
            
        return results
    
    def _find_matching_chords(self, emotion_weights: Dict[str, float], 
                             mode_preference: str, style_preference: str) -> List[Tuple[IndividualChord, float]]:
        """Find chords that match the emotional profile within mode and style constraints"""
        candidate_chords = []
        
        for chord in self.database.chord_emotion_map:
            # Filter by mode context if specified
            if mode_preference != "Any" and chord.mode_context != mode_preference:
                continue
                
            # Filter by style context if specified
            if style_preference != "Any" and chord.style_context != style_preference:
                continue
            
            # Calculate emotional compatibility score
            score = 0.0
            for emotion, user_weight in emotion_weights.items():
                chord_weight = chord.emotion_weights.get(emotion, 0.0)
                score += user_weight * chord_weight
            
            if score > 0.01:  # Only include chords with some emotional relevance
                candidate_chords.append((chord, score))
        
        return candidate_chords
    
    def _select_chords(self, candidates: List[Tuple[IndividualChord, float]], 
                      num_options: int) -> List[Tuple[IndividualChord, float]]:
        """Select the best chord options based on emotional scores"""
        if not candidates:
            # Fallback to a basic major chord
            fallback_chord = IndividualChord(
                symbol="C",
                roman_numeral="I",
                mode_context="Ionian",
                style_context="Classical",
                emotion_weights={"Joy": 1.0},
                chord_id="fallback_001"
            )
            return [(fallback_chord, 1.0)]
        
        # Sort by score and return top options
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:num_options]
    
    def _transpose_chord(self, chord_symbol: str, from_key: str, to_key: str) -> str:
        """Advanced chord transposition with proper chromatic handling"""
        if from_key == to_key:
            return chord_symbol
        
        # Chromatic scale for transposition
        chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        try:
            # Calculate semitone interval
            from_index = chromatic_scale.index(from_key)
            to_index = chromatic_scale.index(to_key)
            interval = (to_index - from_index) % 12
            
            # Simple chord root extraction (basic implementation)
            # This handles simple cases like "C", "Am", "Cmaj7", etc.
            root_note = chord_symbol[0]
            
            # Handle sharp/flat in root
            if len(chord_symbol) > 1 and chord_symbol[1] in ['#', 'b']:
                if chord_symbol[1] == '#':
                    root_note = chord_symbol[0] + '#'
                    suffix = chord_symbol[2:]
                elif chord_symbol[1] == 'b':
                    # Convert flat to sharp equivalent
                    flat_to_sharp = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
                    root_note = flat_to_sharp.get(chord_symbol[:2], chord_symbol[:2])
                    suffix = chord_symbol[2:]
            else:
                suffix = chord_symbol[1:]
            
            # Transpose the root
            if root_note in chromatic_scale:
                root_index = chromatic_scale.index(root_note)
                new_root_index = (root_index + interval) % 12
                new_root = chromatic_scale[new_root_index]
                
                return new_root + suffix
            else:
                # Fallback for unrecognized chord symbols
                return chord_symbol
                
        except (ValueError, IndexError):
            # If transposition fails, return original chord
            return chord_symbol
    
    def get_available_contexts(self) -> Dict[str, List[str]]:
        """Get list of available musical contexts separated by type"""
        mode_contexts = set()
        style_contexts = set()
        for chord in self.database.chord_emotion_map:
            mode_contexts.add(chord.mode_context)
            style_contexts.add(chord.style_context)
        return {
            "modes": sorted(list(mode_contexts)),
            "styles": sorted(list(style_contexts))
        }
    
    def analyze_emotional_content(self, text_prompt: str) -> Dict:
        """Analyze the emotional content of a text prompt"""
        emotion_weights = self.emotion_parser.parse_emotion_weights(text_prompt)
        
        # Find dominant emotions
        dominant_emotions = [(emotion, weight) for emotion, weight in emotion_weights.items() if weight > 0.1]
        dominant_emotions.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "full_emotion_weights": emotion_weights,
            "dominant_emotions": dominant_emotions[:3],
            "primary_emotion": dominant_emotions[0] if dominant_emotions else ("Joy", 1.0),
            "emotional_complexity": len([w for w in emotion_weights.values() if w > 0.1])
        }

# Usage example
def demo_individual_chord_generation():
    """Demonstrate individual chord generation from prompts"""
    print("=== Individual Chord Generation Demo ===")
    
    model = IndividualChordModel()
    
    test_prompts = [
        "I feel happy and bright",
        "melancholy and reflective", 
        "anxious and tense",
        "romantic and warm",
        "mysterious and dark",
        "playful jazz feeling",
        "bluesy sadness"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Generate chord options
        results = model.generate_chord_from_prompt(prompt, num_options=3)
        
        for i, result in enumerate(results, 1):
            primary_emotion = max(result['emotion_weights'].items(), key=lambda x: x[1])
            print(f"  {i}. {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']} ({result['style_context']})")
            print(f"     Primary emotion: {primary_emotion[0]} ({primary_emotion[1]:.2f})")
            print(f"     Score: {result['emotional_score']:.3f}")

if __name__ == "__main__":
    demo_individual_chord_generation()
