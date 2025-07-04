"""
Enhanced Emotion Parser with Hierarchical Classification and Multi-Emotion Detection

This module implements the audit recommendations for improved emotion classification:
- Hierarchical emotion structure (core emotions with sub-emotions)
- Multi-label emotion detection for compound states
- Context awareness and sarcasm detection
- Psychological validity with arousal/dominance dimensions
- Compound emotion targets for better interpolation
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

class EmotionFamily(Enum):
    """Core emotion families based on Plutchik's wheel and psychological research"""
    JOY = "Joy"
    SADNESS = "Sadness"
    ANGER = "Anger"
    FEAR = "Fear"
    TRUST = "Trust"
    DISGUST = "Disgust"
    SURPRISE = "Surprise"
    ANTICIPATION = "Anticipation"
    
    # Extended families for music-specific emotions
    LOVE = "Love"
    AESTHETIC = "Aesthetic"
    SOCIAL = "Social"
    SPIRITUAL = "Spiritual"
    COMPLEX = "Complex"

@dataclass
class EmotionDimensions:
    """Psychological dimensions of emotion based on PAD model"""
    valence: float  # Pleasure/displeasure (-1 to 1)
    arousal: float  # Energy/activation (0 to 1)
    dominance: float  # Control/power (0 to 1)

@dataclass 
class EmotionState:
    """Complete emotion state with hierarchical structure"""
    primary_emotion: str
    sub_emotion: Optional[str]
    family: EmotionFamily
    intensity: float
    dimensions: EmotionDimensions
    confidence: float
    context_modifiers: List[str]

class CompoundEmotion:
    """Defines known compound emotions for better interpolation"""
    def __init__(self, name: str, components: Dict[str, float], description: str):
        self.name = name
        self.components = components  # emotion_name: weight
        self.description = description

class EnhancedEmotionParser:
    """
    Enhanced emotion parser implementing audit recommendations
    """
    
    def __init__(self):
        self.emotion_hierarchy = self._build_emotion_hierarchy()
        self.dimension_map = self._build_dimension_map() 
        self.compound_emotions = self._build_compound_emotions()
        self.context_modifiers = self._build_context_modifiers()
        self.sarcasm_indicators = self._build_sarcasm_indicators()
        
    def parse_emotions(self, text: str) -> Dict[str, float]:
        """
        Enhanced emotion parsing with hierarchical classification and multi-emotion detection
        
        Returns normalized emotion weights for all detected emotions
        """
        text_lower = text.lower()
        
        # Step 1: Check for sarcasm
        sarcasm_detected = self._detect_sarcasm(text_lower)
        
        # Step 2: Extract context modifiers
        context_mods = self._extract_context_modifiers(text_lower)
        
        # Step 3: Detect primary and sub-emotions
        emotion_candidates = self._detect_hierarchical_emotions(text_lower)
        
        # Step 4: Check for compound emotions
        compound_matches = self._detect_compound_emotions(text_lower, emotion_candidates)
        
        # Step 5: Apply context modifiers and sarcasm
        final_emotions = self._apply_context_modifiers(
            emotion_candidates, context_mods, sarcasm_detected
        )
        
        # Step 6: Add compound emotions
        final_emotions.update(compound_matches)
        
        # Step 7: Normalize weights
        total_weight = sum(final_emotions.values())
        if total_weight > 0:
            final_emotions = {k: v/total_weight for k, v in final_emotions.items()}
        else:
            # Default to neutral state
            final_emotions = {"Trust": 0.6, "Joy": 0.4}
        
        return final_emotions
        
    def _build_emotion_hierarchy(self) -> Dict[str, Dict]:
        """Build comprehensive hierarchical emotion structure with music-specific emotions"""
        return {
            "Joy": {
                "family": EmotionFamily.JOY,
                "sub_emotions": {
                    "Excitement": {"keywords": ["excited", "thrilled", "energetic", "pumped"], "intensity_modifier": 1.2},
                    "Contentment": {"keywords": ["content", "peaceful", "serene", "satisfied"], "intensity_modifier": 0.7},
                    "Euphoria": {"keywords": ["euphoric", "ecstatic", "blissful", "elated"], "intensity_modifier": 1.5},
                    "Cheerfulness": {"keywords": ["cheerful", "bright", "sunny", "upbeat"], "intensity_modifier": 1.0}
                }
            },
            "Sadness": {
                "family": EmotionFamily.SADNESS, 
                "sub_emotions": {
                    "Melancholy": {"keywords": ["melancholic", "bittersweet", "wistful", "pensive"], "intensity_modifier": 0.8},
                    "Grief": {"keywords": ["grieving", "heartbroken", "devastated", "mourning"], "intensity_modifier": 1.3},
                    "Despair": {"keywords": ["despair", "hopeless", "despondent", "desolate"], "intensity_modifier": 1.4},
                    "Sorrow": {"keywords": ["sorrowful", "mournful", "doleful", "woeful"], "intensity_modifier": 1.1}
                }
            },
            "Anger": {
                "family": EmotionFamily.ANGER,
                "sub_emotions": {
                    "Frustration": {"keywords": ["frustrated", "annoyed", "irritated", "vexed"], "intensity_modifier": 0.8},
                    "Rage": {"keywords": ["rage", "furious", "livid", "enraged"], "intensity_modifier": 1.4},
                    "Annoyance": {"keywords": ["annoying", "bothersome", "irksome"], "intensity_modifier": 0.6},
                    "Indignation": {"keywords": ["indignant", "outraged", "offended"], "intensity_modifier": 1.2}
                }
            },
            "Fear": {
                "family": EmotionFamily.FEAR,
                "sub_emotions": {
                    "Anxiety": {"keywords": ["anxious", "worried", "nervous", "uneasy"], "intensity_modifier": 0.9},
                    "Terror": {"keywords": ["terrified", "horrified", "petrified"], "intensity_modifier": 1.5},
                    "Panic": {"keywords": ["panic", "frantic", "hysterical"], "intensity_modifier": 1.3},
                    "Apprehension": {"keywords": ["apprehensive", "cautious", "wary"], "intensity_modifier": 0.7}
                }
            },
            "Trust": {
                "family": EmotionFamily.TRUST,
                "sub_emotions": {
                    "Confidence": {"keywords": ["confident", "assured", "certain"], "intensity_modifier": 1.1},
                    "Security": {"keywords": ["secure", "safe", "protected"], "intensity_modifier": 0.9},
                    "Faith": {"keywords": ["faithful", "believing", "devoted"], "intensity_modifier": 1.0},
                    "Comfort": {"keywords": ["comfortable", "cozy", "warm"], "intensity_modifier": 0.8}
                }
            },
            "Disgust": {
                "family": EmotionFamily.DISGUST,
                "sub_emotions": {
                    "Revulsion": {"keywords": ["revolting", "repulsive", "nauseating"], "intensity_modifier": 1.2},
                    "Contempt": {"keywords": ["contemptuous", "scornful", "disdainful"], "intensity_modifier": 1.1},
                    "Aversion": {"keywords": ["averse", "repelled", "repugnant"], "intensity_modifier": 1.0}
                }
            },
            "Surprise": {
                "family": EmotionFamily.SURPRISE,
                "sub_emotions": {
                    "Astonishment": {"keywords": ["astonished", "amazed", "astounded"], "intensity_modifier": 1.2},
                    "Wonder": {"keywords": ["wonderful", "marvelous", "miraculous"], "intensity_modifier": 1.0},
                    "Shock": {"keywords": ["shocked", "stunned", "flabbergasted"], "intensity_modifier": 1.3}
                }
            },
            "Anticipation": {
                "family": EmotionFamily.ANTICIPATION,
                "sub_emotions": {
                    "Eagerness": {"keywords": ["eager", "keen", "enthusiastic"], "intensity_modifier": 1.1},
                    "Hope": {"keywords": ["hopeful", "optimistic", "expectant"], "intensity_modifier": 0.9},
                    "Suspense": {"keywords": ["suspenseful", "tense", "expectant"], "intensity_modifier": 1.2}
                }
            },
            "Love": {
                "family": EmotionFamily.LOVE,
                "sub_emotions": {
                    "Affection": {"keywords": ["affectionate", "fond", "caring"], "intensity_modifier": 0.8},
                    "Passion": {"keywords": ["passionate", "intense", "ardent"], "intensity_modifier": 1.3},
                    "Tenderness": {"keywords": ["tender", "gentle", "soft"], "intensity_modifier": 0.7},
                    "Devotion": {"keywords": ["devoted", "loyal", "committed"], "intensity_modifier": 1.0}
                }
            },
            "Aesthetic": {
                "family": EmotionFamily.AESTHETIC,
                "sub_emotions": {
                    "Awe": {"keywords": ["awe", "awesome", "magnificent", "sublime"], "intensity_modifier": 1.2},
                    "Beauty": {"keywords": ["beautiful", "gorgeous", "stunning"], "intensity_modifier": 1.0},
                    "Transcendence": {"keywords": ["transcendent", "spiritual", "ethereal"], "intensity_modifier": 1.4},
                    "Wonder": {"keywords": ["wondrous", "magical", "enchanting"], "intensity_modifier": 1.1}
                }
            },
            "Social": {
                "family": EmotionFamily.SOCIAL,
                "sub_emotions": {
                    "Shame": {"keywords": ["ashamed", "embarrassed", "humiliated"], "intensity_modifier": 1.1},
                    "Envy": {"keywords": ["envious", "jealous", "covetous"], "intensity_modifier": 1.0},
                    "Pride": {"keywords": ["proud", "accomplished", "triumphant"], "intensity_modifier": 1.1},
                    "Guilt": {"keywords": ["guilty", "remorseful", "regretful"], "intensity_modifier": 1.0}
                }
            },
            "Spiritual": {
                "family": EmotionFamily.SPIRITUAL,
                "sub_emotions": {
                    "Reverence": {"keywords": ["reverent", "sacred", "holy"], "intensity_modifier": 1.0},
                    "Gratitude": {"keywords": ["grateful", "thankful", "appreciative"], "intensity_modifier": 0.9},
                    "Humility": {"keywords": ["humble", "modest", "meek"], "intensity_modifier": 0.8},
                    "Peace": {"keywords": ["peaceful", "serene", "tranquil"], "intensity_modifier": 0.7}
                }
            },
            "Complex": {
                "family": EmotionFamily.COMPLEX,
                "sub_emotions": {
                    "Nostalgia": {"keywords": ["nostalgic", "reminiscent", "longing"], "intensity_modifier": 0.9},
                    "Malice": {"keywords": ["malicious", "spiteful", "vindictive"], "intensity_modifier": 1.2},
                    "Empowerment": {"keywords": ["empowered", "strong", "capable"], "intensity_modifier": 1.1},
                    "Loneliness": {"keywords": ["lonely", "isolated", "alone"], "intensity_modifier": 1.0}
                }
            }
        }
    
    def _build_dimension_map(self) -> Dict[str, EmotionDimensions]:
        """Build comprehensive psychological dimension mappings based on PAD model"""
        return {
            # Core emotions
            "Joy": EmotionDimensions(0.8, 0.6, 0.7),
            "Sadness": EmotionDimensions(-0.7, 0.2, 0.3),
            "Anger": EmotionDimensions(-0.6, 0.8, 0.8),
            "Fear": EmotionDimensions(-0.6, 0.8, 0.2),
            "Trust": EmotionDimensions(0.6, 0.3, 0.6),
            "Disgust": EmotionDimensions(-0.7, 0.4, 0.5),
            "Surprise": EmotionDimensions(0.0, 0.9, 0.3),
            "Anticipation": EmotionDimensions(0.4, 0.7, 0.6),
            
            # Love emotions
            "Love": EmotionDimensions(0.9, 0.5, 0.4),
            "Affection": EmotionDimensions(0.7, 0.3, 0.4),
            "Passion": EmotionDimensions(0.8, 0.9, 0.7),
            "Tenderness": EmotionDimensions(0.8, 0.2, 0.3),
            "Devotion": EmotionDimensions(0.7, 0.4, 0.5),
            
            # Aesthetic emotions
            "Awe": EmotionDimensions(0.6, 0.7, 0.2),
            "Beauty": EmotionDimensions(0.8, 0.4, 0.3),
            "Transcendence": EmotionDimensions(0.9, 0.6, 0.1),
            "Wonder": EmotionDimensions(0.7, 0.6, 0.3),
            
            # Social emotions
            "Shame": EmotionDimensions(-0.6, 0.4, 0.1),
            "Envy": EmotionDimensions(-0.5, 0.6, 0.4),
            "Pride": EmotionDimensions(0.7, 0.6, 0.8),
            "Guilt": EmotionDimensions(-0.5, 0.5, 0.2),
            
            # Spiritual emotions
            "Reverence": EmotionDimensions(0.6, 0.3, 0.2),
            "Gratitude": EmotionDimensions(0.8, 0.4, 0.5),
            "Humility": EmotionDimensions(0.4, 0.2, 0.1),
            "Peace": EmotionDimensions(0.7, 0.1, 0.4),
            
            # Complex emotions
            "Nostalgia": EmotionDimensions(0.2, 0.4, 0.3),
            "Malice": EmotionDimensions(-0.8, 0.7, 0.9),
            "Empowerment": EmotionDimensions(0.8, 0.7, 0.9),
            "Loneliness": EmotionDimensions(-0.6, 0.3, 0.2),
            
            # Joy sub-emotions
            "Excitement": EmotionDimensions(0.9, 0.9, 0.8),
            "Contentment": EmotionDimensions(0.6, 0.2, 0.5),
            "Euphoria": EmotionDimensions(1.0, 0.9, 0.9),
            "Cheerfulness": EmotionDimensions(0.8, 0.6, 0.7),
            
            # Sadness sub-emotions
            "Melancholy": EmotionDimensions(-0.5, 0.3, 0.2),
            "Grief": EmotionDimensions(-0.9, 0.4, 0.1),
            "Despair": EmotionDimensions(-0.9, 0.5, 0.1),
            "Sorrow": EmotionDimensions(-0.7, 0.3, 0.2),
            
            # Anger sub-emotions
            "Frustration": EmotionDimensions(-0.4, 0.6, 0.6),
            "Rage": EmotionDimensions(-0.9, 1.0, 1.0),
            "Annoyance": EmotionDimensions(-0.3, 0.4, 0.5),
            "Indignation": EmotionDimensions(-0.5, 0.7, 0.8),
            
            # Fear sub-emotions
            "Anxiety": EmotionDimensions(-0.5, 0.7, 0.2),
            "Terror": EmotionDimensions(-0.9, 1.0, 0.1),
            "Panic": EmotionDimensions(-0.8, 1.0, 0.1),
            "Apprehension": EmotionDimensions(-0.4, 0.5, 0.3)
        }
    
    def _build_compound_emotions(self) -> Dict[str, CompoundEmotion]:
        """Define comprehensive compound emotions for better interpolation"""
        return {
            "Bittersweet": CompoundEmotion(
                "Bittersweet", 
                {"Joy": 0.4, "Sadness": 0.6}, 
                "Mixed happiness and sadness, often nostalgic"
            ),
            "Triumphant": CompoundEmotion(
                "Triumphant",
                {"Joy": 0.6, "Pride": 0.8},
                "Victorious joy with sense of accomplishment"
            ),
            "Melancholic Joy": CompoundEmotion(
                "Melancholic Joy",
                {"Joy": 0.3, "Melancholy": 0.7},
                "Happiness tinged with wistful sadness"
            ),
            "Anxious Excitement": CompoundEmotion(
                "Anxious Excitement",
                {"Excitement": 0.6, "Anxiety": 0.4},
                "High energy anticipation with underlying worry"
            ),
            "Grateful Awe": CompoundEmotion(
                "Grateful Awe",
                {"Gratitude": 0.5, "Awe": 0.5},
                "Thankful wonder at something magnificent"
            ),
            "Passionate Love": CompoundEmotion(
                "Passionate Love",
                {"Love": 0.7, "Passion": 0.3},
                "Intense romantic affection"
            ),
            "Righteous Anger": CompoundEmotion(
                "Righteous Anger",
                {"Anger": 0.6, "Pride": 0.4},
                "Anger motivated by moral conviction"
            ),
            "Shameful Guilt": CompoundEmotion(
                "Shameful Guilt",
                {"Shame": 0.5, "Guilt": 0.5},
                "Combined personal shame and regret"
            ),
            "Fearful Anticipation": CompoundEmotion(
                "Fearful Anticipation",
                {"Fear": 0.4, "Anticipation": 0.6},
                "Expecting something with dread"
            ),
            "Peaceful Joy": CompoundEmotion(
                "Peaceful Joy",
                {"Peace": 0.6, "Joy": 0.4},
                "Calm, serene happiness"
            ),
            "Envious Admiration": CompoundEmotion(
                "Envious Admiration",
                {"Envy": 0.6, "Awe": 0.4},
                "Jealous appreciation of someone else's qualities"
            ),
            "Nostalgic Love": CompoundEmotion(
                "Nostalgic Love",
                {"Nostalgia": 0.5, "Love": 0.5},
                "Fondness for past relationships or times"
            )
        }
    
    def _build_context_modifiers(self) -> Dict[str, List[str]]:
        """Build context modifiers that affect emotion interpretation"""
        return {
            "intensifiers": ["very", "extremely", "incredibly", "absolutely"],
            "diminishers": ["slightly", "somewhat", "a bit", "rather"],
            "negators": ["not", "never", "no", "without"]
        }
    
    def _build_sarcasm_indicators(self) -> List[str]:
        """Build sarcasm detection patterns"""
        return [
            r"oh (great|wonderful|fantastic|perfect)",
            r"just (great|perfect|wonderful)",
            r"yeah\s+right",
            r"of\s+course"
        ]
    
    def _detect_sarcasm(self, text: str) -> bool:
        """Detect sarcasm using pattern matching"""
        for pattern in self.sarcasm_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _extract_context_modifiers(self, text: str) -> Dict[str, List[str]]:
        """Extract context modifiers from text"""
        found_modifiers = {mod_type: [] for mod_type in self.context_modifiers.keys()}
        
        for mod_type, modifiers in self.context_modifiers.items():
            for modifier in modifiers:
                if modifier in text:
                    found_modifiers[mod_type].append(modifier)
        
        return found_modifiers
    
    def _detect_hierarchical_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions using hierarchical structure"""
        emotion_scores = {}
        
        for primary_emotion, emotion_data in self.emotion_hierarchy.items():
            emotion_scores[primary_emotion] = 0.0
            
            # Check for sub-emotions (higher weight)
            for sub_emotion, sub_data in emotion_data["sub_emotions"].items():
                for keyword in sub_data["keywords"]:
                    if keyword in text:
                        sub_weight = 2.0 * sub_data["intensity_modifier"]
                        emotion_scores[primary_emotion] += sub_weight
                        emotion_scores[sub_emotion] = emotion_scores.get(sub_emotion, 0) + sub_weight
        
        return emotion_scores
    
    def _detect_compound_emotions(self, text: str, current_emotions: Dict[str, float]) -> Dict[str, float]:
        """Detect compound emotions based on current emotion mix"""
        compound_matches = {}
        
        for compound_name, compound_emotion in self.compound_emotions.items():
            component_strength = 0.0
            total_components = len(compound_emotion.components)
            
            for component, weight in compound_emotion.components.items():
                if component in current_emotions and current_emotions[component] > 0:
                    component_strength += (current_emotions[component] * weight) / total_components
            
            if component_strength > 0.3:
                compound_matches[compound_name] = component_strength
        
        return compound_matches
    
    def _apply_context_modifiers(self, emotions: Dict[str, float], 
                               context_mods: Dict[str, List[str]], 
                               sarcasm_detected: bool) -> Dict[str, float]:
        """Apply context modifiers to emotion weights"""
        modified_emotions = emotions.copy()
        
        # Apply sarcasm reversal
        if sarcasm_detected:
            modified_emotions = self._apply_sarcasm_reversal(modified_emotions)
        
        # Apply intensity modifiers
        intensity_factor = 1.0
        if context_mods["intensifiers"]:
            intensity_factor *= 1.3
        if context_mods["diminishers"]:
            intensity_factor *= 0.7
        if context_mods["negators"]:
            modified_emotions = self._apply_negation(modified_emotions)
        
        # Apply intensity factor
        for emotion in modified_emotions:
            modified_emotions[emotion] *= intensity_factor
        
        return modified_emotions
    
    def _apply_sarcasm_reversal(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Reverse emotional valence for sarcastic statements"""
        valence_flip = {
            "Joy": "Sadness",
            "Love": "Disgust", 
            "Trust": "Fear"
        }
        
        reversed_emotions = {}
        for emotion, weight in emotions.items():
            if emotion in valence_flip and weight > 0:
                reversed_emotions[valence_flip[emotion]] = weight
            elif weight > 0:
                reversed_emotions[emotion] = weight * 0.5
        
        return reversed_emotions
    
    def _apply_negation(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply negation logic to emotions"""
        return {emotion: weight * 0.3 for emotion, weight in emotions.items()} 