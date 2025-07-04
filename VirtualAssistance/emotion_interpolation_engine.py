"""
Emotion Interpolation Engine for VirtualAssistance Music Generation System

This module provides advanced emotional interpolation capabilities for creating smooth
transitions between emotional states in music generation, enabling gradual morphing
of chord progressions, modes, and musical elements based on emotional trajectories.

Version: 1.0
Created: July 2, 2025
Author: VirtualAssistance AI System
"""

import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn.functional as F

class InterpolationMethod(Enum):
    """Available interpolation methods for emotional transitions"""
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    CUBIC_SPLINE = "cubic_spline"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"

class TensionCurveType(Enum):
    """Available tension curve types for consonant/dissonant trajectories"""
    LINEAR = "linear"                    # Direct interpolation
    BUILD = "build"                      # Gradual increase in tension
    RELEASE = "release"                  # Gradual decrease in tension
    PEAK = "peak"                        # Build to maximum then release
    VALLEY = "valley"                    # Release to minimum then build
    WAVE = "wave"                        # Sine wave pattern
    ARCH = "arch"                        # Build-peak-release arc
    INVERTED_ARCH = "inverted_arch"      # Release-valley-build arc

@dataclass
class EmotionState:
    """Represents a complete emotional state with weights and metadata"""
    emotion_weights: Dict[str, float]
    primary_emotion: str
    sub_emotion: Optional[str] = None
    intensity: float = 1.0
    mode: Optional[str] = None
    timestamp: float = 0.0
    consonant_dissonant_value: Optional[float] = None  # CD value (0.0-1.0)
    consonant_dissonant_trajectory: Optional[str] = None  # "towards_consonance", "towards_dissonance", "stable"
    style_context: Optional[str] = None  # Musical style context

@dataclass
class InterpolationPath:
    """Defines a path for emotional interpolation between states"""
    start_state: EmotionState
    end_state: EmotionState
    method: InterpolationMethod
    duration: float
    curve_parameters: Dict = None
    tension_curve_type: Optional[TensionCurveType] = None  # CD trajectory type
    tension_curve_intensity: float = 1.0  # Intensity of tension changes

@dataclass
class InterpolatedProgression:
    """Result of interpolated chord progression generation"""
    chords: List[str]
    emotion_trajectory: List[EmotionState]
    transition_points: List[float]
    consonant_dissonant_trajectory: List[float]  # CD values over time
    tension_curve_analysis: Dict  # Analysis of tension patterns
    metadata: Dict

class EmotionInterpolationEngine:
    """
    Advanced emotion interpolation engine for smooth musical transitions
    
    Features:
    - Multiple interpolation algorithms (linear, cosine, sigmoid, spline)
    - Emotional trajectory planning
    - Progressive chord morphing
    - Modal transition management
    - Real-time emotion blending
    - Intensity scaling and curve shaping
    """
    
    def __init__(self, database_path: str = "emotion_progression_database.json"):
        self.database_path = database_path
        self.emotion_database = self._load_emotion_database()
        
        # Complete 23-emotion system including Transcendence
        self.emotion_labels = [
            "Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
            "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe", "Malice",
            "Arousal", "Guilt", "Reverence", "Wonder", "Dissociation", 
            "Empowerment", "Belonging", "Ideology", "Gratitude", "Transcendence"
        ]
        
        # Emotion-to-mode mapping for interpolation
        self.emotion_modes = {
            "Joy": "Ionian", "Sadness": "Aeolian", "Fear": "Phrygian",
            "Anger": "Phrygian Dominant", "Disgust": "Locrian", "Surprise": "Lydian",
            "Trust": "Dorian", "Anticipation": "Melodic Minor", "Shame": "Harmonic Minor",
            "Love": "Mixolydian", "Envy": "Hungarian Minor", "Aesthetic Awe": "Lydian Augmented",
            "Malice": "Locrian", "Arousal": "Dorian", "Guilt": "Harmonic Minor",
            "Reverence": "Lydian", "Wonder": "Lydian Augmented", "Dissociation": "Locrian",
            "Empowerment": "Ionian", "Belonging": "Dorian", "Ideology": "Dorian", "Gratitude": "Ionian",
            "Transcendence": "Multi-modal"
        }
        
        # Emotional compatibility matrix for smooth transitions
        self.compatibility_matrix = self._build_compatibility_matrix()
        
        # Emotion-to-consonance mapping for CD trajectory generation
        self.emotion_consonance_map = self._build_emotion_consonance_map()
        
    def _load_emotion_database(self) -> Dict:
        """Load the emotion progression database"""
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load emotion database: {e}")
            return {}
    
    def _build_compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build compatibility matrix for emotion transitions"""
        # Define emotional compatibility based on psychological theory
        base_compatibilities = {
            "Joy": {"Love": 0.9, "Gratitude": 0.9, "Empowerment": 0.8, "Wonder": 0.7, "Trust": 0.7},
            "Sadness": {"Guilt": 0.8, "Shame": 0.7, "Envy": 0.6, "Fear": 0.5, "Malice": 0.4},
            "Anger": {"Malice": 0.9, "Disgust": 0.7, "Envy": 0.6, "Ideology": 0.6, "Fear": 0.4},
            "Fear": {"Anxiety": 0.8, "Shame": 0.6, "Dissociation": 0.7, "Sadness": 0.5},
            "Love": {"Joy": 0.9, "Gratitude": 0.8, "Trust": 0.8, "Empowerment": 0.6, "Reverence": 0.6},
            "Malice": {"Anger": 0.9, "Disgust": 0.7, "Envy": 0.8, "Contempt": 0.9},
            "Wonder": {"Joy": 0.7, "Surprise": 0.8, "Aesthetic Awe": 0.9, "Reverence": 0.6, "Transcendence": 0.8},
            "Empowerment": {"Joy": 0.8, "Confidence": 0.9, "Trust": 0.7, "Inspiration": 0.8},
            "Dissociation": {"Numbness": 0.9, "Fear": 0.6, "Sadness": 0.5, "Emptiness": 0.8, "Transcendence": 0.7},
            "Reverence": {"Wonder": 0.6, "Gratitude": 0.7, "Humility": 0.8, "Sacred Peace": 0.9, "Transcendence": 0.9},
            "Transcendence": {"Reverence": 0.9, "Wonder": 0.8, "Aesthetic Awe": 0.8, "Dissociation": 0.7, "Gratitude": 0.6}
        }
        
        # Create symmetric matrix with default compatibility of 0.3
        matrix = {}
        for emotion1 in self.emotion_labels:
            matrix[emotion1] = {}
            for emotion2 in self.emotion_labels:
                if emotion1 == emotion2:
                    matrix[emotion1][emotion2] = 1.0
                elif emotion2 in base_compatibilities.get(emotion1, {}):
                    matrix[emotion1][emotion2] = base_compatibilities[emotion1][emotion2]
                elif emotion1 in base_compatibilities.get(emotion2, {}):
                    matrix[emotion1][emotion2] = base_compatibilities[emotion2][emotion1]
                else:
                    # Default compatibility based on emotional distance
                    matrix[emotion1][emotion2] = 0.3
        
        return matrix
    
    def _build_emotion_consonance_map(self) -> Dict[str, float]:
        """Build emotion-to-consonance mapping for CD trajectory generation"""
        return {
            # Consonant emotions (0.0-0.4)
            "Joy": 0.2, "Trust": 0.25, "Love": 0.3, "Gratitude": 0.2,
            "Empowerment": 0.3, "Belonging": 0.3, "Reverence": 0.25,
            "Transcendence": 0.15,  # Very consonant/ethereal
            
            # Moderately dissonant emotions (0.4-0.6)
            "Anticipation": 0.5, "Surprise": 0.45, "Wonder": 0.4,
            "Aesthetic Awe": 0.4, "Arousal": 0.5, "Ideology": 0.45,
            
            # Highly dissonant emotions (0.6-0.8)
            "Sadness": 0.6, "Shame": 0.6, "Envy": 0.7, "Guilt": 0.65,
            "Anger": 0.75, "Fear": 0.7,
            
            # Extremely dissonant emotions (0.8-1.0)
            "Malice": 0.9, "Disgust": 0.85, "Dissociation": 0.9
        }
    
    def create_emotion_state(self, emotion_weights: Dict[str, float], 
                           timestamp: float = 0.0, 
                           consonant_dissonant_value: Optional[float] = None,
                           style_context: Optional[str] = None) -> EmotionState:
        """Create an EmotionState object from emotion weights"""
        # Handle empty emotion weights gracefully
        if not emotion_weights:
            # Default to neutral/Joy state
            emotion_weights = {"Joy": 0.5}
        
        primary_emotion = max(emotion_weights, key=emotion_weights.get)
        intensity = emotion_weights[primary_emotion]
        mode = self.emotion_modes.get(primary_emotion, "Ionian")
        
        # Calculate consonant/dissonant value if not provided
        if consonant_dissonant_value is None:
            consonant_dissonant_value = self._calculate_cd_value_from_emotions(emotion_weights)
        
        # Determine CD trajectory based on primary emotion
        cd_trajectory = self._determine_cd_trajectory(primary_emotion, emotion_weights)
        
        return EmotionState(
            emotion_weights=emotion_weights,
            primary_emotion=primary_emotion,
            intensity=intensity,
            mode=mode,
            timestamp=timestamp,
            consonant_dissonant_value=consonant_dissonant_value,
            consonant_dissonant_trajectory=cd_trajectory,
            style_context=style_context
        )
    
    def interpolate_emotions(self, start_state: EmotionState, end_state: EmotionState,
                           t: float, method: InterpolationMethod = InterpolationMethod.COSINE) -> EmotionState:
        """
        Interpolate between two emotional states at parameter t (0 to 1)
        
        Args:
            start_state: Starting emotional state
            end_state: Ending emotional state  
            t: Interpolation parameter (0 = start, 1 = end)
            method: Interpolation method to use
            
        Returns:
            Interpolated emotional state
        """
        # Apply interpolation curve
        t_curved = self._apply_interpolation_curve(t, method)
        
        # Interpolate emotion weights
        interpolated_weights = {}
        all_emotions = set(start_state.emotion_weights.keys()) | set(end_state.emotion_weights.keys())
        
        for emotion in all_emotions:
            start_weight = start_state.emotion_weights.get(emotion, 0.0)
            end_weight = end_state.emotion_weights.get(emotion, 0.0)
            
            # Consider emotional compatibility for smoother transitions
            compatibility = self.compatibility_matrix.get(
                start_state.primary_emotion, {}
            ).get(emotion, 0.3)
            
            # Weight interpolation with compatibility factor
            weight_diff = end_weight - start_weight
            interpolated_weight = start_weight + (weight_diff * t_curved * compatibility)
            interpolated_weights[emotion] = max(0.0, interpolated_weight)
        
        # Normalize weights
        total_weight = sum(interpolated_weights.values())
        if total_weight > 0:
            interpolated_weights = {k: v/total_weight for k, v in interpolated_weights.items()}
        
        # Interpolate timestamp
        interpolated_timestamp = start_state.timestamp + (end_state.timestamp - start_state.timestamp) * t
        
        return self.create_emotion_state(interpolated_weights, interpolated_timestamp)
    
    def _calculate_cd_value_from_emotions(self, emotion_weights: Dict[str, float]) -> float:
        """Calculate consonant/dissonant value from weighted emotion mix"""
        total_cd_value = 0.0
        total_weight = 0.0
        
        for emotion, weight in emotion_weights.items():
            if emotion in self.emotion_consonance_map and weight > 0:
                cd_value = self.emotion_consonance_map[emotion]
                total_cd_value += cd_value * weight
                total_weight += weight
        
        return total_cd_value / total_weight if total_weight > 0 else 0.4  # Default moderate consonance
    
    def _determine_cd_trajectory(self, primary_emotion: str, emotion_weights: Dict[str, float]) -> str:
        """Determine consonant/dissonant trajectory direction based on emotional content"""
        primary_cd = self.emotion_consonance_map.get(primary_emotion, 0.4)
        
        # Calculate secondary emotions influence
        sorted_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_emotions) > 1:
            secondary_emotion = sorted_emotions[1][0]
            secondary_cd = self.emotion_consonance_map.get(secondary_emotion, 0.4)
            
            # Determine trajectory based on primary vs secondary consonance
            if abs(primary_cd - secondary_cd) < 0.1:
                return "stable"
            elif secondary_cd > primary_cd:
                return "towards_dissonance"
            else:
                return "towards_consonance"
        
        # Single emotion - determine trajectory based on consonance level
        if primary_cd < 0.3:
            return "stable"  # Already consonant
        elif primary_cd > 0.7:
            return "towards_consonance"  # High dissonance seeks resolution
        else:
            return "stable"  # Moderate tension is stable
    
    def _apply_interpolation_curve(self, t: float, method: InterpolationMethod) -> float:
        """Apply different interpolation curves to the parameter t"""
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
        
        if method == InterpolationMethod.LINEAR:
            return t
        elif method == InterpolationMethod.COSINE:
            return 0.5 * (1 - math.cos(t * math.pi))
        elif method == InterpolationMethod.SIGMOID:
            # Sigmoid curve centered at 0.5
            return 1 / (1 + math.exp(-6 * (t - 0.5)))
        elif method == InterpolationMethod.CUBIC_SPLINE:
            # Smooth cubic interpolation
            return t * t * (3 - 2 * t)
        elif method == InterpolationMethod.EXPONENTIAL:
            # Exponential ease-in
            return math.pow(t, 2)
        elif method == InterpolationMethod.LOGARITHMIC:
            # Logarithmic ease-out
            return 1 - math.pow(1 - t, 2)
        else:
            return t  # Default to linear
    
    def create_tension_curve(self, start_cd: float, end_cd: float, steps: int,
                           curve_type: TensionCurveType = TensionCurveType.LINEAR,
                           intensity: float = 1.0) -> List[float]:
        """
        Create consonant/dissonant tension curves
        
        Args:
            start_cd: Starting consonant/dissonant value (0.0-1.0)
            end_cd: Ending consonant/dissonant value (0.0-1.0)
            steps: Number of steps in the curve
            curve_type: Type of tension curve to generate
            intensity: Intensity of tension changes (0.0-2.0)
            
        Returns:
            List of CD values representing the tension curve
        """
        if steps < 2:
            return [start_cd, end_cd]
        
        curve = []
        
        for i in range(steps):
            t = i / (steps - 1)  # Normalize to 0-1
            
            if curve_type == TensionCurveType.LINEAR:
                cd_value = start_cd + (end_cd - start_cd) * t
                
            elif curve_type == TensionCurveType.BUILD:
                # Exponential build-up to maximum tension
                max_tension = max(start_cd, end_cd, 0.8)
                cd_value = start_cd + (max_tension - start_cd) * (t ** (2 / intensity))
                if i == steps - 1:  # End at target
                    cd_value = end_cd
                    
            elif curve_type == TensionCurveType.RELEASE:
                # Exponential release from tension
                cd_value = start_cd + (end_cd - start_cd) * (1 - (1 - t) ** (2 / intensity))
                
            elif curve_type == TensionCurveType.PEAK:
                # Build to peak then release
                if t < 0.5:
                    # Build phase
                    peak_value = min(1.0, max(start_cd, end_cd) + 0.3 * intensity)
                    cd_value = start_cd + (peak_value - start_cd) * (2 * t) ** 2
                else:
                    # Release phase
                    peak_value = min(1.0, max(start_cd, end_cd) + 0.3 * intensity)
                    release_t = (t - 0.5) * 2
                    cd_value = peak_value + (end_cd - peak_value) * release_t ** 0.5
                    
            elif curve_type == TensionCurveType.VALLEY:
                # Release to valley then build
                if t < 0.5:
                    # Release phase
                    valley_value = max(0.0, min(start_cd, end_cd) - 0.3 * intensity)
                    cd_value = start_cd + (valley_value - start_cd) * (2 * t) ** 0.5
                else:
                    # Build phase
                    valley_value = max(0.0, min(start_cd, end_cd) - 0.3 * intensity)
                    build_t = (t - 0.5) * 2
                    cd_value = valley_value + (end_cd - valley_value) * build_t ** 2
                    
            elif curve_type == TensionCurveType.WAVE:
                # Sine wave pattern
                base_value = (start_cd + end_cd) / 2
                amplitude = abs(end_cd - start_cd) / 2 * intensity
                cd_value = base_value + amplitude * math.sin(t * math.pi * 2)
                
            elif curve_type == TensionCurveType.ARCH:
                # Smooth arc: build-peak-release
                peak_value = max(start_cd, end_cd) + 0.2 * intensity
                cd_value = start_cd + (peak_value - start_cd) * math.sin(t * math.pi)
                if i == steps - 1:  # End at target
                    cd_value = end_cd
                    
            elif curve_type == TensionCurveType.INVERTED_ARCH:
                # Inverted arc: release-valley-build
                valley_value = min(start_cd, end_cd) - 0.2 * intensity
                valley_value = max(0.0, valley_value)  # Clamp to valid range
                cd_value = start_cd - (start_cd - valley_value) * math.sin(t * math.pi)
                if i == steps - 1:  # End at target
                    cd_value = end_cd
                    
            else:
                # Default to linear
                cd_value = start_cd + (end_cd - start_cd) * t
            
            # Clamp to valid range
            cd_value = max(0.0, min(1.0, cd_value))
            curve.append(cd_value)
        
        return curve
    
    def interpolate_emotions_with_cd(self, start_state: EmotionState, end_state: EmotionState,
                                   t: float, method: InterpolationMethod = InterpolationMethod.COSINE,
                                   tension_curve_type: Optional[TensionCurveType] = None) -> EmotionState:
        """
        Enhanced emotion interpolation with consonant/dissonant trajectory support
        
        Args:
            start_state: Starting emotional state
            end_state: Ending emotional state
            t: Interpolation parameter (0 = start, 1 = end)
            method: Interpolation method for emotions
            tension_curve_type: Optional tension curve type for CD trajectory
            
        Returns:
            Interpolated emotional state with CD trajectory
        """
        # Standard emotion interpolation
        interpolated_state = self.interpolate_emotions(start_state, end_state, t, method)
        
        # Handle consonant/dissonant interpolation
        if start_state.consonant_dissonant_value is not None and end_state.consonant_dissonant_value is not None:
            if tension_curve_type is not None:
                # Use tension curve for CD interpolation
                cd_curve = self.create_tension_curve(
                    start_state.consonant_dissonant_value,
                    end_state.consonant_dissonant_value,
                    steps=100,  # High resolution for smooth interpolation
                    curve_type=tension_curve_type
                )
                # Sample from curve at parameter t
                curve_index = int(t * (len(cd_curve) - 1))
                interpolated_state.consonant_dissonant_value = cd_curve[curve_index]
            else:
                # Linear CD interpolation
                interpolated_state.consonant_dissonant_value = (
                    start_state.consonant_dissonant_value + 
                    (end_state.consonant_dissonant_value - start_state.consonant_dissonant_value) * t
                )
        
        # Interpolate CD trajectory
        if t < 0.5:
            interpolated_state.consonant_dissonant_trajectory = start_state.consonant_dissonant_trajectory
        else:
            interpolated_state.consonant_dissonant_trajectory = end_state.consonant_dissonant_trajectory
        
        # Interpolate style context
        if t < 0.5:
            interpolated_state.style_context = start_state.style_context
        else:
            interpolated_state.style_context = end_state.style_context
        
        return interpolated_state
    
    def create_emotion_trajectory(self, emotion_states: List[EmotionState],
                                duration: float, num_steps: int = 16,
                                method: InterpolationMethod = InterpolationMethod.COSINE,
                                tension_curve_type: Optional[TensionCurveType] = None) -> List[EmotionState]:
        """
        Create a smooth trajectory through multiple emotional states
        
        Args:
            emotion_states: List of emotional waypoints
            duration: Total duration of the trajectory
            num_steps: Number of interpolation steps
            method: Interpolation method to use
            
        Returns:
            List of interpolated emotional states
        """
        if len(emotion_states) < 2:
            return emotion_states
        
        trajectory = []
        segment_duration = duration / (len(emotion_states) - 1)
        steps_per_segment = num_steps // (len(emotion_states) - 1)
        
        for i in range(len(emotion_states) - 1):
            start_state = emotion_states[i]
            end_state = emotion_states[i + 1]
            
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                timestamp = (i * segment_duration) + (t * segment_duration)
                
                # Use enhanced interpolation with CD support if available
                if tension_curve_type is not None:
                    interpolated_state = self.interpolate_emotions_with_cd(
                        start_state, end_state, t, method, tension_curve_type
                    )
                else:
                    interpolated_state = self.interpolate_emotions(start_state, end_state, t, method)
                
                interpolated_state.timestamp = timestamp
                trajectory.append(interpolated_state)
        
        # Add final state
        final_state = emotion_states[-1]
        final_state.timestamp = duration
        trajectory.append(final_state)
        
        return trajectory
    
    def generate_interpolated_progression(self, start_emotion: Dict[str, float], 
                                        end_emotion: Dict[str, float],
                                        progression_length: int = 8,
                                        method: InterpolationMethod = InterpolationMethod.COSINE,
                                        tension_curve_type: Optional[TensionCurveType] = None,
                                        style_context: Optional[str] = None) -> InterpolatedProgression:
        """
        Generate a chord progression that smoothly transitions between emotions
        
        Args:
            start_emotion: Starting emotion weights
            end_emotion: Ending emotion weights  
            progression_length: Number of chords in progression
            method: Interpolation method
            
        Returns:
            InterpolatedProgression object with chords and emotional trajectory
        """
        # Create emotion states with CD support
        start_state = self.create_emotion_state(start_emotion, style_context=style_context)
        end_state = self.create_emotion_state(end_emotion, style_context=style_context)
        
        # Create emotional trajectory
        trajectory = []
        chords = []
        transition_points = []
        cd_trajectory = []
        
        for i in range(progression_length):
            t = i / (progression_length - 1) if progression_length > 1 else 0
            
            # Enhanced interpolation with CD support
            if tension_curve_type is not None:
                current_state = self.interpolate_emotions_with_cd(
                    start_state, end_state, t, method, tension_curve_type
                )
            else:
                current_state = self.interpolate_emotions(start_state, end_state, t, method)
            
            trajectory.append(current_state)
            transition_points.append(t)
            
            # Track CD trajectory
            if current_state.consonant_dissonant_value is not None:
                cd_trajectory.append(current_state.consonant_dissonant_value)
            else:
                cd_trajectory.append(0.4)  # Default moderate consonance
            
            # Generate chord for current emotional state
            chord = self._select_chord_for_emotion_state(current_state, i)
            chords.append(chord)
        
        # Analyze tension curve
        tension_curve_analysis = self._analyze_tension_curve(cd_trajectory, tension_curve_type)
        
        return InterpolatedProgression(
            chords=chords,
            emotion_trajectory=trajectory,
            transition_points=transition_points,
            consonant_dissonant_trajectory=cd_trajectory,
            tension_curve_analysis=tension_curve_analysis,
            metadata={
                "start_emotion": start_emotion,
                "end_emotion": end_emotion,
                "method": method.value,
                "progression_length": progression_length,
                "tension_curve_type": tension_curve_type.value if tension_curve_type else None,
                "style_context": style_context
            }
        )
    
    def _select_chord_for_emotion_state(self, emotion_state: EmotionState, position: int) -> str:
        """Select appropriate chord for given emotional state and position"""
        primary_emotion = emotion_state.primary_emotion
        
        # Get progressions for primary emotion from database
        emotion_data = self.emotion_database.get('emotions', {}).get(primary_emotion, {})
        
        if 'sub_emotions' in emotion_data:
            # Find best matching sub-emotion
            best_sub_emotion = None
            best_weight = 0
            
            for sub_emotion_name, sub_emotion_data in emotion_data['sub_emotions'].items():
                if 'progression_pool' in sub_emotion_data:
                    weight = emotion_state.emotion_weights.get(primary_emotion, 0)
                    if weight > best_weight:
                        best_weight = weight
                        best_sub_emotion = sub_emotion_data
            
            if best_sub_emotion and 'progression_pool' in best_sub_emotion:
                progressions = best_sub_emotion['progression_pool']
                if progressions:
                    # Select progression and get chord at position
                    progression = progressions[0]  # Use first progression for now
                    chords = progression.get('chords', ['I', 'vi', 'IV', 'V'])
                    return chords[position % len(chords)]
        
        # Fallback to basic emotional chord mapping
        emotion_chord_map = {
            "Joy": ["I", "vi", "IV", "V"],
            "Sadness": ["i", "♭VI", "♭VII", "i"],
            "Anger": ["i", "♭II", "V", "i"],
            "Fear": ["i", "♭ii°", "V", "i"],
            "Love": ["I", "iii", "vi", "IV"],
            "Malice": ["i", "♭ii°", "v°", "i"],
            "Wonder": ["I", "♯IV°", "V", "I"],
            "Empowerment": ["I", "IV", "V7", "I"],
            "Reverence": ["I", "IV", "I", "IV"],
            "Gratitude": ["I", "vi", "IV", "V"]
        }
        
        default_progression = emotion_chord_map.get(primary_emotion, ["I", "vi", "IV", "V"])
        return default_progression[position % len(default_progression)]
    
    def _analyze_tension_curve(self, cd_trajectory: List[float], 
                             tension_curve_type: Optional[TensionCurveType] = None) -> Dict:
        """Analyze the tension curve patterns in a CD trajectory"""
        if not cd_trajectory:
            return {"error": "Empty CD trajectory"}
        
        analysis = {
            "curve_type": tension_curve_type.value if tension_curve_type else "linear",
            "start_tension": cd_trajectory[0],
            "end_tension": cd_trajectory[-1],
            "max_tension": max(cd_trajectory),
            "min_tension": min(cd_trajectory),
            "average_tension": sum(cd_trajectory) / len(cd_trajectory),
            "tension_range": max(cd_trajectory) - min(cd_trajectory),
            "tension_direction": "increasing" if cd_trajectory[-1] > cd_trajectory[0] else "decreasing" if cd_trajectory[-1] < cd_trajectory[0] else "stable"
        }
        
        # Calculate tension peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(cd_trajectory) - 1):
            current = cd_trajectory[i]
            prev = cd_trajectory[i-1]
            next_val = cd_trajectory[i+1]
            
            # Peak detection
            if current > prev and current > next_val:
                peaks.append({"position": i, "value": current})
            
            # Valley detection
            if current < prev and current < next_val:
                valleys.append({"position": i, "value": current})
        
        analysis["peaks"] = peaks
        analysis["valleys"] = valleys
        analysis["peak_count"] = len(peaks)
        analysis["valley_count"] = len(valleys)
        
        # Calculate tension stability
        if len(cd_trajectory) > 1:
            # Calculate variance to measure stability
            mean_tension = analysis["average_tension"]
            variance = sum((x - mean_tension) ** 2 for x in cd_trajectory) / len(cd_trajectory)
            analysis["tension_stability"] = 1.0 - min(1.0, variance)  # Higher value = more stable
        else:
            analysis["tension_stability"] = 1.0
        
        # Musical interpretation
        if analysis["average_tension"] < 0.3:
            analysis["musical_character"] = "consonant"
        elif analysis["average_tension"] < 0.6:
            analysis["musical_character"] = "moderate"
        else:
            analysis["musical_character"] = "dissonant"
        
        # Suggested resolution
        if analysis["end_tension"] > 0.6:
            analysis["resolution_needed"] = True
            analysis["suggested_resolution"] = "consonant_resolution"
        else:
            analysis["resolution_needed"] = False
            analysis["suggested_resolution"] = "stable"
        
        return analysis
    
    def blend_progressions(self, progressions: List[List[str]], 
                         weights: List[float]) -> List[str]:
        """
        Blend multiple chord progressions based on weights
        
        Args:
            progressions: List of chord progressions to blend
            weights: Weights for each progression (should sum to 1.0)
            
        Returns:
            Blended chord progression
        """
        if not progressions or not weights:
            return ["I", "vi", "IV", "V"]  # Default progression
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Find the longest progression to use as base length
        max_length = max(len(prog) for prog in progressions)
        
        # For now, return the progression with highest weight
        # Future enhancement: implement actual chord blending algorithms
        max_weight_index = weights.index(max(weights))
        base_progression = progressions[max_weight_index]
        
        # Extend to max length if needed
        while len(base_progression) < max_length:
            base_progression.extend(base_progression)
        
        return base_progression[:max_length]
    
    def create_emotional_morph(self, start_text: str, end_text: str,
                             steps: int = 8, method: InterpolationMethod = InterpolationMethod.COSINE) -> Dict:
        """
        Create a complete emotional morph from start text to end text
        
        Args:
            start_text: Starting emotional text
            end_text: Ending emotional text
            steps: Number of interpolation steps
            method: Interpolation method
            
        Returns:
            Dictionary containing complete morph data
        """
        # This would integrate with the existing emotion parser
        # For now, create a simplified version
        
        # Parse emotions from text (simplified)
        start_emotions = self._simple_emotion_parse(start_text)
        end_emotions = self._simple_emotion_parse(end_text)
        
        # Generate interpolated progression
        interpolated_prog = self.generate_interpolated_progression(
            start_emotions, end_emotions, steps, method
        )
        
        return {
            "start_text": start_text,
            "end_text": end_text,
            "start_emotions": start_emotions,
            "end_emotions": end_emotions,
            "interpolated_progression": interpolated_prog,
            "method": method.value,
            "steps": steps
        }
    
    def _simple_emotion_parse(self, text: str) -> Dict[str, float]:
        """Simplified emotion parsing for interpolation demo"""
        text_lower = text.lower()
        emotion_weights = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Simple keyword matching
        emotion_keywords = {
            "Joy": ["happy", "joy", "joyful", "excited", "cheerful"],
            "Sadness": ["sad", "depressed", "down", "melancholy", "blue"],
            "Anger": ["angry", "mad", "furious", "rage", "irritated"],
            "Fear": ["afraid", "scared", "anxious", "terrified", "nervous"],
            "Love": ["love", "romantic", "affection", "caring", "tender"],
            "Malice": ["evil", "cruel", "wicked", "malicious", "sinister"],
            "Wonder": ["wonder", "curious", "amazed", "fascinated", "intrigued"],
            "Empowerment": ["strong", "confident", "empowered", "bold", "powerful"],
            "Gratitude": ["grateful", "thankful", "blessed", "appreciative"],
            "Reverence": ["sacred", "holy", "divine", "reverent", "spiritual"]
        }
        
        matches = 0
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_weights[emotion] += 1
                    matches += 1
        
        # Normalize
        if matches > 0:
            emotion_weights = {k: v/matches for k, v in emotion_weights.items()}
        else:
            emotion_weights["Joy"] = 1.0  # Default
        
        return emotion_weights

    # HIGH PRIORITY ADDITION 1: Real-time chord progression morphing
    def morph_progressions_realtime(self, start_progression: List[str], end_progression: List[str],
                                  num_steps: int = 8, preserve_voice_leading: bool = True) -> List[Dict]:
        """
        Real-time morphing between chord progressions with voice leading preservation
        
        Args:
            start_progression: Starting chord progression
            end_progression: Target chord progression  
            num_steps: Number of morphing steps
            preserve_voice_leading: Whether to preserve smooth voice leading
            
        Returns:
            List of morphed progressions with metadata
        """
        # Normalize progression lengths
        max_length = max(len(start_progression), len(end_progression))
        start_padded = (start_progression * (max_length // len(start_progression) + 1))[:max_length]
        end_padded = (end_progression * (max_length // len(end_progression) + 1))[:max_length]
        
        morphed_progressions = []
        
        for step in range(num_steps):
            t = step / (num_steps - 1) if num_steps > 1 else 0
            
            morphed_chords = []
            for i in range(max_length):
                start_chord = start_padded[i]
                end_chord = end_padded[i]
                
                if preserve_voice_leading:
                    morphed_chord = self._morph_chord_with_voice_leading(start_chord, end_chord, t)
                else:
                    morphed_chord = end_chord if t > 0.5 else start_chord
                
                morphed_chords.append(morphed_chord)
            
            morphed_progressions.append({
                'progression': morphed_chords,
                'morph_position': t,
                'step': step,
                'voice_leading_score': self._calculate_voice_leading_score(morphed_chords)
            })
        
        return morphed_progressions
    
    def _morph_chord_with_voice_leading(self, start_chord: str, end_chord: str, t: float) -> str:
        """Morph between chords preserving voice leading"""
        # Simple implementation - can be enhanced with music theory
        if t < 0.3:
            return start_chord
        elif t > 0.7:
            return end_chord
        else:
            # Find intermediate chord that bridges the two
            return self._find_bridge_chord(start_chord, end_chord)
    
    def _find_bridge_chord(self, chord1: str, chord2: str) -> str:
        """Find a chord that bridges between two chords"""
        # Simplified bridge chord logic - can be enhanced
        bridge_chords = {
            ('I', 'vi'): 'iii',
            ('vi', 'IV'): 'ii',
            ('IV', 'V'): 'ii',
            ('V', 'I'): 'V7',
            ('i', 'iv'): 'ii°',
            ('iv', 'VII'): 'VI'
        }
        return bridge_chords.get((chord1, chord2), chord2)
    
    def _calculate_voice_leading_score(self, progression: List[str]) -> float:
        """Calculate voice leading smoothness score"""
        # Simplified scoring - can be enhanced with actual voice leading analysis
        return 0.8  # Placeholder
    
    # HIGH PRIORITY ADDITION 2: Multi-emotion simultaneous blending
    def blend_multiple_emotions(self, emotion_states: List[EmotionState], 
                              weights: List[float]) -> EmotionState:
        """
        Blend multiple emotional states simultaneously with given weights
        
        Args:
            emotion_states: List of emotion states to blend
            weights: Blending weights (should sum to 1.0)
            
        Returns:
            Blended emotional state
        """
        if len(emotion_states) != len(weights):
            raise ValueError("Number of states must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else weights
        
        # Blend emotion weights
        blended_emotions = {}
        all_emotions = set()
        for state in emotion_states:
            all_emotions.update(state.emotion_weights.keys())
        
        for emotion in all_emotions:
            blended_value = 0.0
            for state, weight in zip(emotion_states, normalized_weights):
                emotion_value = state.emotion_weights.get(emotion, 0.0)
                blended_value += emotion_value * weight
            blended_emotions[emotion] = blended_value
        
        # Calculate blended intensity
        blended_intensity = sum(state.intensity * weight for state, weight in zip(emotion_states, normalized_weights))
        
        return self.create_emotion_state(blended_emotions, 0.0)
    
    # HIGH PRIORITY ADDITION 3: Direct model integration
    def integrate_with_progression_model(self, progression_model):
        """
        Direct integration with the chord progression model for seamless operation
        
        Args:
            progression_model: ChordProgressionModel instance
        """
        self.progression_model = progression_model
        self.individual_model = getattr(progression_model, 'individual_model', None)
        
        print("✅ Interpolation engine integrated with progression model")
    
    def generate_morphed_progression_from_text(self, start_text: str, end_text: str,
                                             num_steps: int = 8, genre: str = "Pop") -> List[Dict]:
        """
        Generate morphed progression directly from text prompts using integrated model
        
        Args:
            start_text: Starting emotional description
            end_text: Ending emotional description
            num_steps: Number of morphing steps
            genre: Musical genre preference
            
        Returns:
            List of progressions representing the emotional morph
        """
        if not hasattr(self, 'progression_model'):
            raise ValueError("Must call integrate_with_progression_model() first")
        
        # Generate start and end progressions
        start_result = self.progression_model.generate_from_prompt(start_text, genre)[0]
        end_result = self.progression_model.generate_from_prompt(end_text, genre)[0]
        
        start_progression = start_result['chords']
        end_progression = end_result['chords']
        
        # Morph between progressions
        morphed = self.morph_progressions_realtime(start_progression, end_progression, num_steps)
        
        # Add emotional context to each step
        for i, step in enumerate(morphed):
            t = i / (num_steps - 1) if num_steps > 1 else 0
            
            # Interpolate emotion weights
            start_emotions = start_result['emotion_weights']
            end_emotions = end_result['emotion_weights']
            
            interpolated_emotions = {}
            all_emotions = set(start_emotions.keys()) | set(end_emotions.keys())
            for emotion in all_emotions:
                start_val = start_emotions.get(emotion, 0.0)
                end_val = end_emotions.get(emotion, 0.0)
                interpolated_emotions[emotion] = start_val + (end_val - start_val) * t
            
            step['emotion_weights'] = interpolated_emotions
            step['primary_emotion'] = max(interpolated_emotions, key=interpolated_emotions.get)
            step['start_text'] = start_text
            step['end_text'] = end_text
            step['genre'] = genre
        
        return morphed
    
    # HIGH PRIORITY ADDITION 4: Sub-emotion interpolation support
    def interpolate_sub_emotions(self, start_emotion: str, start_sub: str,
                               end_emotion: str, end_sub: str, t: float) -> Tuple[str, str]:
        """
        Interpolate between sub-emotions with psychological awareness
        
        Args:
            start_emotion: Starting primary emotion
            start_sub: Starting sub-emotion
            end_emotion: Ending primary emotion  
            end_sub: Ending sub-emotion
            t: Interpolation parameter (0 to 1)
            
        Returns:
            Tuple of (interpolated_emotion, interpolated_sub_emotion)
        """
        # Sub-emotion transition mappings for smooth psychological transitions
        sub_emotion_bridges = {
            ('Joy:Excitement', 'Sadness:Melancholy'): ['Joy:Contentment', 'Trust:Comfort', 'Sadness:Wistfulness'],
            ('Malice:Cruelty', 'Gratitude:Thankfulness'): ['Anger:Frustration', 'Shame:Regret', 'Guilt:Remorse'],
            ('Fear:Anxiety', 'Empowerment:Confidence'): ['Fear:Caution', 'Trust:Security', 'Empowerment:Determination'],
            ('Arousal:Lust', 'Reverence:Sacred Peace'): ['Love:Passion', 'Love:Devotion', 'Reverence:Humility']
        }
        
        start_key = f"{start_emotion}:{start_sub}"
        end_key = f"{end_emotion}:{end_sub}"
        bridge_key = (start_key, end_key)
        
        if bridge_key in sub_emotion_bridges:
            bridges = sub_emotion_bridges[bridge_key]
            # Find appropriate bridge based on t value
            if t < 0.25:
                return start_emotion, start_sub
            elif t < 0.5:
                bridge = bridges[0].split(':')
                return bridge[0], bridge[1]
            elif t < 0.75:
                bridge = bridges[1].split(':') if len(bridges) > 1 else bridges[0].split(':')
                return bridge[0], bridge[1]
            else:
                return end_emotion, end_sub
        else:
            # Simple interpolation when no bridge defined
            return (end_emotion, end_sub) if t > 0.5 else (start_emotion, start_sub)
    
    def create_sub_emotion_trajectory(self, emotion_path: List[Tuple[str, str]], 
                                    num_steps: int = 8) -> List[Tuple[str, str]]:
        """
        Create smooth trajectory through sub-emotions
        
        Args:
            emotion_path: List of (emotion, sub_emotion) tuples
            num_steps: Total steps in trajectory
            
        Returns:
            List of interpolated (emotion, sub_emotion) tuples
        """
        if len(emotion_path) < 2:
            return emotion_path
        
        trajectory = []
        steps_per_segment = num_steps // (len(emotion_path) - 1)
        
        for i in range(len(emotion_path) - 1):
            start_emotion, start_sub = emotion_path[i]
            end_emotion, end_sub = emotion_path[i + 1]
            
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                interpolated = self.interpolate_sub_emotions(start_emotion, start_sub, end_emotion, end_sub, t)
                trajectory.append(interpolated)
        
        # Add final point
        trajectory.append(emotion_path[-1])
        
        return trajectory

# Example usage and testing functions
def demo_interpolation():
    """Demonstrate the interpolation engine capabilities"""
    print("🎼 Emotion Interpolation Engine Demo")
    print("=" * 40)
    
    engine = EmotionInterpolationEngine()
    
    # Create sample emotional states
    happy_state = engine.create_emotion_state({"Joy": 0.8, "Love": 0.2})
    sad_state = engine.create_emotion_state({"Sadness": 0.7, "Guilt": 0.3})
    
    print("Starting State:", happy_state.primary_emotion, f"({happy_state.intensity:.2f})")
    print("Ending State:", sad_state.primary_emotion, f"({sad_state.intensity:.2f})")
    
    # Demonstrate interpolation
    print("\nInterpolation Steps:")
    for i in range(5):
        t = i / 4
        interpolated = engine.interpolate_emotions(happy_state, sad_state, t)
        print(f"t={t:.2f}: {interpolated.primary_emotion} ({interpolated.intensity:.2f})")
    
    # Generate interpolated progression
    print("\nGenerating Interpolated Progression...")
    progression = engine.generate_interpolated_progression(
        {"Joy": 1.0}, {"Sadness": 1.0}, 8
    )
    
    print("Chords:", progression.chords)
    print("Transition Points:", [f"{tp:.2f}" for tp in progression.transition_points])

if __name__ == "__main__":
    demo_interpolation() 