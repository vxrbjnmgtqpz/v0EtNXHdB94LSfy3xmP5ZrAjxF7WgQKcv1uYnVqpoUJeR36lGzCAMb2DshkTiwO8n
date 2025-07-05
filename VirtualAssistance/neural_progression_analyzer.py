"""
Neural Progression Analyzer - Contextual Integration Layer
This module bridges individual chord and progression models to:
1. Analyze existing progressions through individual chord context
2. Generate novel progressions using learned patterns
3. Provide contextual weighting for chord-to-emotion translation

Architecture:
- Progression Context Analyzer: Breaks down progressions into contextual chord components
- Neural Pattern Learner: Learns progression patterns from combined models
- Novel Progression Generator: Creates new progressions not explicitly in database
- Contextual Emotion Weighter: Adjusts individual chord emotions based on progression context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
import time

# Import our existing models
from individual_chord_model import IndividualChordModel, IndividualChord
from chord_progression_model import ChordProgressionModel, ChordProgression

@dataclass
class ContextualChordAnalysis:
    """Analysis of an individual chord within progression context"""
    chord_symbol: str
    roman_numeral: str
    position_in_progression: int
    base_emotions: Dict[str, float]  # From individual model
    contextual_emotions: Dict[str, float]  # Adjusted for progression context
    functional_role: str  # e.g., "tonic", "dominant", "subdominant", "transitional"
    harmonic_tension: float  # 0.0 to 1.0
    contextual_weight: float  # How much this chord contributes to overall progression emotion
    consonant_dissonant_value: float  # 0.0 to 1.0 (0.0=consonant, 1.0=dissonant)
    consonant_dissonant_context: str  # Description of CD role in progression

@dataclass
class ProgressionAnalysis:
    """Complete analysis of a chord progression"""
    chords: List[str]
    progression_id: str
    overall_emotion_weights: Dict[str, float]
    contextual_chord_analyses: List[ContextualChordAnalysis]
    harmonic_flow: List[float]  # Tension curve throughout progression
    consonant_dissonant_trajectory: List[float]  # CD values throughout progression
    novel_pattern_score: float  # How different this is from known patterns
    generation_confidence: float  # How confident the model is in this progression
    average_consonant_dissonant: float  # Overall CD character of progression
    cd_flow_description: str  # Description of how CD changes through progression

class NeuralProgressionAnalyzer(nn.Module):
    """
    Neural network that learns to:
    1. Predict progression-level emotions from chord sequences
    2. Adjust individual chord emotions based on context
    3. Generate novel progressions through learned patterns
    """
    
    def __init__(self, 
                 chord_vocab_size: int = 100,  # Max number of unique chords
                 emotion_dim: int = 23,        # 23 core emotions (expanded system including Transcendence)
                 hidden_dim: int = 256,
                 context_window: int = 8):
        super().__init__()
        
        self.chord_vocab_size = chord_vocab_size
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        
        # Embedding layers
        self.chord_embedding = nn.Embedding(chord_vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(context_window, hidden_dim)
        
        # Contextual encoder (LSTM for sequence processing)
        self.context_encoder = nn.LSTM(
            hidden_dim * 2,  # chord + position embeddings
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism for contextual weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Emotion prediction heads
        self.progression_emotion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        self.contextual_chord_emotion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        # Pattern novelty estimator
        self.novelty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Harmonic tension estimator
        self.tension_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Consonant/Dissonant prediction head
        self.consonant_dissonant_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Update emotion labels for 23-emotion system including Transcendence
        self.emotion_labels = ["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
                              "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe", "Malice",
                              "Arousal", "Guilt", "Reverence", "Wonder", "Dissociation", 
                              "Empowerment", "Belonging", "Ideology", "Gratitude", "Transcendence"]
        
    def forward(self, chord_indices: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the neural progression analyzer
        
        Args:
            chord_indices: Tensor of shape (batch_size, sequence_length)
            positions: Tensor of shape (batch_size, sequence_length)
            
        Returns:
            Dictionary containing all predictions
        """
        batch_size, seq_len = chord_indices.shape
        
        # Get embeddings
        chord_embeds = self.chord_embedding(chord_indices)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        combined_embeds = torch.cat([chord_embeds, pos_embeds], dim=-1)
        
        # Encode context
        context_output, (hidden, cell) = self.context_encoder(combined_embeds)
        
        # Apply attention for contextual weighting
        attended_output, attention_weights = self.attention(
            context_output, context_output, context_output
        )
        
        # Global progression emotion (from final hidden state)
        progression_emotion = self.progression_emotion_head(attended_output[:, -1, :])
        
        # Contextual chord emotions (for each position)
        contextual_emotions = []
        for i in range(seq_len):
            # Combine position encoding with global progression context
            position_context = torch.cat([
                attended_output[:, i, :],
                progression_emotion
            ], dim=-1)
            contextual_emotion = self.contextual_chord_emotion_head(position_context)
            contextual_emotions.append(contextual_emotion)
        
        contextual_emotions = torch.stack(contextual_emotions, dim=1)
        
        # Pattern novelty (how different this progression is from training patterns)
        novelty_score = self.novelty_estimator(attended_output[:, -1, :])
        
        # Harmonic tension throughout progression
        tension_scores = self.tension_estimator(attended_output)
        
        # Consonant/Dissonant scores throughout progression
        consonant_dissonant_scores = self.consonant_dissonant_estimator(attended_output)
        
        return {
            'progression_emotions': progression_emotion,
            'contextual_chord_emotions': contextual_emotions,
            'attention_weights': attention_weights,
            'novelty_scores': novelty_score,
            'tension_scores': tension_scores,
            'consonant_dissonant_scores': consonant_dissonant_scores
        }

class ContextualProgressionIntegrator:
    """
    Main integration class that combines individual chord and progression models
    with the neural analyzer to provide contextual chord-emotion weighting
    """
    
    def __init__(self):
        # Load existing models
        self.individual_model = IndividualChordModel()
        self.progression_model = ChordProgressionModel()
        
        # Initialize neural analyzer
        self.neural_analyzer = NeuralProgressionAnalyzer()
        
        # Add emotion labels for compatibility with 23-emotion system
        self.emotion_labels = ["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
                              "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe", "Malice",
                              "Arousal", "Guilt", "Reverence", "Wonder", "Dissociation", 
                              "Empowerment", "Belonging", "Ideology", "Gratitude", "Transcendence"]
        
        # Chord vocabulary for neural network
        self.chord_vocab = self._build_chord_vocabulary()
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(self.chord_vocab)}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        
        # Training data for neural network
        self.training_data = []
        self._prepare_training_data()
        
    def _build_chord_vocabulary(self) -> List[str]:
        """Build vocabulary of all chords from the progression database"""
        vocab = set()
        
        try:
            with open('emotion_progression_database.json', 'r') as f:
                prog_data = json.load(f)
            
            for emotion_name, emotion_data in prog_data['emotions'].items():
                # Handle both old structure (direct progression_pool) and new structure (sub_emotions)
                progressions_to_process = []
                
                if 'progression_pool' in emotion_data:
                    # Old structure - direct progression pool
                    progressions_to_process.extend(emotion_data['progression_pool'])
                
                if 'sub_emotions' in emotion_data:
                    # New structure - progressions in sub-emotions
                    for sub_emotion_name, sub_emotion_data in emotion_data['sub_emotions'].items():
                        if 'progression_pool' in sub_emotion_data:
                            progressions_to_process.extend(sub_emotion_data['progression_pool'])
                
                # Process all collected progressions
                for prog in progressions_to_process:
                    if isinstance(prog, dict) and 'chords' in prog:
                        for chord in prog['chords']:
                            vocab.add(chord)
        except Exception as e:
            print(f"Warning: Could not load progression database for vocabulary: {e}")
            # Fallback vocabulary
            vocab = {
                'I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°',
                'i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII',
                '♭II', '♭III', '♭VI', '♭VII', 'N6', 'Aug6',
                'V7', 'vii7', 'I7', 'ii7', 'iii7', 'IV7', 'vi7'
            }
        
        # Add padding token at index 0
        vocab.add('<PAD>')
        
        return sorted(list(vocab))
    
    def _prepare_training_data(self):
        """Prepare training data from existing databases"""
        # Load progression database
        try:
            with open('emotion_progression_database.json', 'r') as f:
                prog_data = json.load(f)
            
            for emotion_name, emotion_data in prog_data['emotions'].items():
                emotion_vector = self._emotion_name_to_vector(emotion_name)
                
                # Handle both old structure (direct progression_pool) and new structure (sub_emotions)
                progressions_to_process = []
                
                if 'progression_pool' in emotion_data:
                    # Old structure - direct progression pool
                    progressions_to_process.extend(emotion_data['progression_pool'])
                
                if 'sub_emotions' in emotion_data:
                    # New structure - progressions in sub-emotions
                    for sub_emotion_name, sub_emotion_data in emotion_data['sub_emotions'].items():
                        if 'progression_pool' in sub_emotion_data:
                            progressions_to_process.extend(sub_emotion_data['progression_pool'])
                
                # Process all collected progressions
                for prog in progressions_to_process:
                    chord_sequence = prog['chords']
                    
                    # Convert to indices
                    chord_indices = []
                    for chord in chord_sequence:
                        if chord in self.chord_to_idx:
                            chord_indices.append(self.chord_to_idx[chord])
                        else:
                            chord_indices.append(0)  # Unknown chord token
                    
                    # Create position indices
                    positions = list(range(len(chord_indices)))
                    
                    # Pad or truncate to context window
                    if len(chord_indices) < self.neural_analyzer.context_window:
                        chord_indices.extend([0] * (self.neural_analyzer.context_window - len(chord_indices)))
                        positions.extend([0] * (self.neural_analyzer.context_window - len(positions)))
                    else:
                        chord_indices = chord_indices[:self.neural_analyzer.context_window]
                        positions = positions[:self.neural_analyzer.context_window]
                    
                    self.training_data.append({
                        'chord_indices': chord_indices,
                        'positions': positions,
                        'progression_emotion': emotion_vector,
                        'progression_id': prog['progression_id']
                    })
                    
        except FileNotFoundError:
            print("Warning: Could not load progression database for training data")
        except Exception as e:
            print(f"Warning: Error processing progression database: {e}")
    
    def _emotion_name_to_vector(self, emotion_name: str) -> List[float]:
        """Convert emotion name to 23-dimensional vector"""
        vector = [0.0] * 23  # Updated to 23 emotions including Transcendence
        if emotion_name in self.emotion_labels:
            idx = self.emotion_labels.index(emotion_name)
            vector[idx] = 1.0
        return vector
    
    def analyze_progression_context(self, chord_progression: List[str]) -> ProgressionAnalysis:
        """
        Analyze a chord progression using contextual weighting
        
        Args:
            chord_progression: List of roman numeral chords
            
        Returns:
            ProgressionAnalysis with contextual chord information
        """
        # Convert progression to neural network input
        chord_indices = []
        for chord in chord_progression:
            if chord in self.chord_to_idx:
                chord_indices.append(self.chord_to_idx[chord])
            else:
                chord_indices.append(0)  # Unknown chord
        
        positions = list(range(len(chord_indices)))
        
        # Pad to context window
        original_length = len(chord_indices)
        if len(chord_indices) < self.neural_analyzer.context_window:
            chord_indices.extend([0] * (self.neural_analyzer.context_window - len(chord_indices)))
            positions.extend([0] * (self.neural_analyzer.context_window - len(positions)))
        else:
            chord_indices = chord_indices[:self.neural_analyzer.context_window]
            positions = positions[:self.neural_analyzer.context_window]
        
        # Run through neural network
        with torch.no_grad():
            chord_tensor = torch.tensor([chord_indices])
            position_tensor = torch.tensor([positions])
            
            predictions = self.neural_analyzer(chord_tensor, position_tensor)
        
        # Extract predictions
        progression_emotions = predictions['progression_emotions'][0].numpy()
        contextual_emotions = predictions['contextual_chord_emotions'][0].numpy()
        attention_weights = predictions['attention_weights'][0].numpy()
        novelty_score = predictions['novelty_scores'][0].item()
        tension_scores = predictions['tension_scores'][0].numpy()
        consonant_dissonant_scores = predictions['consonant_dissonant_scores'][0].numpy()
        
        # Analyze each chord in context
        contextual_analyses = []
        for i in range(min(original_length, len(chord_progression))):
            chord = chord_progression[i]
            
            # Get base emotions from individual model
            base_emotions, cd_value_from_individual = self._get_individual_chord_data(chord)
            
            # Get contextual emotions from neural network
            context_emotions = {
                emotion: float(contextual_emotions[i][j])
                for j, emotion in enumerate(self.emotion_labels)
            }
            
            # Determine functional role (simplified)
            functional_role = self._determine_functional_role(chord, i, chord_progression)
            
            # Calculate contextual weight from attention
            contextual_weight = float(attention_weights[i].sum()) if i < len(attention_weights) else 0.5
            
            # Harmonic tension
            harmonic_tension = float(tension_scores[i]) if i < len(tension_scores) else 0.5
            
            # Consonant/Dissonant value: Use individual chord data if available, otherwise neural prediction
            if cd_value_from_individual is not None:
                consonant_dissonant_value = cd_value_from_individual
            else:
                consonant_dissonant_value = float(consonant_dissonant_scores[i]) if i < len(consonant_dissonant_scores) else 0.5
            
            analysis = ContextualChordAnalysis(
                chord_symbol=self._roman_to_symbol(chord),
                roman_numeral=chord,
                position_in_progression=i,
                base_emotions=base_emotions,
                contextual_emotions=context_emotions,
                functional_role=functional_role,
                harmonic_tension=harmonic_tension,
                contextual_weight=contextual_weight,
                consonant_dissonant_value=consonant_dissonant_value,
                consonant_dissonant_context=""  # Placeholder, actual context would be determined
            )
            contextual_analyses.append(analysis)
        
        # Calculate average CD and flow description
        cd_values = [analysis.consonant_dissonant_value for analysis in contextual_analyses]
        average_cd = sum(cd_values) / len(cd_values) if cd_values else 0.5
        
        # Generate CD flow description
        cd_flow_description = self._generate_cd_flow_description(cd_values)
        
        # Store CD trajectory
        consonant_dissonant_trajectory = cd_values
        
        # Update CD context descriptions
        for i, analysis in enumerate(contextual_analyses):
            analysis.consonant_dissonant_context = self._generate_cd_context_description(
                analysis.consonant_dissonant_value, i, cd_values
            )
        
        # Build progression emotion dictionary
        overall_emotions = {
            emotion: float(progression_emotions[i]) 
            for i, emotion in enumerate(self.emotion_labels)
        }
        
        # Calculate harmonic flow
        harmonic_flow = [float(tension_scores[i]) if i < len(tension_scores) else 0.5 
                        for i in range(original_length)]
        
        return ProgressionAnalysis(
            chords=chord_progression[:original_length],
            progression_id=f"progression_{int(time.time())}",
            overall_emotion_weights=overall_emotions,
            contextual_chord_analyses=contextual_analyses,
            harmonic_flow=harmonic_flow,
            consonant_dissonant_trajectory=consonant_dissonant_trajectory,
            novel_pattern_score=novelty_score,
            generation_confidence=1.0 - novelty_score,  # More familiar = more confident
            average_consonant_dissonant=average_cd,
            cd_flow_description=cd_flow_description
        )
    
    def _get_individual_chord_data(self, roman_numeral: str) -> Tuple[Dict[str, float], Optional[float]]:
        """
        Get base emotions and consonant/dissonant value from individual chord model
        
        Returns:
            Tuple of (emotion_weights, cd_value)
        """
        try:
            # Search through individual chord database for matching roman numeral
            for chord_obj in self.individual_model.database.chord_emotion_map:
                if chord_obj.roman_numeral == roman_numeral:
                    # Get emotion weights
                    base_emotions = chord_obj.emotion_weights.copy()
                    
                    # Get consonant/dissonant value if available
                    cd_value = None
                    if chord_obj.consonant_dissonant_profile:
                        cd_profile = chord_obj.consonant_dissonant_profile
                        cd_value = cd_profile.get("base_value", 0.4)
                        
                        # Apply context modifiers if available
                        context_modifiers = cd_profile.get("context_modifiers", {})
                        # Use Classical as default context for progression analysis
                        context_modifier = context_modifiers.get("Classical", 1.0)
                        cd_value = cd_value * context_modifier
                    
                    return base_emotions, cd_value
            
            # If no exact match found, try to find a similar chord
            # Look for chords with similar function
            fallback_chords = {
                'I': ['I', 'Imaj7', 'I6'],
                'ii': ['ii', 'ii7', 'iim7'],
                'iii': ['iii', 'iii7', 'iiim7'],
                'IV': ['IV', 'IVmaj7', 'IV6'],
                'V': ['V', 'V7', 'Vmaj7'],
                'vi': ['vi', 'vi7', 'vim7'],
                'vii': ['vii', 'vii7', 'viim7b5'],
                'i': ['i', 'im7', 'i6'],
                'iv': ['iv', 'ivm7', 'iv6'],
                'v': ['v', 'vm7', 'v6']
            }
            
            for base_chord, variants in fallback_chords.items():
                if roman_numeral in variants:
                    # Look for the base chord
                    for chord_obj in self.individual_model.database.chord_emotion_map:
                        if chord_obj.roman_numeral == base_chord:
                            base_emotions = chord_obj.emotion_weights.copy()
                            cd_value = None
                            if chord_obj.consonant_dissonant_profile:
                                cd_value = chord_obj.consonant_dissonant_profile.get("base_value", 0.4)
                            return base_emotions, cd_value
            
            # Final fallback - return neutral emotions
            fallback_emotions = {emotion: 0.1 for emotion in self.emotion_labels}
            fallback_emotions["Trust"] = 0.5  # Slightly more trust as default
            return fallback_emotions, 0.4  # Moderate consonance
            
        except Exception as e:
            print(f"Warning: Error retrieving individual chord data for {roman_numeral}: {e}")
            fallback_emotions = {emotion: 0.1 for emotion in self.emotion_labels}
            fallback_emotions["Joy"] = 0.5  # Default to mild joy
            return fallback_emotions, 0.4

    def _determine_functional_role(self, chord: str, position: int, progression: List[str]) -> str:
        """Determine the functional role of a chord in the progression"""
        # Simplified functional analysis
        if chord in ['I', 'i']:
            return "tonic"
        elif chord in ['V', 'V7', 'vii°', 'vii7']:
            return "dominant"
        elif chord in ['IV', 'iv', 'ii', 'ii7']:
            return "subdominant"
        elif chord in ['vi', 'iii']:
            return "tonic_substitute"
        else:
            return "transitional"
    
    def _roman_to_symbol(self, roman: str, key: str = "C") -> str:
        """Convert roman numeral to chord symbol (simplified)"""
        # This is a simplified conversion - in practice you'd want more sophisticated logic
        conversions = {
            'I': 'C', 'ii': 'Dm', 'iii': 'Em', 'IV': 'F', 'V': 'G', 'vi': 'Am', 'vii°': 'Bdim',
            'i': 'Cm', 'ii°': 'Ddim', 'bIII': 'Eb', 'iv': 'Fm', 'v': 'Gm', 'bVI': 'Ab', 'bVII': 'Bb'
        }
        return conversions.get(roman, roman)
    
    def generate_novel_progression(self, emotion_prompt: str, length: int = 4) -> ProgressionAnalysis:
        """
        Generate a novel chord progression using learned patterns
        
        Args:
            emotion_prompt: Natural language emotion description
            length: Desired progression length
            
        Returns:
            ProgressionAnalysis of the generated progression
        """
        # First, try existing models to get baseline
        try:
            # Get progression suggestions from existing model
            existing_progs = self.progression_model.generate_from_prompt(emotion_prompt, num_progressions=3)
            base_progression = existing_progs[0]['chords'] if existing_progs else ['I', 'vi', 'IV', 'V']
        except:
            base_progression = ['I', 'vi', 'IV', 'V']  # Fallback
        
        # Get individual chord suggestions
        try:
            individual_suggestions = self.individual_model.generate_chord_from_prompt(emotion_prompt, num_options=8)
            candidate_chords = [chord['roman_numeral'] for chord in individual_suggestions]
        except:
            candidate_chords = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']  # Fallback
        
        # Use neural network to score different combinations
        best_progression = None
        best_score = -1
        
        # Try variations on the base progression
        for i in range(10):  # Try 10 variations
            # Create a variation by substituting some chords
            variation = base_progression.copy()
            
            # Substitute 1-2 chords with individual suggestions
            num_substitutions = min(2, len(variation))
            positions_to_substitute = np.random.choice(len(variation), num_substitutions, replace=False)
            
            for pos in positions_to_substitute:
                new_chord = np.random.choice(candidate_chords)
                variation[pos] = new_chord
            
            # Truncate or extend to desired length
            if len(variation) > length:
                variation = variation[:length]
            elif len(variation) < length:
                # Extend with musically logical chords
                while len(variation) < length:
                    last_chord = variation[-1]
                    # Simple logic for next chord
                    if last_chord == 'V':
                        variation.append('I')
                    elif last_chord == 'I':
                        variation.append(np.random.choice(['vi', 'IV', 'ii']))
                    else:
                        variation.append(np.random.choice(candidate_chords))
            
            # Score this variation
            analysis = self.analyze_progression_context(variation)
            
            # Simple scoring: prefer lower novelty (more familiar patterns) 
            # but with good emotional match
            score = analysis.generation_confidence * 0.7 + (1.0 - analysis.novel_pattern_score) * 0.3
            
            if score > best_score:
                best_score = score
                best_progression = variation
        
        # Analyze the best progression found
        if best_progression:
            return self.analyze_progression_context(best_progression)
        else:
            # Fallback to base progression
            return self.analyze_progression_context(base_progression)
    
    def train_neural_analyzer(self, epochs: int = 50, learning_rate: float = 0.001):
        """
        Train the neural progression analyzer on the prepared data
        """
        if not self.training_data:
            print("No training data available. Cannot train neural analyzer.")
            return
        
        optimizer = torch.optim.Adam(self.neural_analyzer.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print(f"Training neural analyzer on {len(self.training_data)} samples...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle training data
            np.random.shuffle(self.training_data)
            
            for batch in self.training_data:
                optimizer.zero_grad()
                
                # Prepare input tensors
                chord_indices = torch.tensor([batch['chord_indices']])
                positions = torch.tensor([batch['positions']])
                target_emotion = torch.tensor([batch['progression_emotion']])
                
                # Forward pass
                predictions = self.neural_analyzer(chord_indices, positions)
                
                # Calculate loss
                loss = criterion(predictions['progression_emotions'], target_emotion)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.training_data)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")
    
    def save_model(self, filepath: str):
        """Save the trained neural analyzer"""
        torch.save({
            'model_state_dict': self.neural_analyzer.state_dict(),
            'chord_vocab': self.chord_vocab,
            'chord_to_idx': self.chord_to_idx,
            'emotion_labels': self.emotion_labels
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained neural analyzer"""
        checkpoint = torch.load(filepath)
        self.neural_analyzer.load_state_dict(checkpoint['model_state_dict'])
        self.chord_vocab = checkpoint['chord_vocab']
        self.chord_to_idx = checkpoint['chord_to_idx']
        self.emotion_labels = checkpoint['emotion_labels']
        print(f"Model loaded from {filepath}")

    def _generate_cd_flow_description(self, cd_values: List[float]) -> str:
        """Generate a description of how consonant/dissonant values change through the progression"""
        if not cd_values:
            return "No CD data available"
        
        if len(cd_values) == 1:
            if cd_values[0] < 0.3:
                return "Consonant throughout"
            elif cd_values[0] > 0.7:
                return "Dissonant throughout"
            else:
                return "Moderately consonant"
        
        # Analyze trend
        start_cd = cd_values[0]
        end_cd = cd_values[-1]
        max_cd = max(cd_values)
        min_cd = min(cd_values)
        
        # Calculate trend
        if end_cd > start_cd + 0.2:
            trend = "increasing tension"
        elif end_cd < start_cd - 0.2:
            trend = "releasing tension"
        else:
            trend = "stable tension"
        
        # Calculate overall character
        avg_cd = sum(cd_values) / len(cd_values)
        if avg_cd < 0.3:
            character = "predominantly consonant"
        elif avg_cd > 0.7:
            character = "predominantly dissonant"
        else:
            character = "moderately dissonant"
        
        return f"{character} with {trend} (range: {min_cd:.2f}-{max_cd:.2f})"
    
    def _generate_cd_context_description(self, cd_value: float, position: int, cd_trajectory: List[float]) -> str:
        """Generate a description of this chord's CD role in the progression context"""
        if cd_value < 0.3:
            base_desc = "consonant"
        elif cd_value > 0.7:
            base_desc = "dissonant"
        else:
            base_desc = "moderately dissonant"
        
        # Add positional context
        if position == 0:
            pos_desc = "opening"
        elif position == len(cd_trajectory) - 1:
            pos_desc = "closing"
        else:
            pos_desc = f"middle (position {position + 1})"
        
        # Add relative context
        if len(cd_trajectory) > 1:
            avg_cd = sum(cd_trajectory) / len(cd_trajectory)
            if cd_value > avg_cd + 0.2:
                relative_desc = "peak tension"
            elif cd_value < avg_cd - 0.2:
                relative_desc = "tension release"
            else:
                relative_desc = "typical tension"
        else:
            relative_desc = "standalone"
        
        return f"{base_desc} {pos_desc} chord providing {relative_desc}"

def demo_contextual_analysis():
    """Demo the contextual progression analyzer"""
    print("=" * 70)
    print("    NEURAL PROGRESSION ANALYZER DEMO")
    print("    Contextual Chord-Emotion Weighting")
    print("=" * 70)
    
    # Initialize the integrator
    integrator = ContextualProgressionIntegrator()
    
    # Test progressions
    test_progressions = [
        ['I', 'vi', 'IV', 'V'],
        ['i', 'iv', 'V', 'i'],
        ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'],
        ['ii', 'V', 'I', 'vi']
    ]
    
    test_emotions = [
        "happy and uplifting",
        "melancholy and introspective", 
        "complex emotional journey",
        "sophisticated jazz feeling"
    ]
    
    for i, (progression, emotion) in enumerate(zip(test_progressions, test_emotions), 1):
        print(f"\n🎵 Test {i}: {emotion}")
        print(f"Progression: {' - '.join(progression)}")
        print("-" * 50)
        
        # Analyze the progression
        analysis = integrator.analyze_progression_context(progression)
        
        # Show overall progression emotions
        top_emotions = sorted(analysis.overall_emotion_weights.items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        print(f"Overall emotions: {', '.join([f'{e}({w:.3f})' for e, w in top_emotions])}")
        print(f"Novel pattern score: {analysis.novel_pattern_score:.3f}")
        print(f"Generation confidence: {analysis.generation_confidence:.3f}")
        
        # Show contextual analysis for each chord
        print("\nContextual Chord Analysis:")
        for chord_analysis in analysis.contextual_chord_analyses:
            print(f"  {chord_analysis.position_in_progression + 1}. {chord_analysis.roman_numeral} ({chord_analysis.chord_symbol})")
            print(f"     Role: {chord_analysis.functional_role}")
            print(f"     Tension: {chord_analysis.harmonic_tension:.3f}")
            print(f"     Weight: {chord_analysis.contextual_weight:.3f}")
            
            # Show top contextual emotions
            top_context_emotions = sorted(chord_analysis.contextual_emotions.items(),
                                        key=lambda x: x[1], reverse=True)[:2]
            print(f"     Context emotions: {', '.join([f'{e}({w:.3f})' for e, w in top_context_emotions])}")
        
        # Show harmonic flow
        print(f"\nHarmonic flow: {[f'{t:.2f}' for t in analysis.harmonic_flow]}")
    
    # Test novel progression generation
    print(f"\n🚀 NOVEL PROGRESSION GENERATION")
    print("-" * 50)
    
    novel_emotion = "bittersweet nostalgia with hope"
    print(f"Generating novel progression for: '{novel_emotion}'")
    
    novel_analysis = integrator.generate_novel_progression(novel_emotion, length=6)
    print(f"Generated: {' - '.join(novel_analysis.chords)}")
    
    top_emotions = sorted(novel_analysis.overall_emotion_weights.items(),
                         key=lambda x: x[1], reverse=True)[:3]
    print(f"Emotions: {', '.join([f'{e}({w:.3f})' for e, w in top_emotions])}")
    print(f"Novelty: {novel_analysis.novel_pattern_score:.3f}")
    print(f"Confidence: {novel_analysis.generation_confidence:.3f}")

if __name__ == "__main__":
    demo_contextual_analysis()
