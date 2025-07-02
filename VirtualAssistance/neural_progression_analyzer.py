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

@dataclass
class ProgressionAnalysis:
    """Complete analysis of a chord progression"""
    chords: List[str]
    progression_id: str
    overall_emotion_weights: Dict[str, float]
    contextual_chord_analyses: List[ContextualChordAnalysis]
    harmonic_flow: List[float]  # Tension curve throughout progression
    novel_pattern_score: float  # How different this is from known patterns
    generation_confidence: float  # How confident the model is in this progression

class NeuralProgressionAnalyzer(nn.Module):
    """
    Neural network that learns to:
    1. Predict progression-level emotions from chord sequences
    2. Adjust individual chord emotions based on context
    3. Generate novel progressions through learned patterns
    """
    
    def __init__(self, 
                 chord_vocab_size: int = 100,  # Max number of unique chords
                 emotion_dim: int = 12,
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
        
        return {
            'progression_emotions': progression_emotion,
            'contextual_chord_emotions': contextual_emotions,
            'attention_weights': attention_weights,
            'novelty_scores': novelty_score,
            'tension_scores': tension_scores
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
        
        # Chord vocabulary for neural network
        self.chord_vocab = self._build_chord_vocabulary()
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(self.chord_vocab)}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        
        # Emotion labels
        self.emotion_labels = [
            "Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise",
            "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe"
        ]
        
        # Training data for neural network
        self.training_data = []
        self._prepare_training_data()
        
    def _build_chord_vocabulary(self) -> List[str]:
        """Build vocabulary of all possible chords from both models"""
        vocab = set()
        
        # Add chords from individual chord database
        for chord in self.individual_model.database.chord_emotion_map:
            vocab.add(chord.roman_numeral)
            
        # Add chords from progression database
        try:
            with open('emotion_progression_database.json', 'r') as f:
                prog_data = json.load(f)
            
            for emotion_name, emotion_data in prog_data['emotions'].items():
                for prog in emotion_data['progression_pool']:
                    for chord in prog['chords']:
                        vocab.add(chord)
        except FileNotFoundError:
            print("Warning: Could not load progression database for vocabulary building")
        
        # Add common extended chords not in database
        common_extensions = ['I7', 'ii7', 'iii7', 'IV7', 'V7', 'vi7', 'vii7',
                            'Imaj7', 'iimaj7', 'IVmaj7', 'Vmaj7', 'vimaj7',
                            'i7', 'iv7', 'v7', 'bVII7', 'bVI7', 'bIII7']
        vocab.update(common_extensions)
        
        return sorted(list(vocab))
    
    def _prepare_training_data(self):
        """Prepare training data from existing databases"""
        # Load progression database
        try:
            with open('emotion_progression_database.json', 'r') as f:
                prog_data = json.load(f)
            
            for emotion_name, emotion_data in prog_data['emotions'].items():
                emotion_vector = self._emotion_name_to_vector(emotion_name)
                
                for prog in emotion_data['progression_pool']:
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
    
    def _emotion_name_to_vector(self, emotion_name: str) -> List[float]:
        """Convert emotion name to 12-dimensional vector"""
        vector = [0.0] * 12
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
        
        # Build progression emotion dictionary
        overall_emotions = {
            emotion: float(progression_emotions[i]) 
            for i, emotion in enumerate(self.emotion_labels)
        }
        
        # Analyze each chord in context
        contextual_analyses = []
        for i in range(min(original_length, len(chord_progression))):
            chord = chord_progression[i]
            
            # Get base emotions from individual model
            try:
                individual_results = self.individual_model.generate_chord_from_prompt(
                    "neutral", num_options=1  # We just want the emotion weights
                )
                # Find matching chord in individual results
                base_emotions = {"Joy": 0.1}  # Default fallback
                for chord_obj in self.individual_model.database.chord_emotion_map:
                    if chord_obj.roman_numeral == chord:
                        base_emotions = chord_obj.emotion_weights
                        break
            except:
                base_emotions = {"Joy": 0.1}  # Fallback
            
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
            
            analysis = ContextualChordAnalysis(
                chord_symbol=self._roman_to_symbol(chord),
                roman_numeral=chord,
                position_in_progression=i,
                base_emotions=base_emotions,
                contextual_emotions=context_emotions,
                functional_role=functional_role,
                harmonic_tension=harmonic_tension,
                contextual_weight=contextual_weight
            )
            contextual_analyses.append(analysis)
        
        # Build harmonic flow
        harmonic_flow = [float(t) for t in tension_scores[:original_length]]
        
        return ProgressionAnalysis(
            chords=chord_progression,
            progression_id=f"analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_emotion_weights=overall_emotions,
            contextual_chord_analyses=contextual_analyses,
            harmonic_flow=harmonic_flow,
            novel_pattern_score=novelty_score,
            generation_confidence=1.0 - novelty_score  # More familiar = more confident
        )
    
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
