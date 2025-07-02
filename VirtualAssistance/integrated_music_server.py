#!/usr/bin/env python3
"""
Integrated Music Theory Server
Complete system for contextual chord-emotion translation

This is the main server interface that combines:
1. Individual chord model - emotion to single chord mapping
2. Chord progression model - emotion to progression mapping  
3. Neural progression analyzer - contextual integration and novel generation

Usage:
    from integrated_music_server import IntegratedMusicServer
    
    server = IntegratedMusicServer()
    
    # Get individual chords
    chords = server.get_individual_chords("sad but beautiful", num_options=3)
    
    # Get chord progressions  
    progressions = server.get_progressions("happy and energetic", num_progressions=2)
    
    # Analyze progression context
    analysis = server.analyze_progression_context(['I', 'vi', 'IV', 'V'])
    
    # Generate novel progression
    novel = server.generate_novel_progression("mysterious tension", length=6)
"""

import json
import torch
from typing import Dict, List, Optional, Union
from datetime import datetime

# Import our models
from individual_chord_model import IndividualChordModel
from chord_progression_model import ChordProgressionModel  
from neural_progression_analyzer import ContextualProgressionIntegrator, ProgressionAnalysis

class IntegratedMusicServer:
    """
    Main server class that provides a unified interface to all music generation models
    """
    
    def __init__(self, load_trained_neural_model: bool = True):
        """Initialize the integrated music server"""
        print("🎵 Initializing Integrated Music Server...")
        
        # Initialize individual models
        print("   Loading individual chord model...")
        self.individual_model = IndividualChordModel()
        
        print("   Loading chord progression model...")
        self.progression_model = ChordProgressionModel()
        
        print("   Loading neural progression analyzer...")
        self.neural_integrator = ContextualProgressionIntegrator()
        
        # Load trained neural model if available
        if load_trained_neural_model:
            try:
                self.neural_integrator.load_model('trained_neural_analyzer.pth')
                print("   ✅ Loaded trained neural analyzer")
                self.neural_trained = True
            except Exception as e:
                print(f"   ⚠️  Could not load trained neural model: {e}")
                print("   Using untrained neural analyzer (will show random patterns)")
                self.neural_trained = False
        else:
            self.neural_trained = False
        
        print("🎵 Server initialization complete!")
        print()
    
    def get_individual_chords(self, 
                            emotion_prompt: str,
                            num_options: int = 3,
                            mode_preference: Optional[str] = None,
                            style_preference: Optional[str] = None,
                            key: str = "C") -> List[Dict]:
        """
        Get individual chord suggestions for an emotion
        
        Args:
            emotion_prompt: Natural language emotion description
            num_options: Number of chord options to return
            mode_preference: Preferred musical mode (e.g., "Ionian", "Aeolian")
            style_preference: Preferred style (e.g., "Jazz", "Classical")
            key: Key signature (default "C")
            
        Returns:
            List of chord dictionaries with symbols, emotions, and metadata
        """
        try:
            results = self.individual_model.generate_chord_from_prompt(
                emotion_prompt,
                num_options=num_options,
                mode_preference=mode_preference,
                style_preference=style_preference,
                key=key
            )
            
            # Add server metadata
            for result in results:
                result['source'] = 'individual_model'
                result['timestamp'] = datetime.now().isoformat()
                result['query'] = emotion_prompt
                
            return results
            
        except Exception as e:
            return [{'error': f"Individual chord generation failed: {e}"}]
    
    def get_progressions(self,
                        emotion_prompt: str,
                        num_progressions: int = 2,
                        genre_preference: Optional[str] = None) -> List[Dict]:
        """
        Get chord progression suggestions for an emotion
        
        Args:
            emotion_prompt: Natural language emotion description
            num_progressions: Number of progressions to return
            genre_preference: Preferred genre/style
            
        Returns:
            List of progression dictionaries with chords and metadata
        """
        try:
            results = self.progression_model.generate_from_prompt(
                emotion_prompt,
                num_progressions=num_progressions
            )
            
            # Add server metadata
            for result in results:
                result['source'] = 'progression_model'
                result['timestamp'] = datetime.now().isoformat()
                result['query'] = emotion_prompt
                
            return results
            
        except Exception as e:
            return [{'error': f"Progression generation failed: {e}"}]
    
    def analyze_progression_context(self, chord_progression: List[str]) -> Dict:
        """
        Analyze a chord progression using contextual weighting
        
        Args:
            chord_progression: List of roman numeral chords
            
        Returns:
            Detailed analysis dictionary with contextual information
        """
        try:
            analysis = self.neural_integrator.analyze_progression_context(chord_progression)
            
            # Convert to dictionary for JSON serialization
            return self._analysis_to_dict(analysis)
            
        except Exception as e:
            return {'error': f"Progression analysis failed: {e}"}
    
    def generate_novel_progression(self,
                                 emotion_prompt: str,
                                 length: int = 4,
                                 creativity: float = 0.5) -> Dict:
        """
        Generate a novel chord progression using learned patterns
        
        Args:
            emotion_prompt: Natural language emotion description
            length: Desired progression length
            creativity: How novel vs familiar (0.0 = very familiar, 1.0 = very novel)
            
        Returns:
            Analysis dictionary of the generated progression
        """
        try:
            analysis = self.neural_integrator.generate_novel_progression(
                emotion_prompt, 
                length=length
            )
            
            # Convert to dictionary and add metadata
            result = self._analysis_to_dict(analysis)
            result['source'] = 'neural_generator'
            result['timestamp'] = datetime.now().isoformat()
            result['query'] = emotion_prompt
            result['requested_length'] = length
            result['creativity_setting'] = creativity
            result['neural_trained'] = self.neural_trained
            
            return result
            
        except Exception as e:
            return {'error': f"Novel progression generation failed: {e}"}
    
    def get_contextual_comparison(self, emotion_prompt: str) -> Dict:
        """
        Compare how the same emotion is interpreted across all models
        
        Args:
            emotion_prompt: Natural language emotion description
            
        Returns:
            Dictionary comparing outputs from all models
        """
        result = {
            'query': emotion_prompt,
            'timestamp': datetime.now().isoformat(),
            'individual_chords': [],
            'progressions': [],
            'novel_progression': {},
            'contextual_analysis': {}
        }
        
        # Get individual chords
        try:
            result['individual_chords'] = self.get_individual_chords(emotion_prompt, num_options=3)
        except Exception as e:
            result['individual_chords'] = [{'error': str(e)}]
        
        # Get progressions
        try:
            result['progressions'] = self.get_progressions(emotion_prompt, num_progressions=2)
        except Exception as e:
            result['progressions'] = [{'error': str(e)}]
        
        # Generate novel progression
        try:
            result['novel_progression'] = self.generate_novel_progression(emotion_prompt, length=5)
        except Exception as e:
            result['novel_progression'] = {'error': str(e)}
        
        # Analyze the first progression if available
        if result['progressions'] and 'chords' in result['progressions'][0]:
            try:
                progression_chords = result['progressions'][0]['chords']
                result['contextual_analysis'] = self.analyze_progression_context(progression_chords)
            except Exception as e:
                result['contextual_analysis'] = {'error': str(e)}
        
        return result
    
    def _analysis_to_dict(self, analysis: ProgressionAnalysis) -> Dict:
        """Convert ProgressionAnalysis to dictionary for JSON serialization"""
        return {
            'chords': analysis.chords,
            'progression_id': analysis.progression_id,
            'overall_emotion_weights': analysis.overall_emotion_weights,
            'novel_pattern_score': analysis.novel_pattern_score,
            'generation_confidence': analysis.generation_confidence,
            'harmonic_flow': analysis.harmonic_flow,
            'contextual_chord_analyses': [
                {
                    'chord_symbol': chord.chord_symbol,
                    'roman_numeral': chord.roman_numeral,
                    'position_in_progression': chord.position_in_progression,
                    'base_emotions': chord.base_emotions,
                    'contextual_emotions': chord.contextual_emotions,
                    'functional_role': chord.functional_role,
                    'harmonic_tension': chord.harmonic_tension,
                    'contextual_weight': chord.contextual_weight
                }
                for chord in analysis.contextual_chord_analyses
            ]
        }
    
    def get_system_status(self) -> Dict:
        """Get status information about all loaded models"""
        return {
            'timestamp': datetime.now().isoformat(),
            'individual_model_loaded': hasattr(self, 'individual_model'),
            'progression_model_loaded': hasattr(self, 'progression_model'),
            'neural_integrator_loaded': hasattr(self, 'neural_integrator'),
            'neural_model_trained': self.neural_trained,
            'available_contexts': self.individual_model.get_available_contexts() if hasattr(self, 'individual_model') else {},
            'chord_vocabulary_size': len(self.neural_integrator.chord_vocab) if hasattr(self, 'neural_integrator') else 0,
            'training_data_size': len(self.neural_integrator.training_data) if hasattr(self, 'neural_integrator') else 0
        }

def demo_server():
    """Demonstrate the integrated server functionality"""
    print("=" * 70)
    print("    INTEGRATED MUSIC SERVER DEMO")
    print("=" * 70)
    
    # Initialize server
    server = IntegratedMusicServer()
    
    # Show system status
    print("📊 SYSTEM STATUS:")
    status = server.get_system_status()
    for key, value in status.items():
        if key != 'timestamp':
            print(f"   {key}: {value}")
    print()
    
    # Test emotional prompts
    test_prompts = [
        "melancholy but hopeful",
        "energetic celebration",
        "mysterious tension with jazz sophistication"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"🎵 TEST {i}: '{prompt}'")
        print("-" * 50)
        
        # Get contextual comparison
        comparison = server.get_contextual_comparison(prompt)
        
        # Show individual chords
        print("Individual chords:")
        for j, chord in enumerate(comparison['individual_chords'][:2], 1):
            if 'error' not in chord:
                print(f"  {j}. {chord['chord_symbol']} ({chord['roman_numeral']}) - {chord['mode_context']}/{chord['style_context']}")
                print(f"     Score: {chord['emotional_score']:.3f}")
        
        # Show progressions
        print("Progressions:")
        for j, prog in enumerate(comparison['progressions'], 1):
            if 'error' not in prog:
                print(f"  {j}. {' → '.join(prog['chords'])} - {prog['primary_mode']}")
        
        # Show novel progression
        if 'error' not in comparison['novel_progression']:
            novel = comparison['novel_progression']
            print(f"Novel progression: {' → '.join(novel['chords'])}")
            print(f"  Novelty: {novel['novel_pattern_score']:.3f}, Confidence: {novel['generation_confidence']:.3f}")
        
        # Show contextual analysis summary
        if 'error' not in comparison['contextual_analysis']:
            analysis = comparison['contextual_analysis']
            top_emotions = sorted(analysis['overall_emotion_weights'].items(),
                                key=lambda x: x[1], reverse=True)[:2]
            print(f"Contextual emotions: {', '.join([f'{e}({w:.2f})' for e, w in top_emotions])}")
        
        print()
    
    print("✨ Server demo complete! The integrated system provides:")
    print("   • Individual chord generation with emotional context")
    print("   • Chord progression generation with modal awareness")
    print("   • Contextual analysis of progression-chord relationships")
    print("   • Novel progression generation using learned patterns")
    print("   • Unified API for all music theory operations")

if __name__ == "__main__":
    demo_server()
