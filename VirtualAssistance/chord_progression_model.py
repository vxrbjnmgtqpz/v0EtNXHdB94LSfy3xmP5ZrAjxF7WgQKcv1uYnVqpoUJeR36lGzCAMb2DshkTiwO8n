"""
Chord Progression Generation Model from Natural Language Prompts
Based on 12-Core Emotion → Musical Mode → Genre-Weighted Progression Mapping

This PyTorch model generates chord progressions from natural language input by:
1. Parsing emotional content from text prompts
2. Mapping emotions to musical modes with weighted blending
3. Selecting appropriate chord progressions based on genre preferences
4. Using JSON tokens for structured music representation

Architecture:
- Emotion Parser: BERT-based text encoder → emotion weight vectors
- Mode Mapper: Weighted emotion blend → modal fingerprint
- Progression Generator: Transformer decoder → chord sequence tokens
- Genre Weighting: Probabilistic selection based on style preferences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChordProgression:
    """Data structure for chord progression with metadata"""
    chords: List[str]
    emotion: str
    mode: str
    genres: Dict[str, float]
    progression_id: str


class EmotionMusicDatabase:
    """
    JSON-based database containing the complete emotion → mode → progression mapping
    This serves as the training data and lookup system for the model
    """
    
    def __init__(self, database_path: Optional[str] = None):
        self.emotions_to_modes = {
            "Joy": "Ionian",
            "Sadness": "Aeolian", 
            "Fear": "Phrygian",
            "Anger": "Phrygian Dominant",
            "Disgust": "Locrian",
            "Surprise": "Lydian",
            "Trust": "Dorian",
            "Anticipation": "Melodic Minor",
            "Shame": "Harmonic Minor",
            "Love": "Mixolydian",
            "Envy": "Hungarian Minor",
            "Aesthetic Awe": "Lydian Augmented"
        }
        
        self.chord_progressions = self._load_progression_database()
        self.emotion_keywords = self._build_emotion_keywords()
        
    def _load_progression_database(self) -> Dict[str, List[ChordProgression]]:
        """Load all 12 emotion progression pools from the JSON database"""
        try:
            with open('emotion_progression_database.json', 'r') as f:
                data = json.load(f)
            
            progressions = {}
            
            for emotion_name, emotion_data in data['emotions'].items():
                emotion_progressions = []
                for prog_data in emotion_data['progression_pool']:
                    progression = ChordProgression(
                        chords=prog_data['chords'],
                        emotion=emotion_name,
                        mode=emotion_data['mode'],
                        genres=prog_data['genres'],
                        progression_id=prog_data['progression_id']
                    )
                    emotion_progressions.append(progression)
                
                progressions[emotion_name] = emotion_progressions
            
            return progressions
            
        except FileNotFoundError:
            print("Warning: emotion_progression_database.json not found. Using sample data.")
            return self._create_sample_progressions()
    
    def _create_sample_progressions(self) -> Dict[str, List[ChordProgression]]:
        """Create sample progressions if database file is not found"""
        progressions = {}
        
        # Sample progressions for each emotion
        sample_data = {
            "Joy": [["I", "IV", "V", "I"], ["I", "vi", "IV", "V"], ["I", "V", "vi", "IV"]],
            "Sadness": [["i", "iv", "i", "v"], ["i", "VI", "III", "VII"], ["i", "VII", "VI", "VII"]],
            "Fear": [["i", "♭II", "i", "v"], ["♭II", "i", "iv", "i"], ["i", "♭II", "♭VII", "i"]],
            "Anger": [["I", "♭II", "iv", "I"], ["I", "♭II", "♭VI", "I"], ["I", "V", "♭II", "I"]],
            "Disgust": [["i°", "♭II", "♭v", "i°"], ["i°", "iv", "♭II", "♭v"], ["♭II", "♭v", "♭VII", "i°"]],
            "Surprise": [["I", "II", "V", "I"], ["I", "II", "iii", "IV"], ["I", "V", "II", "I"]],
            "Trust": [["i", "ii", "IV", "i"], ["i", "IV", "v", "i"], ["i", "ii", "v", "i"]],
            "Anticipation": [["i", "ii", "V", "i"], ["i", "IV", "V", "i"], ["i", "ii", "vi", "V"]],
            "Shame": [["i", "iv", "V", "i"], ["i", "♭VI", "V", "i"], ["i", "iv", "♭II", "i"]],
            "Love": [["I", "♭VII", "IV", "I"], ["I", "IV", "♭VII", "I"], ["I", "♭VII", "V", "I"]],
            "Envy": [["i", "♯iv°", "V", "i"], ["i", "♭VI", "V", "♯iv°"], ["i", "♯iv°", "♭II", "i"]],
            "Aesthetic Awe": [["I", "II", "III+", "I"], ["I", "♯IVdim", "III+", "II"], ["I", "III+", "II", "I"]]
        }
        
        for emotion, chord_lists in sample_data.items():
            emotion_progressions = []
            for i, chords in enumerate(chord_lists):
                progression = ChordProgression(
                    chords=chords,
                    emotion=emotion,
                    mode=self.emotions_to_modes[emotion],
                    genres={"Pop": 0.8, "Rock": 0.6, "Jazz": 0.5},
                    progression_id=f"{emotion.lower()}_{i+1:03d}"
                )
                emotion_progressions.append(progression)
            progressions[emotion] = emotion_progressions
        
        return progressions
    
    def _build_emotion_keywords(self) -> Dict[str, List[str]]:
        """Keyword mapping for emotion detection in natural language"""
        return {
            "Joy": ["happy", "joy", "excited", "cheerful", "uplifted", "bright", "celebratory"],
            "Sadness": ["sad", "depressed", "grieving", "blue", "mournful", "melancholy", "sorrowful"],
            "Fear": ["afraid", "scared", "anxious", "nervous", "terrified", "worried", "tense"],
            "Anger": ["angry", "furious", "frustrated", "pissed", "irritated", "rage", "aggressive"],
            "Disgust": ["disgusted", "grossed out", "repulsed", "nauseated", "revolted"],
            "Surprise": ["surprised", "shocked", "amazed", "startled", "unexpected", "wonder"],
            "Trust": ["trust", "safe", "secure", "supported", "bonded", "intimate", "comfortable"],
            "Anticipation": ["anticipation", "expectation", "eager", "hopeful", "building", "yearning"],
            "Shame": ["guilt", "shame", "regret", "embarrassed", "remorseful"],
            "Love": ["love", "romantic", "affection", "caring", "warm", "tender", "devoted"],
            "Envy": ["jealous", "envious", "spiteful", "competitive", "bitter", "possessive"],
            "Aesthetic Awe": ["awe", "wonder", "sublime", "inspired", "majestic", "transcendent", "beautiful"]
        }


class EmotionParser(nn.Module):
    """
    BERT-based emotion parser that extracts weighted emotional content from text
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.emotion_classifier = nn.Linear(768, 12)  # 12 core emotions
        self.dropout = nn.Dropout(0.1)
        
        # Emotion index mapping
        self.emotion_labels = ["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
                              "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe"]
    
    def forward(self, text_input: str) -> torch.Tensor:
        """Convert text to emotion weight vector"""
        inputs = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            pooled_output = outputs.pooler_output
        
        emotion_logits = self.emotion_classifier(self.dropout(pooled_output))
        emotion_weights = F.softmax(emotion_logits, dim=-1)
        
        return emotion_weights
    
    def parse_emotion_weights(self, text: str) -> Dict[str, float]:
        """Parse text and return emotion weights using keyword matching"""
        text_lower = text.lower()
        
        # Initialize weights
        emotion_weights = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Load keyword mapping from database
        try:
            with open('emotion_progression_database.json', 'r') as f:
                data = json.load(f)
            emotion_keywords = data['parser_rules']['emotion_keywords']
        except:
            # Fallback keywords
            emotion_keywords = {
                "Joy": ["happy", "joy", "excited", "cheerful", "uplifted", "bright", "celebratory"],
                "Sadness": ["sad", "depressed", "grieving", "blue", "mournful", "melancholy", "sorrowful"],
                "Fear": ["afraid", "scared", "anxious", "nervous", "terrified", "worried", "tense"],
                "Anger": ["angry", "furious", "frustrated", "pissed", "irritated", "rage", "aggressive"],
                "Disgust": ["disgusted", "grossed out", "repulsed", "nauseated", "revolted"],
                "Surprise": ["surprised", "shocked", "amazed", "startled", "unexpected", "wonder"],
                "Trust": ["trust", "safe", "secure", "supported", "bonded", "intimate", "comfortable"],
                "Anticipation": ["anticipation", "expectation", "eager", "hopeful", "building", "yearning"],
                "Shame": ["guilt", "shame", "regret", "embarrassed", "remorseful"],
                "Love": ["love", "romantic", "affection", "caring", "warm", "tender", "devoted"],
                "Envy": ["jealous", "envious", "spiteful", "competitive", "bitter", "possessive"],
                "Aesthetic Awe": ["awe", "wonder", "sublime", "inspired", "majestic", "transcendent", "beautiful"]
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
            # Default fallback - check for basic emotions
            if any(word in text_lower for word in ["happy", "joy", "good", "great"]):
                emotion_weights["Joy"] = 1.0
            elif any(word in text_lower for word in ["sad", "down", "depressed"]):
                emotion_weights["Sadness"] = 1.0
            elif any(word in text_lower for word in ["angry", "mad", "frustrated"]):
                emotion_weights["Anger"] = 1.0
            elif any(word in text_lower for word in ["scared", "afraid", "nervous"]):
                emotion_weights["Fear"] = 1.0
            else:
                # Very basic fallback
                emotion_weights["Joy"] = 1.0
        
        return emotion_weights


class ModeBlender(nn.Module):
    """
    Neural network that blends musical modes based on weighted emotional input
    """
    
    def __init__(self):
        super().__init__()
        self.emotion_to_mode = nn.Linear(12, 12)  # 12 emotions → 12 modes
        self.mode_labels = ["Ionian", "Aeolian", "Phrygian", "Phrygian Dominant", "Locrian", 
                           "Lydian", "Dorian", "Melodic Minor", "Harmonic Minor", "Mixolydian", 
                           "Hungarian Minor", "Lydian Augmented"]
    
    def forward(self, emotion_weights: torch.Tensor) -> torch.Tensor:
        """Convert emotion weights to mode blend"""
        mode_logits = self.emotion_to_mode(emotion_weights)
        mode_weights = F.softmax(mode_logits, dim=-1)
        return mode_weights
    
    def get_primary_mode(self, emotion_weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Get primary mode and full mode blend from emotion weights using database mapping"""
        # Direct mapping from database
        emotion_to_mode_mapping = {
            "Joy": "Ionian",
            "Sadness": "Aeolian", 
            "Fear": "Phrygian",
            "Anger": "Phrygian Dominant",
            "Disgust": "Locrian",
            "Surprise": "Lydian",
            "Trust": "Dorian",
            "Anticipation": "Melodic Minor",
            "Shame": "Harmonic Minor", 
            "Love": "Mixolydian",
            "Envy": "Hungarian Minor",
            "Aesthetic Awe": "Lydian Augmented"
        }
        
        # Get primary emotion 
        primary_emotion = max(emotion_weights, key=emotion_weights.get)
        primary_mode = emotion_to_mode_mapping[primary_emotion]
        
        # Create mode blend based on emotion weights
        mode_blend = {}
        for emotion, weight in emotion_weights.items():
            mode = emotion_to_mode_mapping[emotion]
            if mode in mode_blend:
                mode_blend[mode] += weight
            else:
                mode_blend[mode] = weight
        
        # Normalize mode blend
        total_weight = sum(mode_blend.values())
        mode_blend = {mode: weight/total_weight for mode, weight in mode_blend.items()}
        
        return primary_mode, mode_blend


class ChordProgressionGenerator(nn.Module):
    """
    Transformer-based model that generates chord progressions from mode and genre context
    """
    
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=6, max_length=16):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Chord vocabulary (Roman numerals + extensions)
        self.chord_vocab = self._build_chord_vocab()
        self.vocab_size = len(self.chord_vocab)
        
        # Token embeddings
        self.chord_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Context embeddings (mode + genre)
        self.mode_embedding = nn.Embedding(12, d_model)  # 12 modes
        self.genre_embedding = nn.Embedding(50, d_model)  # 50 common genres
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, self.vocab_size)
        
    def _build_chord_vocab(self) -> Dict[str, int]:
        """Build vocabulary of chord symbols"""
        chords = ["I", "II", "III", "IV", "V", "VI", "VII",
                 "i", "ii", "iii", "iv", "v", "vi", "vii",
                 "♭II", "♭III", "♭V", "♭VI", "♭VII",
                 "♯IV", "♯V", "♯vi", "♯iv°", "III+", "♯IVdim",
                 "i°", "ii°", "iii°", "iv°", "v°", "vi°", "vii°",
                 "iiø", "PAD", "START", "END"]
        
        return {chord: idx for idx, chord in enumerate(chords)}
    
    def forward(self, mode_context: torch.Tensor, genre_context: torch.Tensor, 
                target_chords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate chord progression given mode and genre context"""
        batch_size = mode_context.size(0)
        device = mode_context.device
        
        # Create context embedding
        mode_emb = self.mode_embedding(mode_context)
        genre_emb = self.genre_embedding(genre_context)
        context = mode_emb + genre_emb  # [batch_size, d_model]
        
        if target_chords is not None:
            # Training mode
            seq_len = target_chords.size(1)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            chord_emb = self.chord_embedding(target_chords)
            pos_emb = self.position_embedding(pos_ids)
            
            decoder_input = chord_emb + pos_emb
            memory = context.unsqueeze(1).expand(-1, seq_len, -1)
            
            output = self.transformer(decoder_input, memory)
            return self.output_projection(output)
        else:
            # Generation mode
            return self._generate_sequence(context)
    
    def _generate_sequence(self, context: torch.Tensor, max_length: int = 8) -> List[str]:
        """Generate chord sequence autoregressively"""
        batch_size = context.size(0)
        device = context.device
        
        # Start with START token
        generated = [self.chord_vocab["START"]]
        
        for i in range(max_length):
            # Convert to tensor
            input_ids = torch.tensor([generated], device=device)
            pos_ids = torch.arange(len(generated), device=device).unsqueeze(0)
            
            # Embeddings
            chord_emb = self.chord_embedding(input_ids)
            pos_emb = self.position_embedding(pos_ids)
            decoder_input = chord_emb + pos_emb
            
            # Memory from context
            memory = context.unsqueeze(1).expand(-1, len(generated), -1)
            
            # Forward pass
            output = self.transformer(decoder_input, memory)
            logits = self.output_projection(output)
            
            # Sample next token
            next_token_logits = logits[0, -1, :]
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()
            
            if next_token == self.chord_vocab["END"]:
                break
                
            generated.append(next_token)
        
        # Convert back to chord symbols
        idx_to_chord = {idx: chord for chord, idx in self.chord_vocab.items()}
        return [idx_to_chord[idx] for idx in generated[1:]]  # Skip START token


class ChordProgressionModel(nn.Module):
    """
    Complete pipeline: Text → Emotions → Modes → Chord Progressions
    """
    
    def __init__(self):
        super().__init__()
        self.database = EmotionMusicDatabase()
        self.emotion_parser = EmotionParser()
        self.mode_blender = ModeBlender()
        self.progression_generator = ChordProgressionGenerator()
        
    def generate_from_prompt(self, text_prompt: str, genre_preference: str = "Pop", 
                           num_progressions: int = 1) -> List[Dict]:
        """
        Main interface: Generate chord progressions from natural language
        
        Args:
            text_prompt: Natural language description (e.g., "romantic but anxious")
            genre_preference: Preferred musical genre
            num_progressions: Number of progressions to generate
            
        Returns:
            List of progression dictionaries with metadata
        """
        # 1. Parse emotions from text
        emotion_weights = self.emotion_parser.parse_emotion_weights(text_prompt)
        
        # 2. Get mode blend
        primary_mode, mode_blend = self.mode_blender.get_primary_mode(emotion_weights)
        
        # 3. Generate or select progressions
        results = []
        
        for i in range(num_progressions):
            # Option A: Neural generation (if trained)
            # chords = self._neural_generate(mode_blend, genre_preference)
            
            # Option B: Database lookup with blending (for immediate use)
            chords = self._database_select(emotion_weights, mode_blend, genre_preference)
            
            result = {
                "progression_id": f"generated_{i}",
                "prompt": text_prompt,
                "emotion_weights": emotion_weights,
                "primary_mode": primary_mode,
                "mode_blend": mode_blend,
                "chords": chords,
                "genre": genre_preference,
                "metadata": {
                    "generation_method": "database_selection",
                    "timestamp": datetime.now().isoformat()
                }
            }
            results.append(result)
            
        return results
    
    def _database_select(self, emotion_weights: Dict[str, float], 
                        mode_blend: Dict[str, float], genre: str) -> List[str]:
        """Select progression from database based on weighted emotions and genre"""
        # Find dominant emotions (weight > 0.1)
        dominant_emotions = [emotion for emotion, weight in emotion_weights.items() if weight > 0.1]
        
        if not dominant_emotions:
            dominant_emotions = [max(emotion_weights, key=emotion_weights.get)]
        
        # Collect candidate progressions
        candidates = []
        for emotion in dominant_emotions:
            if emotion in self.database.chord_progressions:
                emotion_progressions = self.database.chord_progressions[emotion]
                for prog in emotion_progressions:
                    # Weight by emotion strength and genre match
                    emotion_weight = emotion_weights[emotion]
                    genre_weight = prog.genres.get(genre, 0.1)  # Low default if genre not listed
                    combined_weight = emotion_weight * genre_weight
                    candidates.append((prog, combined_weight))
        
        if not candidates:
            # Fallback to basic progression
            return ["I", "V", "vi", "IV"]
        
        # Weighted random selection
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:5]  # Consider top 5
        
        weights = [c[1] for c in top_candidates]
        selected = random.choices(top_candidates, weights=weights, k=1)[0]
        
        return selected[0].chords
    
    def export_jam_session(self, results: List[Dict], session_id: str = None) -> Dict:
        """Export results in .jam session format"""
        if session_id is None:
            session_id = f"jam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Take first result as primary for session
        primary = results[0]
        
        jam_session = {
            "session_id": session_id,
            "prompt": primary["prompt"],
            "parsed_emotions": [{"emotion": k, "weight": v} for k, v in primary["emotion_weights"].items() if v > 0.05],
            "mode_map": [{"mode": k, "weight": v} for k, v in primary["mode_blend"].items() if v > 0.05],
            "mode_blend": primary["mode_blend"],
            "harmonic_profile": {
                "primary_mode": primary["primary_mode"],
                "secondary_modes": [mode for mode, weight in primary["mode_blend"].items() if 0.1 < weight < max(primary["mode_blend"].values())],
                "blend_type": f"Primary {primary['primary_mode']} with modal borrowing"
            },
            "chord_progressions": [
                {
                    "progression_id": result["progression_id"],
                    "chords": result["chords"],
                    "genre": result["genre"]
                } for result in results
            ],
            "metadata": {
                "key": "C",  # Default, could be parameterized
                "tempo": 120,
                "time_signature": "4/4",
                "created": datetime.now().isoformat(),
                "model_version": "1.0"
            }
        }
        
        return jam_session


# Usage example and training utilities
def create_training_data(database: EmotionMusicDatabase) -> List[Dict]:
    """Create training examples from the progression database"""
    training_data = []
    
    for emotion, progressions in database.chord_progressions.items():
        for prog in progressions:
            # Create synthetic prompts for each progression
            emotion_keywords = database.emotion_keywords[emotion]
            prompt = f"I want something {random.choice(emotion_keywords)}"
            
            example = {
                "prompt": prompt,
                "target_emotion": emotion,
                "target_mode": prog.mode,
                "target_chords": prog.chords,
                "genres": prog.genres
            }
            training_data.append(example)
    
    return training_data


def train_model(model: ChordProgressionModel, training_data: List[Dict], epochs: int = 10):
    """Training loop for the complete model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in training_data:
            optimizer.zero_grad()
            
            # Forward pass through emotion parser and mode blender
            emotion_weights = model.emotion_parser.parse_emotion_weights(batch["prompt"])
            primary_mode, mode_blend = model.mode_blender.get_primary_mode(emotion_weights)
            
            # Convert target data to tensors
            target_chords = [model.progression_generator.chord_vocab.get(chord, 0) for chord in batch["target_chords"]]
            target_tensor = torch.tensor([target_chords])
            
            # Mock mode and genre context (would need proper encoding)
            mode_context = torch.tensor([0])  # Would map mode to index
            genre_context = torch.tensor([0])  # Would map genre to index
            
            # Generate predictions
            predictions = model.progression_generator(mode_context, genre_context, target_tensor)
            
            # Calculate loss (simplified)
            loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), target_tensor.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")


if __name__ == "__main__":
    # Initialize the model
    model = ChordProgressionModel()
    
    # Example usage
    prompts = [
        "romantic but a little anxious",
        "uplifting and hopeful", 
        "dark and mysterious",
        "energetic and angry",
        "sad but beautiful"
    ]
    
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        results = model.generate_from_prompt(prompt, genre_preference="Pop", num_progressions=2)
        
        for result in results:
            print(f"Emotion weights: {result['emotion_weights']}")
            print(f"Primary mode: {result['primary_mode']}")
            print(f"Chord progression: {' - '.join(result['chords'])}")
        
        # Export as .jam session
        session = model.export_jam_session(results, f"session_{prompt.replace(' ', '_')}")
        print(f"Session ID: {session['session_id']}")
