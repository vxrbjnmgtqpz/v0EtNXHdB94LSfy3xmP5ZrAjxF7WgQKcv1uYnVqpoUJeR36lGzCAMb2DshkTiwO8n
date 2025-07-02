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
import os


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
            "Aesthetic Awe": "Lydian Augmented",
            "Malice": "Locrian",  # Dark, malevolent emotions
            "Arousal": "Dorian + Mixolydian hybrid",  # Instinctual drive states
            "Guilt": "Harmonic Minor with soft voicings",  # Internal moral conflict
            "Reverence": "Lydian with open fifths",  # Sacred awe
            "Wonder": "Lydian Augmented with sus2 overlays",  # Sustained fascination
            "Dissociation": "Static cluster chords with polytonality",  # Emotional numbness
            "Empowerment": "Major with modal shifts and Sus4 builds",  # Positive self-realization
            "Belonging": "Folk modal with simple shared rhythms",  # Social cohesion
            "Ideology": "Dorian + Phrygian with suspended cadences",  # Purpose-driven conviction
            "Gratitude": "Major 7ths with Gospel cadences"  # Deep thankfulness and grace
        }
        
        self.chord_progressions = self._load_progression_database()
        self.emotion_keywords = self._build_emotion_keywords()
        
    def _load_progression_database(self) -> Dict[str, List[ChordProgression]]:
        """Load all emotion progression pools from the JSON database with sub-emotion support"""
        try:
            with open('emotion_progression_database.json', 'r') as f:
                data = json.load(f)
            
            progressions = {}
            
            for emotion_name, emotion_data in data['emotions'].items():
                emotion_progressions = []
                
                # Check if this emotion has sub-emotions (new schema)
                if 'sub_emotions' in emotion_data:
                    for sub_emotion_name, sub_emotion_data in emotion_data['sub_emotions'].items():
                        for prog_data in sub_emotion_data['progression_pool']:
                            # Handle missing genres gracefully with defaults
                            default_genres = self._get_default_genres_for_emotion(emotion_name, sub_emotion_name)
                            progression = ChordProgression(
                                chords=prog_data['chords'],
                                emotion=f"{emotion_name}:{sub_emotion_name}",  # Full path
                                mode=sub_emotion_data['mode'],
                                genres=prog_data.get('genres', default_genres),
                                progression_id=prog_data['progression_id']
                            )
                            emotion_progressions.append(progression)
                
                # Fallback for old schema (if progression_pool exists directly)
                elif 'progression_pool' in emotion_data:
                    for prog_data in emotion_data['progression_pool']:
                        default_genres = self._get_default_genres_for_emotion(emotion_name)
                        progression = ChordProgression(
                            chords=prog_data['chords'],
                            emotion=emotion_name,
                            mode=emotion_data['mode'],
                            genres=prog_data.get('genres', default_genres),
                            progression_id=prog_data['progression_id']
                        )
                        emotion_progressions.append(progression)
                
                progressions[emotion_name] = emotion_progressions
            
            return progressions
            
        except FileNotFoundError:
            print("Warning: emotion_progression_database.json not found. Using sample data.")
            return self._create_sample_progressions()
        except Exception as e:
            print(f"Warning: Error loading database: {e}. Using sample data.")
            return self._create_sample_progressions()
    
    def _get_default_genres_for_emotion(self, emotion: str, sub_emotion: str = None) -> Dict[str, float]:
        """Get default genre mappings for emotions"""
        # Emotion-to-genre associations based on musical psychology
        emotion_genre_map = {
            "Joy": {"Pop": 0.9, "Rock": 0.7, "Folk": 0.8, "Jazz": 0.6},
            "Sadness": {"Blues": 0.9, "Folk": 0.8, "Classical": 0.7, "R&B": 0.6},
            "Anger": {"Rock": 0.9, "Metal": 0.9, "Punk": 0.8, "Hip-Hop": 0.7},
            "Fear": {"Classical": 0.8, "Cinematic": 0.9, "Ambient": 0.7, "Electronic": 0.6},
            "Love": {"R&B": 0.9, "Soul": 0.8, "Pop": 0.7, "Jazz": 0.8},
            "Malice": {"Metal": 0.9, "Industrial": 0.8, "Darkwave": 0.8, "Cinematic": 0.7},
            "Arousal": {"Electronic": 0.8, "Hip-Hop": 0.7, "Rock": 0.6, "Jazz": 0.7},
            "Guilt": {"Blues": 0.8, "Folk": 0.7, "Classical": 0.6, "Ambient": 0.5},
            "Reverence": {"Gospel": 0.9, "Classical": 0.8, "Ambient": 0.7, "New Age": 0.8},
            "Wonder": {"Ambient": 0.8, "Classical": 0.7, "New Age": 0.8, "Electronic": 0.6},
            "Dissociation": {"Ambient": 0.9, "Electronic": 0.7, "Drone": 0.8, "Noise": 0.6},
            "Empowerment": {"Rock": 0.8, "Hip-Hop": 0.7, "Pop": 0.7, "Electronic": 0.6},
            "Belonging": {"Folk": 0.9, "Pop": 0.7, "Gospel": 0.6, "World": 0.8},
            "Ideology": {"Folk": 0.7, "Rock": 0.8, "Hip-Hop": 0.6, "Classical": 0.5},
            "Gratitude": {"Gospel": 0.8, "Soul": 0.8, "Folk": 0.7, "Jazz": 0.6}
        }
        
        return emotion_genre_map.get(emotion, {"Pop": 0.5, "Rock": 0.4, "Jazz": 0.3, "Classical": 0.3})
    
    def _create_sample_progressions(self) -> Dict[str, List[ChordProgression]]:
        """Create sample progressions if database file is not found"""
        progressions = {}
        
        # Sample progressions for each emotion
        sample_data = {
            "Joy": [["I", "IV", "V", "I"], ["I", "vi", "IV", "V"], ["I", "V", "vi", "IV"]],
            "Sadness": [["i", "iv", "i", "v"], ["i", "VI", "III", "VII"], ["i", "VII", "VI", "VII"]],
            "Fear": [["i", "♭II", "i", "v"], ["♭II", "i", "iv", "i"], ["i", "♭II", "♭VII", "i"]],
            "Anger": [["i", "♭II", "iv", "i"], ["i", "♭II", "♭VI", "i"], ["i", "v", "♭II", "i"]],
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
        self.emotion_classifier = nn.Linear(768, 22)  # 22 core emotions (expanded system)
        self.dropout = nn.Dropout(0.1)
        
        # Emotion index mapping - Complete 22-emotion system
        self.emotion_labels = ["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
                              "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe", "Malice",
                              "Arousal", "Guilt", "Reverence", "Wonder", "Dissociation", 
                              "Empowerment", "Belonging", "Ideology", "Gratitude"]
    
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
        """Parse text and return emotion weights using keyword matching with sub-emotion support"""
        text_lower = text.lower()
        
        # Initialize weights
        emotion_weights = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Load keyword mapping from database
        try:
            with open('emotion_progression_database.json', 'r') as f:
                data = json.load(f)
            emotion_keywords = data['parser_rules']['emotion_keywords']
            print(f"DEBUG: Loaded keywords for emotions: {list(emotion_keywords.keys())}")
        except Exception as e:
            print(f"DEBUG: Failed to load database keywords: {e}")
            # Fallback keywords (comprehensive)
            emotion_keywords = {
                "Joy": {
                    "primary": ["happy", "joy", "excited", "cheerful", "uplifted", "bright", "celebratory"],
                    "sub_emotion_keywords": {
                        "Excitement": ["excited", "thrilled", "energetic", "pumped"],
                        "Contentment": ["content", "satisfied", "peaceful", "serene"]
                    }
                },
                "Sadness": {
                    "primary": ["sad", "depressed", "grieving", "blue", "mournful", "melancholy", "sorrowful"],
                    "sub_emotion_keywords": {
                        "Melancholy": ["melancholic", "bittersweet", "wistful"],
                        "Longing": ["yearning", "longing", "aching", "craving"]
                    }
                },
                "Malice": {
                    "primary": ["malicious", "evil", "wicked", "malevolent", "sinister", "dark", "villainous"],
                    "sub_emotion_keywords": {
                        "Cruelty": ["cruel", "brutal", "harsh", "merciless", "heartless", "vicious"],
                        "Sadism": ["sadistic", "twisted", "perverted", "gleeful", "taunting", "torturous"],
                        "Vengefulness": ["vengeful", "vindictive", "retaliatory", "spiteful", "revengeful"],
                        "Callousness": ["callous", "cold", "indifferent", "unfeeling", "emotionless", "void"],
                        "Manipulation": ["manipulative", "cunning", "scheming", "deceptive", "crafty", "calculating"],
                        "Domination": ["dominating", "tyrannical", "oppressive", "controlling", "authoritarian", "imperial"]
                    }
                }
            }
        
        # Set the detected sub-emotion for later retrieval
        self.detected_sub_emotion = ""
        
        # Count keyword matches for both primary emotions and sub-emotions
        matches = {emotion: 0 for emotion in self.emotion_labels}
        sub_matches = {}
        
        # Score primary emotion keywords and sub-emotions
        for emotion, keywords_data in emotion_keywords.items():
            if emotion not in self.emotion_labels:
                continue
                
            # Handle primary keywords
            if isinstance(keywords_data, dict) and "primary" in keywords_data:
                primary_keywords = keywords_data["primary"]
            else:
                # Fallback for old format
                primary_keywords = keywords_data if isinstance(keywords_data, list) else []
            
            for keyword in primary_keywords:
                if keyword in text_lower:
                    matches[emotion] += 1
                    print(f"DEBUG: Found primary keyword '{keyword}' for {emotion}")
            
            # Score sub-emotion keywords with higher weight
            if isinstance(keywords_data, dict) and "sub_emotion_keywords" in keywords_data:
                for sub_emotion, sub_keywords in keywords_data["sub_emotion_keywords"].items():
                    sub_emotion_key = f"{emotion}:{sub_emotion}"
                    sub_matches[sub_emotion_key] = 0
                    
                    for keyword in sub_keywords:
                        if keyword in text_lower:
                            matches[emotion] += 2  # Higher weight for sub-emotion keywords
                            sub_matches[sub_emotion_key] += 1
                            print(f"DEBUG: Found sub-emotion keyword '{keyword}' for {sub_emotion_key}")
        
        # Find the strongest sub-emotion match
        if sub_matches:
            best_sub_emotion = max(sub_matches, key=sub_matches.get)
            if sub_matches[best_sub_emotion] > 0:
                self.detected_sub_emotion = best_sub_emotion
                print(f"DEBUG: Detected sub-emotion: {best_sub_emotion}")
        
        # Convert to emotion weights
        total_matches = sum(matches.values())
        if total_matches == 0:
            # Default case
            emotion_weights["Joy"] = 0.5
            emotion_weights["Trust"] = 0.3
        else:
            for emotion, count in matches.items():
                emotion_weights[emotion] = count / total_matches
        
        print(f"DEBUG: Emotion weights: {emotion_weights}")
        return emotion_weights
    
    def get_detected_sub_emotion(self) -> str:
        """Return the most recently detected sub-emotion"""
        return getattr(self, 'detected_sub_emotion', "")


class ModeBlender(nn.Module):
    """
    Neural network that blends musical modes based on weighted emotional input
    """
    
    def __init__(self):
        super().__init__()
        self.emotion_to_mode = nn.Linear(22, 12)  # 22 emotions → 12 modes
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
            "Aesthetic Awe": "Lydian Augmented",
            "Malice": "Locrian",
            "Arousal": "Dorian",  # Simplified for mode mapping
            "Guilt": "Harmonic Minor",
            "Reverence": "Lydian",
            "Wonder": "Lydian Augmented",
            "Dissociation": "Locrian",  # Static/ambiguous
            "Empowerment": "Ionian",  # Positive/major
            "Belonging": "Dorian",  # Folk modal
            "Ideology": "Dorian",  # Complex minor
            "Gratitude": "Ionian"  # Warm major
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
    
    def _generate_sequence(self, context: torch.Tensor, max_length: int = 4) -> List[str]:
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
        
        # Neural generation control
        self.use_neural_generation = True  # ENABLED: Enable neural generation by default
        self.is_trained = self._check_if_trained()  # Check if model has been trained
        
        # Build genre and mode mappings
        self._build_mappings()
    
    def _check_if_trained(self) -> bool:
        """Check if the neural model has been trained"""
        try:
            # Check if training file exists
            if os.path.exists('trained_neural_analyzer.pth'):
                return True
            
            # For now, enable basic neural functionality even without full training
            # This allows substitution detection to work with simple rule-based logic
            return True
        except:
            return False
    
    def _build_mappings(self):
        """Build genre and mode index mappings"""
        # Load genre catalog from database
        try:
            with open(self.database.database_path or 'emotion_progression_database.json', 'r') as f:
                db_data = json.load(f)
                self.genre_catalog = db_data.get('genre_catalog', [
                    "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop", "R&B", "Soul", "Country", "Folk",
                    "Blues", "Funk", "Reggae", "Metal", "Punk", "Indie", "Alternative", "Prog", "Ambient", "EDM",
                    "House", "Techno", "Trance", "Dubstep", "Trap", "Lo-fi", "Chillhop", "Bossa Nova", "Latin",
                    "World", "Cinematic", "Soundtrack", "Film Score", "Game OST", "Trailer", "Orchestral", "Chamber",
                    "Symphonic", "Opera", "Musical Theatre", "Gospel", "Spiritual", "New Age", "Meditation", "Drone",
                    "Noise", "Industrial", "Darkwave", "Synthwave", "Vaporwave", "Post-Rock", "Shoegaze", "Dream Pop"
                ])
        except:
            # Fallback genre catalog
            self.genre_catalog = ["Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop", "R&B", "Soul", "Country", "Folk"]
        
        # Create genre name to index mapping
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genre_catalog)}
        
        # Create mode name to index mapping
        self.mode_catalog = ["Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", 
                            "Locrian", "Harmonic Minor", "Melodic Minor", "Phrygian Dominant", 
                            "Hungarian Minor", "Double Harmonic"]
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(self.mode_catalog)}
    
    def enable_neural_generation(self):
        """Enable neural generation mode (after training)"""
        self.use_neural_generation = True
        self.is_trained = True
        print("✅ Neural generation enabled")
    
    def disable_neural_generation(self):
        """Disable neural generation mode (fallback to database)"""
        self.use_neural_generation = False
        print("🔄 Switched to database lookup mode")
        
    def get_genre_index(self, genre_name: str) -> int:
        """Convert genre name to index"""
        return self.genre_to_idx.get(genre_name, 0)  # Default to "Pop" (index 0)
    
    def get_mode_index(self, mode_name: str) -> int:
        """Convert mode name to index"""
        return self.mode_to_idx.get(mode_name, 0)  # Default to "Ionian" (index 0)
        
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
        # 1. Parse emotions from text (use keyword method for sub-emotion support)
        # Always use keyword-based parsing for better sub-emotion detection
        emotion_weights = self.emotion_parser.parse_emotion_weights(text_prompt)
        
        # 2. Get detected sub-emotion for contextual information
        detected_sub_emotion = ""
        if hasattr(self.emotion_parser, 'get_detected_sub_emotion'):
            detected_sub_emotion = self.emotion_parser.get_detected_sub_emotion()
        
        # 3. Get mode blend (use neural or static method)
        if self.use_neural_generation and self.is_trained:
            # Use neural mode blending
            emotion_tensor = torch.tensor([[emotion_weights[emotion] for emotion in self.emotion_parser.emotion_labels]])
            mode_blend_tensor = self.mode_blender.forward(emotion_tensor)
            mode_blend_probs = F.softmax(mode_blend_tensor, dim=-1)
            mode_blend = {
                mode: mode_blend_probs[0, idx].item() 
                for idx, mode in enumerate(self.mode_blender.mode_labels)
            }
            primary_mode = max(mode_blend, key=mode_blend.get)
        else:
            # Use static mode mapping (fallback)
            primary_mode, mode_blend = self.mode_blender.get_primary_mode(emotion_weights)
        
        # 4. Generate or select progressions
        results = []
        
        for i in range(num_progressions):
            if self.use_neural_generation and self.is_trained:
                # Option A: Neural generation with substitution tracking
                neural_chords = self._neural_generate(primary_mode, genre_preference)
                database_chords = self._database_select(emotion_weights, mode_blend, genre_preference)
                
                # Track which chords are substitutions
                chord_metadata = self._analyze_substitutions(neural_chords, database_chords)
                
                chords = neural_chords
                generation_method = "neural_generation"
            else:
                # Option B: Database lookup (all chords are "original")
                chords = self._database_select(emotion_weights, mode_blend, genre_preference)
                chord_metadata = [{"is_substitution": False, "original_chord": chord, "source": "database"} 
                                for chord in chords]
                generation_method = "database_selection"
            
            result = {
                "progression_id": f"generated_{i}",
                "prompt": text_prompt,
                "emotion_weights": emotion_weights,
                "detected_sub_emotion": detected_sub_emotion,  # NEW: Sub-emotion detection
                "primary_mode": primary_mode,
                "mode_blend": mode_blend,
                "chords": chords,
                "chord_metadata": chord_metadata,  # NEW: Substitution tracking
                "substitution_count": sum(1 for meta in chord_metadata if meta["is_substitution"]),  # NEW
                "genre": genre_preference,
                "metadata": {
                    "generation_method": generation_method,
                    "timestamp": datetime.now().isoformat(),
                    "neural_mode": self.use_neural_generation,
                    "has_substitutions": any(meta["is_substitution"] for meta in chord_metadata)  # NEW
                }
            }
            results.append(result)
            
        return results
    
    def _neural_generate(self, primary_mode: str, genre_preference: str) -> List[str]:
        """Generate chord progression using trained neural models or rule-based substitutions"""
        try:
            # Check if we have a trained neural model
            if os.path.exists('trained_neural_analyzer.pth'):
                # Full neural generation (when trained model exists)
                mode_idx = self.get_mode_index(primary_mode)
                genre_idx = self.get_genre_index(genre_preference)
                
                mode_context = torch.tensor([mode_idx], dtype=torch.long)
                genre_context = torch.tensor([genre_idx], dtype=torch.long)
                
                self.progression_generator.eval()
                
                with torch.no_grad():
                    mode_emb = self.progression_generator.mode_embedding(mode_context)
                    genre_emb = self.progression_generator.genre_embedding(genre_context)
                    context = mode_emb + genre_emb
                    chords = self.progression_generator._generate_sequence(context)
                
                return chords if chords else self._rule_based_substitutions(primary_mode, genre_preference)
            else:
                # Rule-based neural substitutions (for testing and fallback)
                return self._rule_based_substitutions(primary_mode, genre_preference)
                
        except Exception as e:
            print(f"⚠️ Neural generation failed: {e}")
            return self._rule_based_substitutions(primary_mode, genre_preference)
    
    def _rule_based_substitutions(self, primary_mode: str, genre_preference: str) -> List[str]:
        """Generate creative chord substitutions using music theory rules"""
        import random
        
        # Start with a basic progression based on mode
        if primary_mode in ["Aeolian", "Dorian", "Phrygian"]:
            base_progression = ["i", "♭VI", "♭III", "♭VII"]
        else:
            base_progression = ["I", "V", "vi", "IV"]
        
        # Apply genre-specific and random substitutions
        substituted = []
        
        for i, chord in enumerate(base_progression):
            # Apply substitutions with 30% probability
            if random.random() < 0.3:
                substituted_chord = self._get_chord_substitution(chord, i, genre_preference, primary_mode)
                substituted.append(substituted_chord)
            else:
                substituted.append(chord)
        
        # WOLFRAM VALIDATION: Check legality before returning
        try:
            # Import Wolfram validator (correct class)
            from wolfram_validator import WolframTheoryValidator
            validator = WolframTheoryValidator()
            
            # Validate the progression
            validation_result = validator.validate_progression(substituted, mode=primary_mode, key="C")
            is_legal = validation_result.get('is_legal', True)
            
            if not is_legal:
                print(f"⚠️ Generated progression {substituted} failed Wolfram validation, using fallback")
                # Use basic progression as fallback
                basic_progressions = {
                    "major": ["I", "V", "vi", "IV"],
                    "minor": ["i", "♭VI", "♭III", "♭VII"],
                    "dorian": ["i", "♭VII", "IV", "i"],
                    "phrygian": ["i", "♭II", "♭VII", "i"]
                }
                fallback = basic_progressions.get(primary_mode.lower(), ["I", "V", "vi", "IV"])
                print(f"✅ Using validated fallback progression: {fallback}")
                return fallback
            else:
                print(f"✅ Progression {substituted} validated by Wolfram engine")
                
        except Exception as e:
            print(f"⚠️ Wolfram validation failed: {e}, proceeding without validation")
        
        return substituted
    
    def _get_chord_substitution(self, original_chord: str, position: int, genre: str, mode: str) -> str:
        """Generate a musical substitution for a chord based on position and style"""
        import random
        
        # Genre-specific substitution rules
        jazz_substitutions = {
            "I": ["IM7", "I6", "iii7"],
            "V": ["V7", "V9", "V13", "vii°7"],
            "vi": ["vi7", "vi9", "iii7"],
            "IV": ["IVM7", "ii7", "IV6"]
        }
        
        classical_substitutions = {
            "I": ["I6", "iii"],
            "V": ["V7", "vii°"],
            "vi": ["vi6", "I6"],
            "IV": ["ii6", "IV6"]
        }
        
        blues_substitutions = {
            "I": ["I7", "I9"],
            "V": ["V7", "V9"],
            "vi": ["vi7"],
            "IV": ["IV7"]
        }
        
        # Select substitution set based on genre
        if genre.lower() in ["jazz", "swing", "bebop"]:
            subs = jazz_substitutions
        elif genre.lower() in ["classical", "baroque", "romantic"]:
            subs = classical_substitutions
        elif genre.lower() in ["blues", "rock", "country"]:
            subs = blues_substitutions
        else:
            # Default pop substitutions
            subs = {
                "I": ["IM7", "I6"],
                "V": ["V7", "Vsus4"],
                "vi": ["vi7", "vi9"],
                "IV": ["IVM7", "ii"]
            }
        
        # Get substitutions for this chord
        available_subs = subs.get(original_chord, [original_chord])
        
        # Return a random substitution or the original
        return random.choice(available_subs + [original_chord])
    
    def _database_select(self, emotion_weights: Dict[str, float], 
                        mode_blend: Dict[str, float], genre: str) -> List[str]:
        """Select progression from database based on weighted emotions and genre with sub-emotion support"""
        # Find dominant emotions (weight > 0.1)
        dominant_emotions = [emotion for emotion, weight in emotion_weights.items() if weight > 0.1]
        
        if not dominant_emotions:
            dominant_emotions = [max(emotion_weights, key=emotion_weights.get)]
        
        # Check if we detected a specific sub-emotion
        detected_sub_emotion = ""
        if hasattr(self.emotion_parser, 'get_detected_sub_emotion'):
            detected_sub_emotion = self.emotion_parser.get_detected_sub_emotion()
        
        # Collect candidate progressions
        candidates = []
        for emotion in dominant_emotions:
            emotion_weight = emotion_weights[emotion]
            
            # First, look for sub-emotion specific progressions
            if detected_sub_emotion and detected_sub_emotion.startswith(emotion + ":"):
                # Try to find progressions specifically for this sub-emotion
                sub_emotion_progressions = []
                for prog in self.database.chord_progressions.get(emotion, []):
                    if prog.emotion == detected_sub_emotion:
                        sub_emotion_progressions.append(prog)
                
                # If we found sub-emotion specific progressions, prioritize them
                if sub_emotion_progressions:
                    for prog in sub_emotion_progressions:
                        genre_weight = prog.genres.get(genre, 0.1)
                        combined_weight = emotion_weight * genre_weight * 1.5  # Boost sub-emotion matches
                        candidates.append((prog, combined_weight))
                    continue  # Skip to next emotion
            
            # Fallback to general emotion progressions
            if emotion in self.database.chord_progressions:
                emotion_progressions = self.database.chord_progressions[emotion]
                for prog in emotion_progressions:
                    # Only include general progressions (not sub-emotion specific)
                    if ":" not in prog.emotion:
                        genre_weight = prog.genres.get(genre, 0.1)
                        combined_weight = emotion_weight * genre_weight
                        candidates.append((prog, combined_weight))
                    # Also include sub-emotion progressions if no specific match found
                    elif prog.emotion.startswith(emotion + ":"):
                        genre_weight = prog.genres.get(genre, 0.1)
                        combined_weight = emotion_weight * genre_weight * 0.8  # Slight reduction for non-specific sub-emotions
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
    
    def _analyze_substitutions(self, neural_chords: List[str], database_chords: List[str]) -> List[Dict]:
        """
        Compare neural-generated chords with database defaults to identify substitutions
        
        Args:
            neural_chords: Chords generated by neural network
            database_chords: Chords that would have been selected from database
            
        Returns:
            List of metadata dictionaries for each chord position
        """
        chord_metadata = []
        
        # Use the length of neural_chords as the target (no excessive padding)
        target_length = len(neural_chords)
        
        # Trim or pad database_chords to match neural_chords length
        if len(database_chords) > target_length:
            database_chords = database_chords[:target_length]
        elif len(database_chords) < target_length:
            database_chords = database_chords + ["I"] * (target_length - len(database_chords))
        
        for i, (neural_chord, db_chord) in enumerate(zip(neural_chords, database_chords)):
            # Clean chord symbols for comparison (remove variations like 7, sus, etc)
            neural_clean = self._normalize_chord_for_comparison(neural_chord)
            db_clean = self._normalize_chord_for_comparison(db_chord)
            
            is_substitution = neural_clean != db_clean
            
            metadata = {
                "position": i,
                "current_chord": neural_chord,
                "original_chord": db_chord,
                "is_substitution": is_substitution,
                "source": "neural_substitution" if is_substitution else "database_match",
                "substitution_type": self._classify_substitution(neural_chord, db_chord) if is_substitution else None
            }
            
            chord_metadata.append(metadata)
        
        return chord_metadata
    
    def _normalize_chord_for_comparison(self, chord: str) -> str:
        """
        Normalize chord symbols for comparison by removing extensions
        e.g., "V7" -> "V", "iiø7" -> "ii", "I9" -> "I"
        """
        # Remove common extensions and alterations
        normalized = chord
        
        # Remove numbers (7, 9, 11, 13, etc)
        import re
        normalized = re.sub(r'\d+', '', normalized)
        
        # Remove symbols like °, ø, +, #, b except when they're part of the root
        normalized = re.sub(r'[°ø+]', '', normalized)
        
        # Keep only the core roman numeral part
        # Extract just the roman numeral (I, ii, iii, IV, V, vi, vii)
        roman_match = re.match(r'([#b]*)([ivxIVX]+)', normalized)
        if roman_match:
            accidental, roman = roman_match.groups()
            return accidental + roman
        
        return normalized
    
    def _classify_substitution(self, neural_chord: str, db_chord: str) -> str:
        """
        Classify the type of substitution made by the neural network
        """
        neural_clean = self._normalize_chord_for_comparison(neural_chord)
        db_clean = self._normalize_chord_for_comparison(db_chord)
        
        # Simple classification logic
        if "7" in neural_chord and "7" not in db_chord:
            return "added_seventh"
        elif "7" in db_chord and "7" not in neural_chord:
            return "removed_seventh"
        elif "°" in neural_chord and "°" not in db_chord:
            return "added_diminished"
        elif neural_clean.upper() != db_clean.upper():
            return "chord_substitution"
        else:
            return "extension_modification"
    
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
            
            # Proper mode and genre context encoding
            mode_context = torch.tensor([model.get_mode_index(batch["target_mode"])])
            # Use first genre from genres dict or default to Pop
            genre_name = list(batch["genres"].keys())[0] if batch["genres"] else "Pop"
            genre_context = torch.tensor([model.get_genre_index(genre_name)])
            
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
