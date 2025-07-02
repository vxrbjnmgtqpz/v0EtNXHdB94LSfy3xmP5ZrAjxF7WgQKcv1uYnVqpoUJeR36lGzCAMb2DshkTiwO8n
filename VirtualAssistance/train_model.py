"""
Training and Evaluation Script for Chord Progression Generation Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Dict, Tuple
import random
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from chord_progression_model import (
    ChordProgressionModel, 
    EmotionMusicDatabase, 
    ChordProgression,
    EmotionParser,
    ModeBlender,
    ChordProgressionGenerator
)


class ChordProgressionDataset(Dataset):
    """PyTorch Dataset for training the chord progression model"""
    
    def __init__(self, database_path: str, augment_data: bool = True):
        self.database = EmotionMusicDatabase()
        
        # Load the JSON database
        with open(database_path, 'r') as f:
            self.data = json.load(f)
        
        self.training_examples = self._create_training_examples(augment_data)
        
    def _create_training_examples(self, augment: bool) -> List[Dict]:
        """Generate training examples from the database"""
        examples = []
        
        for emotion_name, emotion_data in self.data['emotions'].items():
            for prog_data in emotion_data['progression_pool']:
                # Create base example
                example = {
                    'prompt': self._generate_prompt(emotion_name),
                    'emotion': emotion_name,
                    'mode': emotion_data['mode'],
                    'chords': prog_data['chords'],
                    'genres': prog_data['genres'],
                    'progression_id': prog_data['progression_id']
                }
                examples.append(example)
                
                if augment:
                    # Create augmented examples with variations
                    for _ in range(3):  # 3 variations per base example
                        augmented = {
                            'prompt': self._generate_prompt(emotion_name, variation=True),
                            'emotion': emotion_name,
                            'mode': emotion_data['mode'],
                            'chords': prog_data['chords'],
                            'genres': prog_data['genres'],
                            'progression_id': f"{prog_data['progression_id']}_aug_{_}"
                        }
                        examples.append(augmented)
        
        return examples
    
    def _generate_prompt(self, emotion: str, variation: bool = False) -> str:
        """Generate natural language prompts for training"""
        keywords = self.data['parser_rules']['emotion_keywords'][emotion]
        modifiers = list(self.data['parser_rules']['intensity_modifiers'].keys())
        
        if not variation:
            # Simple prompt
            return f"I want something {random.choice(keywords)}"
        else:
            # More complex prompts with modifiers and combinations
            templates = [
                f"I'm feeling {random.choice(modifiers)} {random.choice(keywords)}",
                f"Create music that's {random.choice(keywords)} and {random.choice(keywords)}",
                f"Something {random.choice(modifiers)} {random.choice(keywords)} but also {random.choice(keywords)}",
                f"I need {random.choice(keywords)} vibes",
                f"Make it {random.choice(keywords)} with a touch of {random.choice(keywords)}"
            ]
            return random.choice(templates)
    
    def __len__(self):
        return len(self.training_examples)
    
    def __getitem__(self, idx):
        example = self.training_examples[idx]
        return {
            'prompt': example['prompt'],
            'emotion': example['emotion'],
            'mode': example['mode'],
            'chords': example['chords'],
            'genres': example['genres']
        }


class ModelTrainer:
    """Trainer class for the chord progression model"""
    
    def __init__(self, model: ChordProgressionModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'loss': [], 'emotion_accuracy': [], 'mode_accuracy': []}
        
    def train_emotion_parser(self, dataloader: DataLoader, epochs: int = 10, lr: float = 0.001):
        """Train the emotion parsing component"""
        print("Training Emotion Parser...")
        
        optimizer = optim.Adam(self.model.emotion_parser.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Emotion label mapping
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.model.emotion_parser.emotion_labels)}
        
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            self.model.emotion_parser.train()
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # Get emotion predictions
                emotion_weights = []
                true_emotions = []
                
                for prompt, emotion in zip(batch['prompt'], batch['emotion']):
                    pred_weights = self.model.emotion_parser.forward(prompt)
                    emotion_weights.append(pred_weights)
                    true_emotions.append(emotion_to_idx[emotion])
                
                emotion_weights = torch.stack(emotion_weights)
                true_emotions = torch.tensor(true_emotions, device=self.device)
                
                # Calculate loss
                loss = criterion(emotion_weights, true_emotions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(emotion_weights, dim=1)
                correct_predictions += (predictions == true_emotions).sum().item()
                total_predictions += len(true_emotions)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['emotion_accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def train_mode_blender(self, dataloader: DataLoader, epochs: int = 10, lr: float = 0.001):
        """Train the mode blending component"""
        print("Training Mode Blender...")
        
        optimizer = optim.Adam(self.model.mode_blender.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Mode label mapping
        mode_to_idx = {mode: idx for idx, mode in enumerate(self.model.mode_blender.mode_labels)}
        
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            self.model.mode_blender.train()
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # Get emotion weights and predict modes
                mode_predictions = []
                true_modes = []
                
                for prompt, mode in zip(batch['prompt'], batch['mode']):
                    emotion_weights = self.model.emotion_parser.parse_emotion_weights(prompt)
                    emotion_tensor = torch.tensor([emotion_weights[emotion] for emotion in 
                                                 self.model.emotion_parser.emotion_labels], device=self.device)
                    
                    mode_pred = self.model.mode_blender.forward(emotion_tensor.unsqueeze(0))
                    mode_predictions.append(mode_pred.squeeze())
                    true_modes.append(mode_to_idx[mode])
                
                mode_predictions = torch.stack(mode_predictions)
                true_modes = torch.tensor(true_modes, device=self.device)
                
                # Calculate loss
                loss = criterion(mode_predictions, true_modes)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(mode_predictions, dim=1)
                correct_predictions += (predictions == true_modes).sum().item()
                total_predictions += len(true_modes)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            self.training_history['mode_accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Mode Accuracy = {accuracy:.4f}")
    
    def train_progression_generator(self, dataloader: DataLoader, epochs: int = 15, lr: float = 0.0005):
        """Train the chord progression generation component"""
        print("Training Progression Generator...")
        
        optimizer = optim.Adam(self.model.progression_generator.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.model.progression_generator.chord_vocab.get('PAD', 0))
        
        for epoch in range(epochs):
            total_loss = 0
            
            self.model.progression_generator.train()
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                batch_loss = 0
                
                for prompt, chords, genres in zip(batch['prompt'], batch['chords'], batch['genres']):
                    # Get context
                    emotion_weights = self.model.emotion_parser.parse_emotion_weights(prompt)
                    primary_mode, mode_blend = self.model.mode_blender.get_primary_mode(emotion_weights)
                    
                    # Convert to indices (simplified)
                    mode_idx = self.model.mode_blender.mode_labels.index(primary_mode)
                    genre_idx = 0  # Simplified - would need proper genre encoding
                    
                    mode_context = torch.tensor([mode_idx], device=self.device)
                    genre_context = torch.tensor([genre_idx], device=self.device)
                    
                    # Convert chord sequence to indices
                    chord_indices = []
                    chord_indices.append(self.model.progression_generator.chord_vocab['START'])
                    for chord in chords:
                        if chord in self.model.progression_generator.chord_vocab:
                            chord_indices.append(self.model.progression_generator.chord_vocab[chord])
                        else:
                            chord_indices.append(self.model.progression_generator.chord_vocab['PAD'])
                    chord_indices.append(self.model.progression_generator.chord_vocab['END'])
                    
                    # Pad sequence
                    max_len = 16
                    while len(chord_indices) < max_len:
                        chord_indices.append(self.model.progression_generator.chord_vocab['PAD'])
                    
                    chord_tensor = torch.tensor([chord_indices[:max_len]], device=self.device)
                    
                    # Forward pass
                    predictions = self.model.progression_generator(mode_context, genre_context, chord_tensor[:, :-1])
                    targets = chord_tensor[:, 1:]
                    
                    # Calculate loss
                    loss = criterion(predictions.view(-1, predictions.size(-1)), targets.view(-1))
                    batch_loss += loss
                
                batch_loss = batch_loss / len(batch['prompt'])
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Generation Loss = {avg_loss:.4f}")
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the complete model"""
        self.model.eval()
        
        emotion_correct = 0
        mode_correct = 0
        total_samples = 0
        
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.model.emotion_parser.emotion_labels)}
        mode_to_idx = {mode: idx for idx, mode in enumerate(self.model.mode_blender.mode_labels)}
        
        with torch.no_grad():
            for batch in test_dataloader:
                for prompt, true_emotion, true_mode in zip(batch['prompt'], batch['emotion'], batch['mode']):
                    # Get predictions
                    emotion_weights = self.model.emotion_parser.parse_emotion_weights(prompt)
                    predicted_emotion = max(emotion_weights, key=emotion_weights.get)
                    
                    primary_mode, _ = self.model.mode_blender.get_primary_mode(emotion_weights)
                    
                    # Check accuracy
                    if predicted_emotion == true_emotion:
                        emotion_correct += 1
                    
                    if primary_mode == true_mode:
                        mode_correct += 1
                    
                    total_samples += 1
        
        return {
            'emotion_accuracy': emotion_correct / total_samples,
            'mode_accuracy': mode_correct / total_samples,
            'total_samples': total_samples
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if self.training_history['loss']:
            axes[0].plot(self.training_history['loss'])
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
        
        if self.training_history['emotion_accuracy']:
            axes[1].plot(self.training_history['emotion_accuracy'])
            axes[1].set_title('Emotion Classification Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
        
        if self.training_history['mode_accuracy']:
            axes[2].plot(self.training_history['mode_accuracy'])
            axes[2].set_title('Mode Prediction Accuracy')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_train_test_split(dataset: ChordProgressionDataset, test_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and test sets"""
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def main():
    """Main training pipeline"""
    print("Initializing Chord Progression Model Training...")
    
    # Configuration
    config = {
        'database_path': 'emotion_progression_database.json',
        'batch_size': 8,
        'learning_rate': 0.001,
        'epochs_emotion': 10,
        'epochs_mode': 10,
        'epochs_generation': 15,
        'test_ratio': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ChordProgressionDataset(config['database_path'], augment_data=True)
    print(f"Total examples: {len(dataset)}")
    
    # Create train/test split
    train_dataset, test_dataset = create_train_test_split(dataset, config['test_ratio'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Initialize model
    model = ChordProgressionModel()
    trainer = ModelTrainer(model, device=config['device'])
    
    # Training pipeline
    print("\n=== Starting Training Pipeline ===")
    
    # 1. Train emotion parser
    trainer.train_emotion_parser(
        train_dataloader, 
        epochs=config['epochs_emotion'], 
        lr=config['learning_rate']
    )
    
    # 2. Train mode blender
    trainer.train_mode_blender(
        train_dataloader,
        epochs=config['epochs_mode'],
        lr=config['learning_rate']
    )
    
    # 3. Train progression generator
    trainer.train_progression_generator(
        train_dataloader,
        epochs=config['epochs_generation'],
        lr=config['learning_rate'] * 0.5  # Lower LR for generation
    )
    
    # Evaluation
    print("\n=== Evaluating Model ===")
    eval_results = trainer.evaluate(test_dataloader)
    print(f"Test Results:")
    print(f"  Emotion Accuracy: {eval_results['emotion_accuracy']:.4f}")
    print(f"  Mode Accuracy: {eval_results['mode_accuracy']:.4f}")
    print(f"  Total Samples: {eval_results['total_samples']}")
    
    # Save model
    trainer.save_model('chord_progression_model.pth')
    
    # Plot training history
    trainer.plot_training_history()
    
    # Demo generation
    print("\n=== Demo Generation ===")
    test_prompts = [
        "I'm feeling really happy and excited",
        "Something sad but beautiful",
        "Dark and mysterious vibes",
        "Romantic but a little anxious",
        "Angry and aggressive energy"
    ]
    
    model.eval()
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        results = model.generate_from_prompt(prompt, genre_preference="Pop", num_progressions=1)
        result = results[0]
        print(f"  Primary emotion: {max(result['emotion_weights'], key=result['emotion_weights'].get)}")
        print(f"  Mode: {result['primary_mode']}")
        print(f"  Progression: {' - '.join(result['chords'])}")


if __name__ == "__main__":
    main()
