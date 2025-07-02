# Chord Progression Generation from Natural Language

A PyTorch-based system that generates chord progressions from natural language prompts using emotion-to-music mapping. Based on 12 core human emotions mapped to musical modes with genre-weighted progressions.

## Features

🎯 **Natural Language Input**: Generate chord progressions from descriptions like "romantic but anxious" or "dark and mysterious"

🧠 **Emotion Analysis**: BERT-based emotion parser that extracts weighted emotional content from text

🎼 **Musical Intelligence**: Maps emotions to musical modes using music theory principles

🎸 **Genre Awareness**: 50+ genre styles with weighted progression selection

🎵 **MIDI Export**: Convert generated progressions to MIDI files with full arrangements

📊 **Training Pipeline**: Complete PyTorch training system with custom datasets

## Architecture

```
Natural Language → Emotion Parser → Mode Blender → Progression Generator → MIDI Output
     (BERT)          (Neural Net)     (Neural Net)      (Transformer)       (Music21)
```

### Core Components

1. **Emotion Parser**: BERT-based text encoder that maps natural language to 12 core emotions
2. **Mode Blender**: Neural network that converts emotion weights to musical mode preferences
3. **Progression Generator**: Transformer decoder that creates chord sequences from modal context
4. **Database System**: 144 curated chord progressions (12 per emotion) with genre weights

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd chord-progression-generator

# Install dependencies
pip install -r requirements.txt

# Download BERT model (automatic on first run)
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
```

## Quick Start

### Basic Usage

```python
from chord_progression_model import ChordProgressionModel

# Initialize model
model = ChordProgressionModel()

# Generate chord progression
results = model.generate_from_prompt(
    "romantic but a little anxious", 
    genre_preference="Jazz", 
    num_progressions=2
)

for result in results:
    print(f"Progression: {' → '.join(result['chords'])}")
    print(f"Primary emotion: {max(result['emotion_weights'], key=result['emotion_weights'].get)}")
    print(f"Mode: {result['primary_mode']}")
```

### Interactive Demo

```bash
python demo.py
```

Features:
- Interactive CLI for real-time generation
- Emotion analysis visualization
- Genre comparison mode
- Batch processing
- Export to .jam session files

### MIDI Generation

```python
from midi_generator import MIDIGenerator

midi_gen = MIDIGenerator(model)

# Generate MIDI file
filename, result = midi_gen.generate_midi_from_prompt(
    "epic and triumphant",
    genre="Orchestral",
    key="C",
    tempo=120
)

print(f"MIDI saved as: {filename}")
```

## Training

Train your own model with custom data:

```bash
python train_model.py
```

The training pipeline includes:
- Emotion classification training
- Mode prediction training  
- Chord sequence generation training
- Evaluation and visualization

### Training Configuration

```python
config = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'epochs_emotion': 10,
    'epochs_mode': 10, 
    'epochs_generation': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

## Database Structure

The system uses a comprehensive JSON database with:

### 12 Core Emotions → Musical Modes

| Emotion | Mode | Characteristics |
|---------|------|----------------|
| Joy | Ionian | Bright, resolved, stable |
| Sadness | Aeolian | Melancholy, reflective |
| Fear | Phrygian | Dark, tense, claustrophobic |
| Anger | Phrygian Dominant | Aggressive, exotic tension |
| Disgust | Locrian | Unstable, dissonant |
| Surprise | Lydian | Bright, floating, curious |
| Trust | Dorian | Warm, grounded, hopeful |
| Anticipation | Melodic Minor | Forward motion, yearning |
| Shame | Harmonic Minor | Tragic, haunting |
| Love | Mixolydian | Nostalgic, soulful |
| Envy | Hungarian Minor | Bitter elegance, intense |
| Aesthetic Awe | Lydian Augmented | Transcendent, sublime |

### Genre Weights

Each progression includes genre compatibility scores:

```json
{
  "chords": ["I", "V", "vi", "IV"],
  "genres": { 
    "Pop": 1.0, 
    "Rock": 0.8, 
    "Folk": 0.6 
  }
}
```

## API Reference

### ChordProgressionModel

Main interface for the generation system.

#### Methods

- `generate_from_prompt(text, genre, num_progressions)`: Generate progressions from text
- `export_jam_session(results, session_id)`: Export to .jam format

### EmotionParser

BERT-based emotion analysis.

#### Methods

- `parse_emotion_weights(text)`: Extract emotion weights from text
- `forward(text_input)`: Neural network forward pass

### ModeBlender

Convert emotions to musical modes.

#### Methods

- `get_primary_mode(emotion_weights)`: Get dominant mode and blend
- `forward(emotion_weights)`: Neural network forward pass

### ChordProgressionGenerator

Generate chord sequences.

#### Methods

- `forward(mode_context, genre_context, target_chords)`: Training forward pass
- `_generate_sequence(context, max_length)`: Autoregressive generation

## File Structure

```
├── chord_progression_model.py    # Main model architecture
├── emotion_progression_database.json  # Complete emotion→progression mapping
├── train_model.py               # Training pipeline
├── demo.py                      # Interactive demonstration
├── midi_generator.py            # MIDI export functionality
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Example Outputs

### Emotion Analysis
```
Input: "romantic but a little anxious"
Emotions: Love(0.65) Fear(0.35)
Primary Mode: Mixolydian
Progression: I → ♭VII → ii → V
```

### Genre Comparison
```
Prompt: "dark and mysterious"
Pop:    i → VI → III → VII
Jazz:   i → ii° → V7 → i  
Metal:  i → ♭II → ♭VII → i
```

## Advanced Features

### Custom Training Data

Add your own progressions to the database:

```json
{
  "emotion": "Custom_Emotion",
  "mode": "Custom_Mode", 
  "progression_pool": [
    {
      "chords": ["custom", "progression"],
      "genres": {"Genre": 1.0}
    }
  ]
}
```

### Real-time Mode Blending

The system supports emotion blending:

```python
# Input: "happy but nostalgic"
emotion_weights = {"Joy": 0.6, "Love": 0.4}
mode_blend = {"Ionian": 0.6, "Mixolydian": 0.4}
# Result: Major progressions with ♭VII borrowed chords
```

### Voice Leading (Future)

Planned features:
- Smooth voice leading between chords
- Inversion preferences
- Register-aware arrangements
- Rhythmic patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

Areas for contribution:
- Additional musical modes
- More genre styles  
- Rhythm generation
- Melody generation
- Voice leading algorithms

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work in research, please cite:

```bibtex
@software{chord_progression_generator,
  title={Chord Progression Generation from Natural Language},
  author={Your Name},
  year={2025},
  url={https://github.com/username/chord-progression-generator}
}
```

## Acknowledgments

- Music theory research from Berklee College of Music
- Emotion psychology from Plutchik's Wheel of Emotions
- BERT implementation from Hugging Face Transformers
- MIDI handling via mido and pretty_midi libraries

---

*Generate music that matches your emotions. From "happy and excited" to "dark and mysterious", let AI compose the perfect chord progressions for your creative projects.*
