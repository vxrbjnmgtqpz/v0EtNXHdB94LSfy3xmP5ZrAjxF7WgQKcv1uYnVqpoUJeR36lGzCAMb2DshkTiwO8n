"""
MIDI Generation Extension for Chord Progression Model
Converts generated chord progressions to MIDI files
"""

import torch
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
import pretty_midi
from typing import List, Dict, Tuple, Optional
import random
from datetime import datetime
from chord_progression_model import ChordProgressionModel


class ChordToMIDI:
    """Convert chord progressions to MIDI files"""
    
    def __init__(self):
        # Note mappings for different keys
        self.note_mapping = {
            'C': {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71},
            'G': {'G': 67, 'A': 69, 'B': 71, 'C': 72, 'D': 74, 'E': 76, 'F#': 78},
            'F': {'F': 65, 'G': 67, 'A': 69, 'Bb': 70, 'C': 72, 'D': 74, 'E': 76},
            'D': {'D': 62, 'E': 64, 'F#': 66, 'G': 67, 'A': 69, 'B': 71, 'C#': 73},
            'A': {'A': 69, 'B': 71, 'C#': 73, 'D': 74, 'E': 76, 'F#': 78, 'G#': 80},
            'E': {'E': 64, 'F#': 66, 'G#': 68, 'A': 69, 'B': 71, 'C#': 73, 'D#': 75},
            'B': {'B': 71, 'C#': 73, 'D#': 75, 'E': 76, 'F#': 78, 'G#': 80, 'A#': 82}
        }
        
        # Chord formulas (intervals from root)
        self.chord_formulas = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            '7': [0, 4, 7, 10],
            'dim7': [0, 3, 6, 9],
            'ø7': [0, 3, 6, 10]  # half-diminished
        }
    
    def roman_to_midi_notes(self, roman_chord: str, key: str = 'C', octave: int = 4) -> List[int]:
        """Convert Roman numeral chord to MIDI note numbers"""
        # Parse Roman numeral
        chord_info = self._parse_roman_numeral(roman_chord)
        
        # Get scale degrees for the key
        scale_notes = self._get_scale_notes(key)
        
        # Get root note
        root_note = scale_notes[chord_info['degree'] - 1]
        root_midi = self._note_to_midi(root_note, octave)
        
        # Get chord formula
        formula = self.chord_formulas.get(chord_info['quality'], [0, 4, 7])
        
        # Build chord
        chord_notes = [root_midi + interval for interval in formula]
        
        return chord_notes
    
    def _parse_roman_numeral(self, roman: str) -> Dict[str, any]:
        """Parse Roman numeral chord notation"""
        # Remove special characters and determine quality
        clean_roman = roman.replace('♭', 'b').replace('♯', '#')
        
        # Determine chord quality and degree
        if clean_roman.lower().endswith('°') or clean_roman.lower().endswith('dim'):
            quality = 'dim'
            degree_part = clean_roman.replace('°', '').replace('dim', '')
        elif clean_roman.endswith('+'):
            quality = 'aug'
            degree_part = clean_roman.replace('+', '')
        elif clean_roman.lower().endswith('ø') or 'ø' in clean_roman:
            quality = 'ø7'
            degree_part = clean_roman.replace('ø', '').replace('7', '')
        elif '7' in clean_roman:
            if clean_roman[0].isupper():
                quality = '7'  # dominant 7th
            else:
                quality = 'min7'
            degree_part = clean_roman.replace('7', '')
        elif clean_roman[0].isupper():
            quality = 'major'
            degree_part = clean_roman
        else:
            quality = 'minor'
            degree_part = clean_roman
        
        # Parse degree number
        degree_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
                      'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7}
        
        # Handle accidentals
        accidental = ''
        if degree_part.startswith('b'):
            accidental = 'b'
            degree_part = degree_part[1:]
        elif degree_part.startswith('#'):
            accidental = '#'
            degree_part = degree_part[1:]
        
        degree = degree_map.get(degree_part.upper(), 1)
        
        return {
            'degree': degree,
            'quality': quality,
            'accidental': accidental
        }
    
    def _get_scale_notes(self, key: str) -> List[str]:
        """Get scale notes for a given key"""
        major_scales = {
            'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
            'F': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
            'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
            'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
            'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
            'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#']
        }
        
        return major_scales.get(key, major_scales['C'])
    
    def _note_to_midi(self, note: str, octave: int) -> int:
        """Convert note name to MIDI number"""
        note_values = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                      'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                      'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
        
        return (octave + 1) * 12 + note_values.get(note, 0)
    
    def progression_to_midi(self, progression: List[str], key: str = 'C', 
                           tempo: int = 120, chord_duration: float = 1.0,
                           filename: str = None) -> str:
        """Convert chord progression to MIDI file"""
        if filename is None:
            filename = f"progression_{int(datetime.now().timestamp())}.mid"
        
        # Create MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        tempo_msg = mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo))
        track.append(tempo_msg)
        
        # Convert chord duration to ticks
        ticks_per_beat = mid.ticks_per_beat
        chord_ticks = int(chord_duration * ticks_per_beat)
        
        current_time = 0
        
        for chord_roman in progression:
            # Get MIDI notes for chord
            chord_notes = self.roman_to_midi_notes(chord_roman, key, octave=4)
            
            # Add note on messages
            for note in chord_notes:
                note_on = Message('note_on', channel=0, note=note, velocity=80, time=0)
                track.append(note_on)
            
            # Add note off messages after chord duration
            for i, note in enumerate(chord_notes):
                time_offset = chord_ticks if i == 0 else 0
                note_off = Message('note_off', channel=0, note=note, velocity=0, time=time_offset)
                track.append(note_off)
            
            current_time += chord_ticks
        
        # Save MIDI file
        mid.save(filename)
        return filename
    
    def create_arrangement(self, progression: List[str], key: str = 'C', 
                          tempo: int = 120, style: str = "basic") -> str:
        """Create a full arrangement with melody, bass, and percussion"""
        filename = f"arrangement_{int(datetime.now().timestamp())}.mid"
        
        # Create multi-track MIDI
        mid = MidiFile()
        
        # Chord track
        chord_track = MidiTrack()
        mid.tracks.append(chord_track)
        chord_track.append(mido.MetaMessage('track_name', name='Chords'))
        
        # Bass track  
        bass_track = MidiTrack()
        mid.tracks.append(bass_track)
        bass_track.append(mido.MetaMessage('track_name', name='Bass'))
        
        # Melody track
        melody_track = MidiTrack()
        mid.tracks.append(melody_track)
        melody_track.append(mido.MetaMessage('track_name', name='Melody'))
        
        # Set tempo for all tracks
        tempo_msg = mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo))
        chord_track.append(tempo_msg)
        
        ticks_per_beat = mid.ticks_per_beat
        chord_duration = 2 * ticks_per_beat  # 2 beats per chord
        
        for i, chord_roman in enumerate(progression):
            chord_notes = self.roman_to_midi_notes(chord_roman, key, octave=4)
            bass_note = chord_notes[0] - 12  # Bass an octave lower
            
            # Add chords
            for j, note in enumerate(chord_notes):
                time_offset = 0 if j > 0 else 0
                chord_track.append(Message('note_on', channel=0, note=note, velocity=60, time=time_offset))
            
            # Add bass
            bass_track.append(Message('note_on', channel=1, note=bass_note, velocity=80, time=0))
            
            # Add simple melody (highest chord tone with variations)
            melody_note = chord_notes[-1] + 12  # Melody an octave higher
            melody_track.append(Message('note_on', channel=2, note=melody_note, velocity=70, time=0))
            
            # Note offs
            for j, note in enumerate(chord_notes):
                time_offset = chord_duration if j == 0 else 0
                chord_track.append(Message('note_off', channel=0, note=note, velocity=0, time=time_offset))
            
            bass_track.append(Message('note_off', channel=1, note=bass_note, velocity=0, time=chord_duration))
            melody_track.append(Message('note_off', channel=2, note=melody_note, velocity=0, time=chord_duration))
        
        mid.save(filename)
        return filename


class MIDIGenerator:
    """High-level MIDI generation interface"""
    
    def __init__(self, model: ChordProgressionModel):
        self.model = model
        self.midi_converter = ChordToMIDI()
    
    def generate_midi_from_prompt(self, prompt: str, genre: str = "Pop", 
                                 key: str = "C", tempo: int = 120,
                                 style: str = "arrangement") -> str:
        """Generate MIDI file directly from natural language prompt"""
        # Generate chord progression
        results = self.model.generate_from_prompt(prompt, genre, 1)
        progression = results[0]['chords']
        
        # Convert to MIDI
        if style == "arrangement":
            filename = self.midi_converter.create_arrangement(progression, key, tempo)
        else:
            filename = self.midi_converter.progression_to_midi(progression, key, tempo)
        
        return filename, results[0]
    
    def batch_generate_midi(self, prompts: List[str], output_dir: str = "midi_output"):
        """Generate MIDI files for multiple prompts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        for i, prompt in enumerate(prompts):
            filename, result = self.generate_midi_from_prompt(prompt)
            
            # Move to output directory
            new_filename = os.path.join(output_dir, f"prompt_{i+1}_{filename}")
            os.rename(filename, new_filename)
            
            generated_files.append({
                'prompt': prompt,
                'filename': new_filename,
                'progression': result['chords'],
                'emotion': max(result['emotion_weights'], key=result['emotion_weights'].get),
                'mode': result['primary_mode']
            })
        
        return generated_files


# Example usage
if __name__ == "__main__":
    # Initialize model and MIDI generator
    model = ChordProgressionModel()
    midi_gen = MIDIGenerator(model)
    
    # Generate MIDI from prompts
    test_prompts = [
        "happy and uplifting",
        "sad but beautiful",
        "dark and mysterious",
        "romantic and warm",
        "energetic and exciting"
    ]
    
    print("Generating MIDI files from prompts...")
    files = midi_gen.batch_generate_midi(test_prompts)
    
    for file_info in files:
        print(f"'{file_info['prompt']}' → {file_info['filename']}")
        print(f"  Progression: {' → '.join(file_info['progression'])}")
        print(f"  Emotion: {file_info['emotion']} | Mode: {file_info['mode']}")
        print()
