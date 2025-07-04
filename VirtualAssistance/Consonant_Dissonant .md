# Consonant / Dissonant Framework

## VirtualAssistance Music Generation System

### 🎵 **Core Concept**

Consonance and dissonance are fundamental properties of harmonic intervals and chords that determine their perceived stability, tension, and emotional character. This framework integrates consonant/dissonant qualities with the existing 22-emotion system to provide more nuanced and psychologically accurate chord generation.

---

## 🎯 **Theoretical Foundation**

### **Classical Consonance/Dissonance Hierarchy**

#### **Perfect Consonances (0.0 - 0.2)**

- **Unison (0.0)**: Perfect stability, no tension
- **Octave (0.1)**: Near-perfect stability, sense of completion
- **Perfect Fifth (0.2)**: Strong stability, foundational harmony

#### **Imperfect Consonances (0.2 - 0.4)**

- **Major Third (0.25)**: Warm, stable, bright
- **Minor Third (0.3)**: Warm, stable, darker
- **Major Sixth (0.35)**: Open, stable, expansive
- **Minor Sixth (0.4)**: Stable but with subtle tension

#### **Mild Dissonances (0.4 - 0.6)**

- **Perfect Fourth (0.45)**: Context-dependent, mildly tense
- **Major Second (0.5)**: Gentle tension, forward motion
- **Minor Seventh (0.55)**: Smooth tension, jazz character
- **Major Seventh (0.6)**: Sophisticated tension, modern harmony

#### **Strong Dissonances (0.6 - 0.8)**

- **Minor Second (0.7)**: Sharp tension, requires resolution
- **Tritone (0.75)**: Maximum traditional dissonance, "devil's interval"
- **Major Ninth (0.8)**: Complex tension, extended harmony

#### **Extreme Dissonances (0.8 - 1.0)**

- **Minor Ninth (0.85)**: Harsh, requires careful handling
- **Sharp Eleventh (0.9)**: Piercing, avant-garde quality
- **Cluster Harmonies (1.0)**: Maximum dissonance, experimental

---

## 🎼 **Chord Classification System**

### **Triadic Consonance/Dissonance**

#### **Consonant Chords (0.0 - 0.4)**

```
Major Triad (I, IV, V)           → 0.2   // Stable, foundational
Minor Triad (ii, iii, vi)        → 0.3   // Stable, darker character
Suspended Chords (sus2, sus4)    → 0.35  // Stable with subtle motion
```

#### **Moderately Dissonant Chords (0.4 - 0.6)**

```
Major 7th (maj7)                 → 0.45  // Sophisticated, jazzy
Minor 7th (m7)                   → 0.4   // Smooth, versatile
Dominant 7th (7)                 → 0.55  // Classic tension-resolution
Add9 Chords                      → 0.5   // Contemporary, open
```

#### **Highly Dissonant Chords (0.6 - 0.8)**

```
Diminished 7th (dim7)            → 0.75  // Dramatic tension
Augmented Chords (aug)           → 0.7   // Unstable, mysterious
Minor/Major 7th (mM7)            → 0.65  // Complex, unsettled
Altered Dominants (7alt)         → 0.8   // Jazz sophistication
```

#### **Extreme Dissonance Chords (0.8 - 1.0)**

```
Cluster Chords                   → 0.95  // Experimental, avant-garde
Poly-chords                      → 0.9   // Multiple tonal centers
Microtonal Harmonies             → 1.0   // Beyond traditional harmony
```

---

## 🎨 **Context-Dependent Consonance**

### **Genre-Specific Consonance Values**

#### **Classical Context**

- Traditional consonance hierarchy applies strictly
- Dissonance requires careful preparation and resolution
- Context weight: 1.0 (full traditional values)

#### **Jazz Context**

- Extended harmony is more consonant
- 7th chords are functionally consonant
- Context weight: 0.8 (reduced dissonance perception)

```
Minor 7th: 0.4 → 0.32 in jazz
Major 7th: 0.6 → 0.48 in jazz
```

#### **Blues Context**

- Dominant 7th chords are consonant
- Blue notes create acceptable dissonance
- Context weight: 0.7

```
Dominant 7th: 0.55 → 0.39 in blues
```

#### **Rock/Pop Context**

- Power chords (root + fifth) are highly consonant
- Some dissonance is stylistically expected
- Context weight: 0.9

#### **Experimental/Avant-garde Context**

- Dissonance is embraced and normalized
- Context weight: 0.5 (greatly reduced dissonance perception)

---

## 🧠 **Psychological Integration**

### **Emotion-Consonance Correlation Matrix**

#### **Consonant Emotions (0.0 - 0.4)**

```
Joy (0.2)           → Bright, stable harmony
Trust (0.25)        → Reliable, foundational chords
Love (0.3)          → Warm, enveloping harmony
Gratitude (0.2)     → Simple, pure harmony
Contentment (0.15)  → Stable, restful harmony
Belonging (0.3)     → Inclusive, welcoming harmony
```

#### **Moderately Dissonant Emotions (0.4 - 0.6)**

```
Anticipation (0.5)  → Forward-moving harmony
Surprise (0.45)     → Unexpected but manageable
Wonder (0.4)        → Sophisticated, complex beauty
Aesthetic Awe (0.4) → Sophisticated harmony
Arousal (0.5)       → Energizing tension
```

#### **Highly Dissonant Emotions (0.6 - 0.8)**

```
Sadness (0.6)       → Complex, unresolved harmony
Anger (0.75)        → Sharp, aggressive dissonance
Fear (0.7)          → Unsettling, unstable harmony
Anxiety (0.65)      → Nervous, tense harmony
Envy (0.7)          → Bitter, twisted harmony
Shame (0.6)         → Uncomfortable, hidden tension
```

#### **Extreme Dissonance Emotions (0.8 - 1.0)**

```
Malice (0.9)        → Deliberately harsh, destructive
Disgust (0.85)      → Repulsive, rejecting harmony
Guilt (0.8)         → Self-attacking dissonance
Dissociation (0.9)  → Disconnected, alien harmony
```

---

## 🔢 **Implementation Schema**

### **Individual Chord Database Extension**

```json
{
  "chord": "I",
  "symbol": "C",
  "mode_context": "Ionian",
  "style_context": "Classical",
  "emotion_weights": { ... },
  "consonant_dissonant_profile": {
    "base_value": 0.2,
    "context_modifiers": {
      "Classical": 1.0,
      "Jazz": 0.8,
      "Blues": 0.9,
      "Rock": 0.9,
      "Experimental": 0.5
    },
    "emotional_resonance": {
      "Joy": 0.9,
      "Trust": 0.8,
      "Love": 0.7
    },
    "description": "Perfect consonance, foundational stability"
  }
}
```

### **Interpolation Integration**

```python
@dataclass
class EnhancedEmotionState:
    emotion_weights: Dict[str, float]
    consonant_dissonant_value: float
    consonant_dissonant_trajectory: str  # "towards_consonance", "towards_dissonance", "stable"
    primary_emotion: str
    mode: str
    style_context: str
```

### **Selection Algorithm**

```python
def calculate_chord_fitness(chord, emotion_weights, consonant_dissonant_preference):
    # Base emotional fitness
    emotional_score = sum(emotion_weights[e] * chord.emotion_weights[e]
                         for e in emotion_weights)

    # Consonant/dissonant fitness
    cd_preference = consonant_dissonant_preference
    cd_value = chord.consonant_dissonant_profile.base_value
    cd_score = 1.0 - abs(cd_preference - cd_value)

    # Emotional resonance with consonance/dissonance
    resonance_score = sum(chord.consonant_dissonant_profile.emotional_resonance.get(e, 0.5) * w
                         for e, w in emotion_weights.items())

    return emotional_score * 0.4 + cd_score * 0.4 + resonance_score * 0.2
```

---

## 🎭 **Advanced Applications**

### **Consonant/Dissonant Interpolation**

#### **Tension Curves**

```python
def create_tension_curve(start_cd, end_cd, steps, curve_type):
    """
    Create consonant/dissonant tension curves

    Curve types:
    - "linear": Direct interpolation
    - "build": Gradual increase in tension
    - "release": Gradual decrease in tension
    - "peak": Build to maximum tension then release
    - "valley": Release to minimum tension then build
    """
```

#### **Emotional Journey Mapping**

```python
def map_emotional_journey_to_consonance(emotion_trajectory):
    """
    Map emotional journey to consonant/dissonant trajectory

    Example:
    Joy → Sadness → Anger → Resolution
    0.2 → 0.6 → 0.75 → 0.2
    """
```

### **Context-Aware Generation**

#### **Style-Specific Consonance**

- **Classical**: Strict consonance hierarchy
- **Jazz**: Extended harmony normalization
- **Blues**: Dominant 7th normalization
- **Rock**: Power chord optimization
- **Experimental**: Dissonance acceptance

#### **Modal Consonance Variations**

- **Ionian**: Traditional consonance values
- **Dorian**: Slightly increased acceptance of modal dissonance
- **Phrygian**: Normalization of b2 intervals
- **Lydian**: Acceptance of #4 intervals
- **Mixolydian**: Normalization of b7 intervals
- **Aeolian**: Acceptance of minor-specific dissonances
- **Locrian**: Extreme dissonance normalization

---

## 🎯 **Implementation Priorities**

### **Phase 1: Foundation**

1. Add consonant/dissonant values to individual chord database
2. Implement basic chord selection with CD preference
3. Test with existing emotion system

### **Phase 2: Integration**

1. Extend interpolation engine with CD support
2. Add context-aware CD values
3. Implement emotional resonance calculations

### **Phase 3: Advanced Features**

1. Add tension curve generation
2. Implement style-specific CD modifications
3. Add modal CD variations

### **Phase 4: Neural Integration**

1. Update neural analyzer with CD features
2. Retrain models with CD data
3. Implement contextual CD predictions

---

## 📊 **Validation Metrics**

### **Theoretical Validation**

- Consonance values align with music theory
- Context modifications are musically logical
- Emotional correlations are psychologically sound

### **Practical Validation**

- Generated chords match expected CD characteristics
- Interpolation creates musically coherent transitions
- Style contexts produce appropriate CD modifications

### **User Experience Validation**

- Consonant preferences produce stable, pleasant harmony
- Dissonant preferences create appropriate tension
- Emotional descriptions align with CD characteristics

---

## 🔮 **Future Enhancements**

### **Micro-Consonance**

- Sub-chord consonance analysis
- Voice leading consonance optimization
- Inversion-specific consonance values

### **Dynamic Consonance**

- Time-dependent consonance evolution
- Rhythmic consonance patterns
- Harmonic rhythm integration

### **Cultural Consonance**

- Culture-specific consonance norms
- Historical period consonance variations
- Regional harmonic preferences

---

_This framework provides the theoretical foundation and practical implementation guide for integrating consonant/dissonant qualities into the VirtualAssistance music generation system while maintaining compatibility with the existing 22-emotion architecture._
