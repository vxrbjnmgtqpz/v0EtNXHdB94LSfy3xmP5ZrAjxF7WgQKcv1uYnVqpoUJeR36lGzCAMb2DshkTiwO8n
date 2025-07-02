Great — I’ll now build a new JSON database that maps each of the additional sub-emotions to:

- One or more musically valid chord progressions
- The mode or blended mode that best expresses the emotion
- A brief explanation of the musical-emotional connection
- Genre suggestions when applicable

This will expand the original palette and give your `.jam` system deeper expressive capability. I’ll let you know as soon as it’s ready.

```json
{
  "Joy": {
    "Excitement": {
      "mode": "Ionian (Major) with secondary dominant (II7)",
      "progressions": ["I – II7 – V – I"],
      "explanation": "Using the major scale’s supertonic as a dominant (II7 chord) before the V creates an expectant, giddy build-up:contentReference[oaicite:0]{index=0} fitting for excitement.",
      "genres": ["Pop", "Gospel", "Rock"]
    },
    "Contentment": {
      "mode": "Ionian (Major)",
      "progressions": ["I – IV – V – IV"],
      "explanation": "A simple loop of major chords avoids strong cadences, producing a neutral, soothing atmosphere:contentReference[oaicite:1]{index=1} that mirrors contentment.",
      "genres": ["Ambient", "New Age", "Classical"]
    },
    "Pride": {
      "mode": "Ionian (Major)",
      "progressions": ["ii – V – I"],
      "explanation": "A classic ii–V–I cadence (borrowed from jazz) offers a satisfying resolution:contentReference[oaicite:2]{index=2}, conveying confidence and triumph appropriate for pride.",
      "genres": ["Jazz", "Big Band", "Anthemic"]
    },
    "Romantic Longing": {
      "mode": "Ionian (Major) with modal mixture",
      "progressions": ["I – iv"],
      "explanation": "Shifting from a major tonic to a minor iv chord introduces a “colorful, bittersweet” sense of longing:contentReference[oaicite:3]{index=3}, famously used in love themes (e.g. Princess Leia’s Theme).",
      "genres": ["Soundtrack", "Romantic", "Classical"]
    }
  },
  "Sadness": {
    "Melancholy": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – VI – III – VII"],
      "explanation": "A rise from minor i to major III and VII before descending back creates a melancholic, bittersweet sound:contentReference[oaicite:4]{index=4}.",
      "genres": ["Ballad", "Classical"]
    },
    "Longing": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – III – VII – iv"],
      "explanation": "Alternating minor and major chords yields a bittersweet, yearning effect:contentReference[oaicite:5]{index=5}, perfectly capturing a sense of longing.",
      "genres": ["Ballad", "Pop"]
    },
    "Nostalgia": {
      "mode": "Ionian (Major)",
      "progressions": ["I – vi – IV – V"],
      "explanation": "The “50s progression” (I–vi–IV–V) evokes a classic sadness and nostalgia reminiscent of mid-century pop ballads:contentReference[oaicite:6]{index=6}.",
      "genres": ["50s Pop", "Doo-wop", "Oldies"]
    },
    "Despair": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["VI – iv – i – v"],
      "explanation": "A somber minor progression like VI–iv–i–v sustains a haunting, reflective mood:contentReference[oaicite:7]{index=7}, expressing deep despair.",
      "genres": ["Dramatic Score", "Classical"]
    },
    "Loneliness": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – VII – VI – VII"],
      "explanation": "A descending minor sequence (i–VII–VI–VII) creates a cyclical feeling of melancholy and isolation:contentReference[oaicite:8]{index=8}, echoing loneliness.",
      "genres": ["Alternative", "Ambient"]
    }
  },
  "Anger": {
    "Rage": {
      "mode": "Phrygian",
      "progressions": ["i – ♭II – i"],
      "explanation": "Phrygian mode’s signature ♭II chord delivers an extra dark, aggressive sound:contentReference[oaicite:9]{index=9} suited for unbridled rage.",
      "genres": ["Metal", "Industrial", "Orchestral"]
    },
    "Resentment": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – VI – i – VII"],
      "explanation": "A brooding minor pattern that never fully resolves lends a haunting, introspective edge:contentReference[oaicite:10]{index=10}, matching the simmering nature of resentment.",
      "genres": ["Alternative Rock", "Cinematic"]
    },
    "Hatred": {
      "mode": "Double Harmonic (Byzantine)",
      "progressions": ["i – ♭II – i"],
      "explanation": "The Double Harmonic scale (with its two augmented seconds) is considered one of the darkest musical modes:contentReference[oaicite:11]{index=11}. Its sharp, exotic intervals intensify a feeling of deep hatred.",
      "genres": ["Horror Score", "World"]
    }
  },
  "Fear": {
    "Anxiety": {
      "mode": "Locrian / Diminished",
      "progressions": ["iº – iiº – iº …"],
      "explanation": "Revolving through diminished chords and tense intervals amplifies an unresolved tension, keeping the listener on edge:contentReference[oaicite:12]{index=12}—much like anxiety.",
      "genres": ["Horror Score", "Ambient"]
    },
    "Dread": {
      "mode": "Harmonic Minor",
      "progressions": ["i – iv – ♭II – V7 – i"],
      "explanation": "The harmonic minor scale’s augmented second adds dramatic, uneasy gravity:contentReference[oaicite:13]{index=13}, fitting the slow-burning, ominous tension of dread.",
      "genres": ["Horror", "Classical"]
    },
    "Suspense": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – VI – iv – V"],
      "explanation": "Combining minor and major chords without resolving (e.g. i–VI–iv–V) sustains tension and keeps the listener on edge:contentReference[oaicite:14]{index=14}, perfect for suspenseful scenes.",
      "genres": ["Thriller Score", "Cinematic"]
    }
  },
  "Surprise": {
    "Shock": {
      "mode": "N/A (Chromatic surprise)",
      "progressions": ["I – ♭II – I"],
      "explanation": "A sudden dissonant chord (like a diminished or Neapolitan ♭II) delivers a jarring shock, exploiting the fear and spookiness of diminished harmony:contentReference[oaicite:15]{index=15}.",
      "genres": ["Horror", "Experimental"]
    },
    "Amazement": {
      "mode": "Lydian",
      "progressions": ["I – II"],
      "explanation": "Lydian mode (major scale with a raised 4th) evokes a bright, ethereal sense of wonder and energy:contentReference[oaicite:16]{index=16}—ideal for conveying amazement or awe.",
      "genres": ["Fantasy Score", "Soundtrack"]
    },
    "Confusion": {
      "mode": "Polytonal / Atonal",
      "progressions": ["I – ♭III – ♭VI – ♭II"],
      "explanation": "Rapid chromatic shifts and ambiguous tonal centers can induce confusion. Borrowing unexpected chords “plays on expectations” and puts the listener in a weird, disoriented place:contentReference[oaicite:17]{index=17}.",
      "genres": ["Avant-Garde", "Experimental"]
    }
  },
  "Disgust": {
    "Revulsion": {
      "mode": "Chromatic",
      "progressions": ["i – ♭v° – i"],
      "explanation": "Clashing dissonances (e.g. an augmented fourth or cluster of close notes) emphasize an unsettling, nauseated tone, capturing the feeling of revulsion.",
      "genres": ["Horror", "Avant-Garde"]
    },
    "Loathing": {
      "mode": "Phrygian Dominant",
      "progressions": ["i – ♭II – ♭VII – i"],
      "explanation": "Extreme loathing combines anger and disgust. The Phrygian dominant mode (with both ♭2 and ♭7 scale tones) provides an intensely dark, scornful sound to match this emotion.",
      "genres": ["Metal", "Symphonic"]
    }
  },
  "Contempt": {
    "Scorn": {
      "mode": "Major with modal interchange",
      "progressions": ["I – i"],
      "explanation": "Switching abruptly from a bright major chord to its parallel minor injects a biting sourness into the music, as if the harmony itself is \"sneering\"—a fitting analog for scorn or disdain.",
      "genres": ["Satire", "Film Score"]
    }
  },
  "Self-Hostility": {
    "Self-Loathing": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – iv – i°"],
      "explanation": "A despairing minor progression infused with dissonance (e.g. a diminished tonic chord) underscores the inward-directed anger of self-loathing. The lack of resolution mirrors a tortured inner state.",
      "genres": ["Dramatic Score", "Dark Ambient"]
    },
    "Inner Conflict": {
      "mode": "Mixed (Major/Minor)",
      "progressions": ["I – i – I – i"],
      "explanation": "Frequent shifts between major and minor versions of the tonic (I to i and back) convey instability. This unsettled oscillation reflects the turmoil of inner conflict against oneself.",
      "genres": ["Contemporary Classical", "Experimental"]
    }
  },
  "Shame": {
    "Embarrassment": {
      "mode": "Major with chromatic passing",
      "progressions": ["I – V/VII – I"],
      "explanation": "A mostly light major tonality interrupted by an odd chord (like a quick diminished passing chord or odd modulation) can musically emulate the jolt of embarrassment, often used in comedic scoring.",
      "genres": ["Comedy", "Cartoon"]
    },
    "Humiliation": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["i – ♭II – V7 – i"],
      "explanation": "Slow, heavy minor progressions with dramatic chromatic chords (such as a Neapolitan ♭II leading into V7) heighten the feeling of tragic shame. The weighty resolution back to i mirrors the gravity of humiliation.",
      "genres": ["Opera", "Symphonic"]
    }
  },
  "Shyness": {
    "Timidity": {
      "mode": "Major (soft voicings)",
      "progressions": ["I – ii^ø – I"],
      "explanation": "Soft, sparse harmonies with delicate dissonances can portray timidity. For instance, a tonic chord followed by a half-diminished supertonic (ii^ø) creates a gentle tension that quickly resolves, like a timid thought emerging then retreating.",
      "genres": ["Chamber Music", "Ambient"]
    },
    "Social Anxiety": {
      "mode": "Minor",
      "progressions": ["i – ii° – i"],
      "explanation": "A looping minor progression that oscillates with a diminished chord (i ↔ ii°) conveys nervous energy. The music never quite relaxes, reflecting the uneasy self-consciousness of social anxiety.",
      "genres": ["Minimalist", "Ambient"]
    }
  },
  "Guilt": {
    "Remorse": {
      "mode": "Minor",
      "progressions": ["i – iv – i"],
      "explanation": "A plaintive minor plagal motion (i–iv–i) resembles a musical sigh, conveying remorse and penitence (akin to an “amen” in a minor key). The minor iv adds a soulful, pleading quality.",
      "genres": ["Gospel", "Blues"]
    },
    "Regret": {
      "mode": "Aeolian (Natural Minor)",
      "progressions": ["VI – VII – i – v"],
      "explanation": "A minor progression with an initial rise (VI–VII–i) and unresolved end (ending on v) creates a push-and-pull between hope and disappointment:contentReference[oaicite:18]{index=18}, perfectly suited to the bittersweet pain of regret.",
      "genres": ["Country", "Singer-Songwriter"]
    }
  },
  "Interest": {
    "Curiosity": {
      "mode": "Mixolydian",
      "progressions": ["I – ♭VII – IV – I"],
      "explanation": "Mixolydian mode’s use of the ♭VII chord gives a slightly unexpected, exploratory color:contentReference[oaicite:19]{index=19}. This folky, open sound evokes a sense of playful curiosity and wonder.",
      "genres": ["Folk", "Children’s Music"]
    },
    "Fascination": {
      "mode": "Ionian (Major) with added tones",
      "progressions": ["Imaj7 – IVmaj7"],
      "explanation": "Lush major seventh chords (Imaj7, IVmaj7) provide a soft, luxurious sound:contentReference[oaicite:20]{index=20} that can capture the feeling of fascination. The gentle, jazzy quality of major7 harmony conveys dreamy captivation.",
      "genres": ["Jazz", "Lounge"]
    }
  }
}
```
