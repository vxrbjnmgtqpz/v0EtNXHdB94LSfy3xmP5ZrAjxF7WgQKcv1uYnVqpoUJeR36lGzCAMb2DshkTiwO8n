// Comprehensive modal logic for the chord progression interface

// Circle of Fifths data structure for UI rendering
const circleOfFifths = {
  major: ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"],
  minor: [
    "Am",
    "Em",
    "Bm",
    "F#m",
    "C#m",
    "G#m",
    "D#m",
    "Bbm",
    "Fm",
    "Cm",
    "Gm",
    "Dm",
  ],
  steps: 12, // positions around the circle
  getRelativeMinor(majorKey) {
    const index = this.major.indexOf(majorKey);
    return this.minor[index] || null;
  },
  getRelativeMajor(minorKey) {
    const index = this.minor.indexOf(minorKey);
    return this.major[index] || null;
  },
};

const MODES = [
  "Ionian",
  "Dorian",
  "Phrygian",
  "Lydian",
  "Mixolydian",
  "Aeolian",
  "Locrian",
];

const KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// Mode-specific progressions with correct Roman numeral qualities
const MODE_PROGRESSIONS = {
  Ionian: [
    "I - IV - V - I",
    "I - V - vi - IV",
    "ii - V - I",
    "I7 - IV7 - V7 - I7",
    "ii7 - V7 - I7",
    "custom",
  ],
  Dorian: [
    "i - IV - v - i",
    "i - bVII - IV - i",
    "ii - v - i",
    "i7 - IV7 - v7 - i7",
    "ii7 - v7 - i7",
    "custom",
  ],
  Phrygian: [
    "i - bII - bVII - i",
    "i - bII - bIII - i",
    "bII - bIII - i",
    "i7 - bII7 - bVII7 - i7",
    "bII7 - bIII7 - i7",
    "custom",
  ],
  Lydian: [
    "I - II - V - I",
    "I - ii - V - I",
    "I - ii - vii - I",
    "I7 - II7 - V7 - I7",
    "I7 - ii7 - V7 - I7",
    "custom",
  ],
  Mixolydian: [
    "I - bVII - IV - I",
    "I - v - bVII - I",
    "I - IV - bVII - I",
    "I7 - bVII7 - IV7 - I7",
    "I7 - v7 - bVII7 - I7",
    "custom",
  ],
  Aeolian: [
    "i - bVI - bVII - i",
    "i - iv - v - i",
    "i - bVII - bVI - i",
    "i7 - bVI7 - bVII7 - i7",
    "i7 - iv7 - v7 - i7",
    "custom",
  ],
  Locrian: [
    "i - bII - bV - i",
    "i - bVI - bII - i",
    "bII - biii - i",
    "i7 - bII7 - bV7 - i7",
    "bII7 - biii7 - i7",
    "custom",
  ],
};

// Roman numeral mapping for each mode (1-indexed) - EXACT from old project
const MODE_ROMANS = {
  Ionian: ["", "I", "ii", "iii", "IV", "V", "vi", "vii"],
  Dorian: ["", "i", "ii", "bIII", "IV", "v", "vi", "bVII"],
  Phrygian: ["", "i", "bII", "bIII", "iv", "v", "bVI", "bvii"],
  Lydian: ["", "I", "II", "iii", "iv", "V", "vi", "vii"],
  Mixolydian: ["", "I", "ii", "iii", "IV", "v", "vi", "bVII"],
  Aeolian: ["", "i", "ii", "bIII", "iv", "v", "bVI", "bVII"],
  Locrian: ["", "i", "bII", "biii", "iv", "bV", "bVI", "bvii"],
};

const TIME_SIGNATURES = [
  "4/4",
  "3/4",
  "2/4",
  "6/8",
  "9/8",
  "12/8",
  "5/4",
  "7/8",
];

const BARS_PER_PHRASE = ["1", "2", "3", "4"];

// --- GROUPED PROGRESSIONS FOR ADVANCED MENU (matches Playwright test) ---
const MODE_PROGRESSION_GROUPS = {
  Ionian: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "I – IV – V – I",
        "ii – V – I",
        "I – V – vi – IV",
        "I – vi – ii – vii° (– I)",
        "I – IV – vii° – iii (– I)",
        "ii – vii° – I",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: ["IΔ7 – IVΔ7 – VΔ7 – IΔ7", "ii7 – VΔ7 – IΔ7", "ii7 – viiø7 – IΔ7"],
    },
    {
      name: "Ninths",
      type: "9th",
      items: [
        "IΔ9 – IVΔ9 – VΔ9 – IΔ9",
        "ii9 – VΔ9 – IΔ9",
        "IΔ9 – viiø7 – iii7 – IΔ9",
      ],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: [
        "I – I+ – vi – IV",
        "I – iii+ – vi – IV",
        "V+ – I",
        "bIII+ – IV – I",
      ],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: [
        "I – Vsus4 – V – I",
        "IV – Vsus4 – I",
        "I – IVsus2 – IV – V (– I)",
      ],
    },
  ],
  Dorian: [
    {
      name: "Triads",
      type: "triad",
      items: ["i – IV – v – i", "i – bVII – IV – i", "ii – v – i"],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "i7 – IVΔ7 – v7 – i7",
        "ii7 – v7 – i7",
        "i7 – bVII7 – ii7♭5 – i7",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: ["i9 – IVΔ9 – v9 – i9", "ii9 – v9 – i9"],
    },
    {
      name: "Borrowed/Chromatic",
      type: "augmented",
      items: ["i – IV – #iv° – V"],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["i – i+ – IV – v", "ii – IV+ – i", "v+ – i"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: [
        "i – Vsus4 – v – i",
        "i – IVsus2 – IV – i",
        "i – bVII – Vsus4 – v",
      ],
    },
  ],
  Phrygian: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "i – bII – bVII – i",
        "bII – bIII – i",
        "i – bII° – bVI – i",
        "bII – v° – i",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "i7 – bIIΔ7 – bVII7 – i7",
        "bIIΔ7 – bIII7 – i7",
        "i7 – iiø7 – bVIΔ7 – i7",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: ["i9 – bIIΔ9 – bVII9 – i9", "bIIΔ9 – bIII9 – i9"],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["bII – bIII+ – i", "i – bVI+ – bVII – i"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: ["i – bII – Vsus4 – i", "i – bVI – bII – Vsus2"],
    },
  ],
  Lydian: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "I – II – V – I",
        "I – ii – V – I",
        "I – IV – V – I",
        "I – II – vii° – I",
        "ii – vii° – I",
        "IV – vii° – I",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "IΔ7 – II7 – VΔ7 – IΔ7",
        "IΔ7 – ii7 – VΔ7 – IΔ7",
        "IΔ7 – IVΔ7 – VΔ7 – IΔ7",
        "ii7 – viiø7 – IΔ9",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: [
        "IΔ9 – II9 – VΔ9 – IΔ9",
        "IΔ9 – ii9 – VΔ9 – IΔ9",
        "IΔ9 – IVΔ9 – VΔ9 – IΔ9",
        "IΔ9 – viiø7 – iii7 – vi7",
      ],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["I – II+ – V – I", "I – IV+ – V – I", "I – I+ – vi – II"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: ["I – II – Vsus4 – V (– I)", "I – IVsus2 – IV – V"],
    },
  ],
  Mixolydian: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "I – bVII – IV – I",
        "I – IV – bVII – I",
        "I – v – bVII – I",
        "I – IV – vii° – V (– I)",
        "I – iii° – bVII – I",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "IΔ7 – bVII7 – IV7 – IΔ7",
        "IΔ7 – IVΔ7 – bVII7 – IΔ7",
        "IΔ7 – v7 – bVII7 – IΔ7",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: [
        "IΔ9 – bVII9 – IV9 – IΔ9",
        "IΔ9 – IVΔ9 – bVII9 – IΔ9",
        "IΔ9 – v9 – bVII9 – IΔ9",
      ],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["I – I+ – IV – I", "I – V+ – bVII – I"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: [
        "I – bVII – IVsus4 – IV (– I)",
        "I – V – IVsus2 – IV",
        "I – Vsus4 – V – I",
      ],
    },
  ],
  Aeolian: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "i – bVI – bVII – i",
        "i – iv – v – i",
        "i – bVII – bVI – i",
        "i – ii° – bVI – bVII",
        "i – iv – ii° – v",
        "ii° – v – i",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "i7 – bVI7 – bVII7 – i7",
        "i7 – iv7 – v7 – i7",
        "i7 – iiø7 – v7 – i7",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: [
        "i9 – bVI9 – bVII9 – i9",
        "i9 – iv9 – v9 – i9",
        "bVImaj7 – iiø7 – i9",
      ],
    },
    {
      name: "Diminished",
      type: "7th",
      items: ["i – bVI – ii° – V", "i7 – bVIΔ7 – ii7♭5 – V7"],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["i – i+ – iv – bVI", "i – III+ – bVI – bVII", "v+ – i"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: ["i – ivsus4 – iv – v (– i)", "i – bVI – Vsus4 – v"],
    },
  ],
  Locrian: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "i° – bII – bV – i°",
        "bII – biii – i",
        "i° – bII – bV",
        "bIII – i° – bVII",
        "v° – bII – i°",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: ["iø7 – bIIΔ7 – bV7", "bIIΔ7 – biii7 – iø7", "v°7 – bIII7 – i°7"],
    },
    {
      name: "Ninths",
      type: "9th",
      items: ["iø9 – bIIΔ9 – bV9", "bIIΔ9 – biii9 – iø9"],
    },
    {
      name: "Borrowed / Chromatic",
      type: "augmented",
      items: ["i° – bIII – bvii°", "bII – bVΔ7 – i°"],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["i° – bII+ – bV", "i – bIII+ – bVII"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: ["i° – bII – Vsus4 – i°", "bIII – bII – Vsus2 – bV"],
    },
  ],
  // Harmonic Minor Progressions
  HarmonicMinor: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "i – iv – V – i",
        "i – bVI – V – i",
        "ii° – V – i",
        "i – ii° – V – bVI (– i)",
        "iv – V – bVI – i",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "i(Δ7) – iv7 – V7 – i(Δ7)",
        "i(Δ7) – bVIΔ7 – V7 – i(Δ7)",
        "iiø7 – V7 – i(Δ7)",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: ["i(Δ9) – iv9 – V9 – i(Δ9)", "iiø9 – V9 – i(Δ9)"],
    },
    {
      name: "Diminished",
      type: "diminished",
      items: ["vii°7 – i – iv – V", "ii° – vii°7 – i – V"],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["i – bIII+ – iv – V", "i – bVI – bIII+ – i"],
    },
    {
      name: "Suspended",
      type: "suspended",
      items: ["i – Vsus4 – V – i", "iv – Vsus4 – V – i"],
    },
  ],
  // Melodic Minor Progressions (Ascending)
  MelodicMinor: [
    {
      name: "Triads",
      type: "triad",
      items: [
        "i – ii – IV – V",
        "i – IV – V – i",
        "i – ii – bIII+ – V (– i)",
        "ii – V – i",
      ],
    },
    {
      name: "Sevenths",
      type: "7th",
      items: [
        "i(Δ7) – ii7 – IV7 – V7",
        "i(Δ7) – IV7 – V7 – i(Δ7)",
        "ii7 – V7 – i(Δ7)",
      ],
    },
    {
      name: "Ninths",
      type: "9th",
      items: ["i(Δ9) – ii9 – IV9 – V9", "ii9 – V9 – i(Δ9)"],
    },
    {
      name: "Augmented",
      type: "augmented",
      items: ["i – bIII+ – IV – V", "i – V+ – IV – i"],
    },
    {
      name: "Characteristic",
      type: "characteristic",
      items: ["i – IV7 – V7 – i", "i – ii7 – viiø7 – V7"],
    },
  ],
};

function getGroupedProgressionsForMode(mode) {
  return MODE_PROGRESSION_GROUPS[mode] || [];
}

// Get progressions for a specific mode
function getProgressionsForMode(mode) {
  return MODE_PROGRESSIONS[mode] || MODE_PROGRESSIONS["Ionian"];
}

// Convert Roman numeral to scale degree number - EXACT from old project
function romanToNumber(roman) {
  const romanMap = {
    i: 1,
    I: 1,
    ii: 2,
    II: 2,
    bII: 2,
    iii: 3,
    III: 3,
    biii: 3,
    bIII: 3,
    iv: 4,
    IV: 4,
    v: 5,
    V: 5,
    bV: 5,
    vi: 6,
    VI: 6,
    bVI: 6,
    vii: 7,
    VII: 7,
    bvii: 7,
    bVII: 7,
  };
  return romanMap[roman] || null;
}

// Convert scale degree number to Roman numeral for specific mode - EXACT from old project
function numberToRomanForMode(number, mode) {
  const romanMap = MODE_ROMANS[mode] || MODE_ROMANS["Ionian"];
  return romanMap[number] || "I";
}

// Adapt a chord from one mode to another - EXACT from old project
function adaptChordToMode(chord, mode) {
  // Extract the chord number and any extensions - handle flats properly
  const match = chord.match(/^(b?[ivIV]+)(.*)$/);
  if (!match) return chord;

  const [, roman, extension] = match;
  const chordNumber = romanToNumber(roman);

  if (chordNumber === null) return chord;

  // Convert back to roman numeral appropriate for the mode
  return numberToRomanForMode(chordNumber, mode) + extension;
}

// Update progressions when mode changes
function updateProgressionsForMode(wrapper, mode) {
  const progressionSelect = wrapper.querySelector(".progression-select");
  if (!progressionSelect) return;

  // Store current selection
  const currentValue = progressionSelect.value;

  // Clear existing options
  progressionSelect.innerHTML = "";

  // Get progressions for the mode
  const progressions = getProgressionsForMode(mode);

  progressions.forEach((prog) => {
    const option = document.createElement("option");
    option.value = prog;
    option.textContent = prog;
    progressionSelect.appendChild(option);
  });

  // Try to maintain selection if it exists in new mode
  if (progressions.includes(currentValue)) {
    progressionSelect.value = currentValue;
  } else {
    // Set to first progression and adapt existing chords
    progressionSelect.value = progressions[0];
  }
}

// Adapt existing chords to new mode
function adaptChordsToNewMode(wrapper, newMode) {
  // Get current chord values from the grid
  const chordGrid = wrapper.querySelector(".chord-grid");
  const progressionSelect = wrapper.querySelector(".progression-select");
  let currentChords = [];

  // Priority 1: Get chords from the grid if it exists and has chords
  if (chordGrid && chordGrid.chordGridInstance) {
    const currentProgression =
      chordGrid.chordGridInstance.getCurrentProgression();
    if (currentProgression && currentProgression !== "custom") {
      currentChords = currentProgression.split(" - ").map((s) => s.trim());
    }
  }

  // Priority 2: If no grid chords, get from progression select (if not custom)
  if (
    currentChords.length === 0 &&
    progressionSelect.value &&
    progressionSelect.value !== "custom"
  ) {
    currentChords = progressionSelect.value.split(" - ").map((s) => s.trim());
  }

  // Update available progressions for new mode
  updateProgressionsForMode(wrapper, newMode);
  const progressions = getProgressionsForMode(newMode);

  // If we had chords, adapt them to the new mode
  if (currentChords.length > 0) {
    const adaptedChords = currentChords.map((chord) =>
      adaptChordToMode(chord, newMode)
    );
    const adaptedProgression = adaptedChords.join(" - ");

    // Check if this adapted progression matches any preset in the new mode
    const matchingProgression = progressions.find(
      (prog) => prog === adaptedProgression
    );

    if (matchingProgression) {
      // Found a match, select it
      progressionSelect.value = matchingProgression;
    } else {
      // No match, switch to custom
      progressionSelect.value = "custom";
    }

    // Update the grid with the adapted chords
    if (chordGrid && chordGrid.chordGridInstance) {
      // Use the grid's adapt method which preserves timing/positions
      chordGrid.chordGridInstance.adaptToMode(newMode);
    }
  } else {
    // No existing chords, set to first progression of new mode
    progressionSelect.value = progressions[0];

    // Update the grid with the new progression
    if (chordGrid && chordGrid.chordGridInstance) {
      chordGrid.chordGridInstance.updateProgression(progressions[0]);
    }
  }
}

const modeFunctionalMap = {
  Ionian: {
    I: ["ii", "iii", "IV", "V", "vi", "vii°"],
    ii: ["V", "vii°"],
    iii: ["vi", "IV"],
    IV: ["ii", "V", "I"],
    V: ["I", "vi"],
    vi: ["ii", "IV"],
    "vii°": ["I"],
  },
  Dorian: {
    i: ["IV", "v", "bVII", "ii"],
    ii: ["v", "i"],
    IV: ["v", "i", "bVII"],
    v: ["i", "bVII"],
    bVII: ["i", "IV"],
  },
  Phrygian: {
    i: ["bII", "bVI", "bVII"],
    bII: ["i", "bVI"],
    bIII: ["bVI", "bVII", "i"],
    bVI: ["bVII", "bII"],
    bVII: ["i", "bVI"],
  },
  Lydian: {
    I: ["II", "V", "vi", "iii"],
    II: ["V", "I"],
    iii: ["vi", "IV"],
    IV: ["I", "V"],
    V: ["I"],
  },
  Mixolydian: {
    I: ["bVII", "IV", "v"],
    bVII: ["IV", "I"],
    IV: ["I", "bVII"],
    v: ["I", "bVII"],
  },
  Aeolian: {
    i: ["iv", "bVI", "bVII", "v"],
    "ii°": ["v", "bVI"],
    iii: ["iv", "bVII"],
    iv: ["v", "bVII"],
    v: ["i", "bVI"],
    bVI: ["iv", "bVII"],
    bVII: ["i", "iv"],
  },
  Locrian: {
    "i°": ["bII", "bV"],
    bII: ["biii", "i°"],
    biii: ["bVII", "i°"],
    iv: ["bV", "bVI"],
    bV: ["bII", "i°"],
    bVI: ["bII", "bVII"],
    bVII: ["i°"],
  },
};

// Export the functional map for use in other modules
if (typeof window !== "undefined") {
  window.modeFunctionalMap = modeFunctionalMap;
}
