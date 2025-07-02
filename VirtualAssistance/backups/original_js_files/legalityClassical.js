const modeLegalProgressionMap = {
  Ionian: {
    I: ["I", "ii", "iii", "IV", "V", "vi", "vii°"],
    ii: ["ii", "V", "vii°"],
    iii: ["iii", "vi", "IV"],
    IV: ["IV", "ii", "V", "I"],
    V: ["V", "I", "vi"],
    vi: ["vi", "ii", "IV"],
    "vii°": ["vii°", "I", "V"],
  },
  Dorian: {
    i: ["i", "IV", "v", "bVII", "ii"],
    ii: ["ii", "v", "i"],
    bIII: ["bIII", "vi", "i"],
    IV: ["IV", "bVII", "v", "i"],
    v: ["v", "i", "bVII"],
    vi: ["vi", "ii", "IV"],
    bVII: ["bVII", "i", "IV"],
  },
  Phrygian: {
    i: ["i", "bII", "bIII", "bVI", "bVII"],
    bII: ["bII", "bVI", "i"],
    bIII: ["bIII", "bVI", "bVII", "i"],
    iv: ["iv", "bV", "bVI"],
    v: ["v", "i", "bVI"],
    bVI: ["bVI", "bVII", "bII", "i"],
    bVII: ["bVII", "i", "bII"],
  },
  Lydian: {
    I: ["I", "II", "V", "IV", "vi", "iii"],
    II: ["II", "V", "I"],
    iii: ["iii", "vi", "IV"],
    IV: ["IV", "I", "V"],
    V: ["V", "I", "vi"],
    vi: ["vi", "ii", "IV"],
    vii: ["vii", "I", "V"],
  },
  Mixolydian: {
    I: ["I", "bVII", "IV", "v", "ii"],
    ii: ["ii", "v", "bVII"],
    iii: ["iii", "vi", "IV"],
    IV: ["IV", "I", "bVII", "v"],
    v: ["v", "I", "bVII"],
    vi: ["vi", "ii", "IV"],
    bVII: ["bVII", "IV", "I"],
  },
  Aeolian: {
    i: ["i", "iv", "bVI", "bVII", "v"],
    "ii°": ["ii°", "v", "bVI"],
    bIII: ["bIII", "iv", "bVII", "i"],
    iv: ["iv", "bVII", "v"],
    v: ["v", "i", "bVI"],
    bVI: ["bVI", "iv", "bVII", "i"],
    bVII: ["bVII", "i", "iv"],
  },
  Locrian: {
    "i°": ["i°", "bII", "bV", "bVI"],
    bII: ["bII", "biii", "i°"],
    biii: ["biii", "bVII", "i°"],
    iv: ["iv", "bV", "bVI"],
    bV: ["bV", "bII", "i°"],
    bVI: ["bVI", "bII", "bVII"],
    bVII: ["bVII", "i°"],
  },
};

// function normalizeChord(chord) { // Older version removed
//   return chord
//     .replace(/7|9|11|13|\+|sus(2|4)?|add\d+|Δ|M/gi, "")
//     .replace(/ø/g, "°");
// }

export { modeLegalProgressionMap, normalizeChord };

function normalizeChord(chord) {
  return chord
    .replace(/7|9|11|13|\+|aug|sus(2|4)?|add\d+|Δ|M/gi, "")
    .replace(/ø/g, "°")
    .trim();
}
