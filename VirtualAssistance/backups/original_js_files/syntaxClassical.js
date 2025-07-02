import {
  modeLegalProgressionMap,
  normalizeChord,
} from "./legalityClassical.js";

// Classical Syntax - contains classical harmony rules, chord maps, and legality
class ClassicalSyntax {
  constructor() {
    this.name = "Classical";
    this.id = "classical";
  }

  // Get mode-specific chord data for classical harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"],
          "7th": ["IM7"],
          "9th": ["IM9"],
          augmented: ["I+"],
          suspended: ["Isus4", "Isus2"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["ii9", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4", "IVsus2"],
        },
        dominant: {
          triad: ["vii°", "V"],
          "7th": ["viiø7", "V7"],
          "9th": ["viiø9", "V9"],
          augmented: ["V+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["iii", "vi"],
          "7th": ["iii7", "vi7"],
          "9th": ["iii9", "vi9"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Dorian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7"],
          "9th": ["i9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["ii9", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus2"],
        },
        dominant: {
          triad: ["bVII", "v"],
          "7th": ["bVII7", "v7"],
          "9th": ["bVII9", "v9"],
          augmented: ["v+"],
          suspended: ["Vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "vi"],
          "7th": ["bIII7", "vi7"],
          "9th": ["bIII9", "vi9"],
          augmented: ["bIII+"],
          suspended: ["bIIIsus4"],
        },
      },
      Phrygian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7"],
          "9th": ["i9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["iv", "bII"],
          "7th": ["iv7", "bII7"],
          "9th": ["iv9", "bII9"],
          augmented: ["iv+", "bII+"],
          suspended: ["ivsus4", "bIIsus4"],
        },
        dominant: {
          triad: ["bVII", "v"],
          "7th": ["bVII7", "v7"],
          "9th": ["bVII9", "v9"],
          augmented: ["v+"],
          suspended: ["Vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "bVI"],
          "7th": ["bIII7", "bVI7"],
          "9th": ["bIII9", "bVI9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4"],
        },
      },
      Lydian: {
        tonic: {
          triad: ["I"],
          "7th": ["IM7"],
          "9th": ["IM9"],
          augmented: ["I+"],
          suspended: ["Isus4"],
        },
        subdominant: {
          triad: ["ii", "#iv°"],
          "7th": ["ii7", "#ivø7"],
          "9th": ["ii9", "#ivø9"],
          augmented: ["ii+"],
          suspended: ["iisus4", "#ivsus4"],
        },
        dominant: {
          triad: ["vii", "V"],
          "7th": ["vii7", "V7"],
          "9th": ["vii9", "V9"],
          augmented: ["V+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["II", "iii", "vi"],
          "7th": ["II7", "iii7", "vi7"],
          "9th": ["II9", "iii9", "vi9"],
          augmented: ["II+", "iii+"],
          suspended: ["IIsus4"],
        },
      },
      Mixolydian: {
        tonic: {
          triad: ["I"],
          "7th": ["I7"],
          "9th": ["I9"],
          augmented: ["I+"],
          suspended: ["Isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["ii9", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["bVII", "v"],
          "7th": ["bVII7", "v7"],
          "9th": ["bVII9", "v9"],
          augmented: ["v+"],
          suspended: ["Vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["iii", "vi"],
          "7th": ["iii7", "vi7"],
          "9th": ["iii9", "vi9"],
          augmented: ["iii+"],
          suspended: ["iiisus4"],
        },
      },
      Aeolian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7"],
          "9th": ["i9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv"],
          "7th": ["iiø7", "iv7"],
          "9th": ["iiø9", "iv9"],
          augmented: ["iv+"],
          suspended: ["iisus4", "ivsus4"],
        },
        dominant: {
          triad: ["bVII", "v"],
          "7th": ["bVII7", "v7"],
          "9th": ["bVII9", "v9"],
          augmented: ["v+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["bIII", "bVI"],
          "7th": ["bIII7", "bVI7"],
          "9th": ["bIII9", "bVI9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4"],
        },
      },
      Locrian: {
        tonic: {
          triad: ["i°"],
          "7th": ["iø7"],
          "9th": ["iø9"],
          augmented: [],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["iv", "bVI"],
          "7th": ["iv7", "bVI7"],
          "9th": ["iv9", "bVI9"],
          augmented: ["bVI+"],
          suspended: ["ivsus4", "bVIsus4"],
        },
        dominant: {
          triad: ["bVII", "bV"],
          "7th": ["bVII7", "bV7"],
          "9th": ["bVII9", "bV9"],
          augmented: [],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["bII", "biii"],
          "7th": ["bII7", "biii7"],
          "9th": ["bII9", "biii9"],
          augmented: ["bII+", "biii+"],
          suspended: ["bIIsus4", "biiisus4"],
        },
      },
      HarmonicMinor: {
        tonic: {
          triad: ["i"],
          "7th": ["i7"],
          "9th": ["i9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv"],
          "7th": ["iiø7", "iv7"],
          "9th": ["iiø9", "iv9"],
          augmented: ["iv+"],
          suspended: ["iisus4", "ivsus4"],
        },
        dominant: {
          triad: ["V", "vii°"],
          "7th": ["V7", "viiø7"],
          "9th": ["V9", "viiø9"],
          augmented: ["V+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["bIII+", "bVI"],
          "7th": ["bIII+7", "bVIM7"],
          "9th": ["bIII+9", "bVIM9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
      MelodicMinor: {
        tonic: {
          triad: ["i"],
          "7th": ["i7"],
          "9th": ["i9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IV7"],
          "9th": ["ii9", "IV9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V", "vii°"],
          "7th": ["V7", "viiø7"],
          "9th": ["V9", "viiø9"],
          augmented: ["V+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["bIII+", "vi"],
          "7th": ["bIII+7", "vi7"],
          "9th": ["bIII+9", "vi9"],
          augmented: ["bIII+", "vi+"],
          suspended: ["bIIIsus4", "visus4"],
        },
      },
    };
  }

  // Get chord characteristics for modal analysis (classical perspective)
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    const chordLower = chord.toLowerCase();

    // Define borrowed chords by mode (classical theory)
    const borrowedChords = {
      Phrygian: ["V", "V7", "V9", "Vsus4"], // V is borrowed in Phrygian
      Dorian: ["vii°", "viiø7", "viiø9"], // vii° is borrowed in Dorian
      MelodicMinor: ["bIII+", "vi+"], // Augmented chords borrowed in Melodic Minor
    };

    // Define modal signature chords (classical analysis)
    const modalSignatures = {
      Phrygian: ["bII", "bii"], // bII is the Phrygian signature
      Lydian: ["#iv°", "#ivø7", "#ivø9", "#IV", "#iv"], // #iv° is Lydian signature
      Mixolydian: ["bVII", "bvii"], // bVII is Mixolydian signature
      Dorian: ["bVII", "bvii"], // bVII is also Dorian signature
      MelodicMinor: ["IV", "ii"], // IV and ii are signature chords
      HarmonicMinor: ["V", "vii°"], // V and vii° are signature chords
    };

    // Define tonic substitutions (classical perspective)
    const tonicSubstitutions = {
      Dorian: ["vi", "vi7", "vi9"], // vi as tonic substitution
      Mixolydian: ["vi", "vi7", "vi9"], // vi as tonic substitution
    };

    // Check if chord is borrowed
    if (
      borrowedChords[mode] &&
      borrowedChords[mode].some(
        (borrowed) => chord === borrowed || chord.startsWith(borrowed)
      )
    ) {
      characteristics.isBorrowed = true;
      if (mode === "Phrygian" && chord.startsWith("V")) {
        characteristics.tooltip =
          "Borrowed chord - not diatonic in Phrygian but often used for cadences";
      } else if (mode === "Dorian" && chordLower.includes("vii")) {
        characteristics.tooltip =
          "Borrowed chord - leading tone chord borrowed from parallel major";
      } else if (
        mode === "MelodicMinor" &&
        (chord === "bIII+" || chord === "vi+")
      ) {
        characteristics.tooltip =
          "Borrowed chord - augmented chords not naturally occurring in melodic minor";
      }
    }

    // Check if chord is modal signature
    if (
      modalSignatures[mode] &&
      modalSignatures[mode].some(
        (signature) => chord === signature || chord.startsWith(signature)
      )
    ) {
      characteristics.isModalSignature = true;
      if (mode === "Phrygian" && chordLower.includes("bii")) {
        characteristics.tooltip =
          "Modal signature - bII is the characteristic sound of Phrygian mode";
      } else if (mode === "Lydian" && chordLower.includes("#iv")) {
        characteristics.tooltip =
          "Modal signature - #iv° is the characteristic sound of Lydian mode";
      } else if (
        (mode === "Mixolydian" || mode === "Dorian") &&
        chordLower.includes("bvii")
      ) {
        characteristics.tooltip = `Modal signature - bVII is characteristic of ${mode} mode`;
      } else if (
        mode === "MelodicMinor" &&
        (chord === "IV" || chord === "ii")
      ) {
        characteristics.tooltip = `Modal signature - ${chord} is characteristic of Melodic Minor mode`;
      } else if (
        mode === "HarmonicMinor" &&
        (chord === "V" || chord === "vii°")
      ) {
        characteristics.tooltip = `Modal signature - ${chord} is characteristic of Harmonic Minor mode`;
      }
    }

    // Check if chord is tonic substitution
    if (
      tonicSubstitutions[mode] &&
      tonicSubstitutions[mode].some(
        (extension) => chord === extension || chord.startsWith(extension)
      )
    ) {
      characteristics.isTonicSubstitution = true;
      characteristics.tooltip = `Tonic substitution - extends the tonic harmony in ${mode} mode`;
    }

    // Special cases (classical analysis)
    if (mode === "Locrian" && (chord === "bV" || chord.startsWith("bV"))) {
      characteristics.tooltip =
        "Dual function - can serve as both tonic substitute and dominant";
    }

    return characteristics;
  }

  // Render legend for classical syntax
  renderLegend(container) {
    const legend = document.createElement("div");
    legend.className = "progression-legend";
    legend.style.position = "absolute";
    legend.style.bottom = "12px";
    legend.style.right = "18px";
    legend.style.background = "rgba(30,30,30,0.95)";
    legend.style.border = "1px solid #444";
    legend.style.borderRadius = "6px";
    legend.style.padding = "8px 14px 8px 10px";
    legend.style.fontSize = "12px";
    legend.style.color = "#ccc";
    legend.style.zIndex = "10001";
    legend.style.boxShadow = "0 2px 8px #0006";
    legend.innerHTML = `
      <b>Classical Map Key</b><br>
      <span style="font-size:14px;">✨</span> Modal signature<br>
      <span style="font-size:14px;">🥷</span> Borrowed chord<br>
      <span style="font-size:14px;">⭐</span> Tonic substitution
    `;
    container.appendChild(legend);
  }

  // Classical voice leading rules and progression validation
  validateProgressionMove(fromChord, toChord, mode) {
    const fromBase = normalizeChord(fromChord);
    const toBase = normalizeChord(toChord);

    // Use Ionian as fallback for unknown modes within this classical map
    const actualModeMap =
      modeLegalProgressionMap[mode] || modeLegalProgressionMap["Ionian"];
    const legalBaseMoves = actualModeMap[fromBase] || [];

    const isLegal = legalBaseMoves.includes(toBase);

    // For suggestions, we'd ideally expand extensions. For now, return base moves.
    // A more complete getLegalMoves function could be built here or in legalityClassical.js
    const legalOptions = legalBaseMoves;

    return {
      isLegal: isLegal,
      fromChord: fromChord,
      toChord: toChord,
      legalOptions: legalOptions, // Suggests base chords for now
      mode: mode,
    };
  }
}

// Export the classical syntax
export { ClassicalSyntax };
