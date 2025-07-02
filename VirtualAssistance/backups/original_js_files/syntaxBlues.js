// Blues Syntax - contains blues harmony rules, chord maps, and progressions
class BluesSyntax {
  constructor() {
    this.name = "Blues";
    this.id = "blues";
  }

  // Get mode-specific chord data for blues harmony
  getModeChordData() {
    // Blues does not use modal harmony in the classical sense.
    // Return a static dominant-based chord map for all modes.
    const bluesChordMap = {
      tonic: {
        "7th": ["I7", "bIII7"],
        "9th": ["I9", "bIII9"],
        augmented: ["I+"],
        suspended: ["Isus4"],
      },
      subdominant: {
        "7th": ["IV7", "bVI7"],
        "9th": ["IV9", "bVI9"],
        augmented: ["IV+"],
        suspended: ["IVsus4"],
      },
      dominant: {
        "7th": ["V7", "bVII7"],
        "9th": ["V9", "bVII9"],
        augmented: ["V+"],
        suspended: ["Vsus4"],
      },
      other: {
        "7th": ["vi7"],
        "9th": ["vi9"],
        augmented: ["bII+"],
        suspended: [],
      },
    };

    // Return the same chord map for all modes since blues doesn't change by mode
    return {
      Ionian: bluesChordMap,
      Dorian: bluesChordMap,
      Phrygian: bluesChordMap,
      Lydian: bluesChordMap,
      Mixolydian: bluesChordMap,
      Aeolian: bluesChordMap,
      Locrian: bluesChordMap,
    };
  }

  // Get chord characteristics for blues analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    const chordLower = chord.toLowerCase();

    // Blues-specific characteristics
    if (chord.includes("7") && !chord.includes("M7")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip = "Dominant 7th - essential blues harmony";
    }

    if (chord === "I7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Blues tonic - dominant 7th on the tonic creates blues flavor";
    }

    if (chord === "IV7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Blues subdominant - the heart of the 12-bar blues";
    }

    if (chord === "V7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Blues dominant - creates tension and resolution in blues";
    }

    if (chord === "bVII7" && mode === "Mixolydian") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Mixolydian blues - flat VII dominant adds modal flavor";
    }

    // Blue notes and alterations
    if (chord.includes("b5") || chord.includes("#11")) {
      characteristics.isBorrowed = true;
      characteristics.tooltip =
        "Blue note harmony - incorporates the blues scale";
    }

    return characteristics;
  }

  // Render legend for blues syntax
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
      <b>Blues Map Key</b><br>
      <span style="font-size:14px;">🎵</span> Dominant 7th core<br>
      <span style="font-size:14px;">🎸</span> Blue note harmony<br>
      <span style="font-size:14px;">⭐</span> 12-bar standard
    `;
    container.appendChild(legend);
  }

  // Blues progression validation
  validateProgressionMove(fromChord, toChord, mode) {
    // Blues has very flexible voice leading
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the blues syntax
export { BluesSyntax };
