// Rock Syntax - contains rock harmony rules, chord maps, and progressions
class RockSyntax {
  constructor() {
    this.name = "Rock";
    this.id = "rock";
  }

  // Get mode-specific chord data for rock harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"], // Power chords and simple triads
          "7th": ["I7"], // Rock often uses dominant 7th
          "9th": ["Iadd9"],
          augmented: ["I+"],
          suspended: ["Isus4", "Isus2"], // Sus chords very common
        },
        subdominant: {
          triad: ["ii", "IV"], // IV chord is essential
          "7th": ["ii7", "IV7"],
          "9th": ["iiadd9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V"],
          "7th": ["V7"], // Classic rock V7
          "9th": ["Vadd9"],
          augmented: ["V+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["iii", "vi"], // vi common for emotional sections
          "7th": ["iii7", "vi7"],
          "9th": ["iiiadd9", "viadd9"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Mixolydian: {
        tonic: {
          triad: ["I"],
          "7th": ["I7"], // I7 gives rock edge
          "9th": ["Iadd9"],
          augmented: ["I+"],
          suspended: ["Isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IV7"],
          "9th": ["iiadd9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["bVII", "V"], // bVII crucial for rock
          "7th": ["bVII7", "V7"],
          "9th": ["bVIIadd9", "Vadd9"],
          augmented: ["V+"],
          suspended: ["bVIIsus4", "Vsus4"],
        },
        other: {
          triad: ["iii", "vi"],
          "7th": ["iii7", "vi7"],
          "9th": ["iiiadd9", "viadd9"],
          augmented: ["iii+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Aeolian: {
        tonic: {
          triad: ["i"], // Minor rock very common
          "7th": ["i7"],
          "9th": ["iadd9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv"],
          "7th": ["iiø7", "iv7"],
          "9th": ["ivadd9"],
          augmented: ["iv+"],
          suspended: ["iisus4", "ivsus4"],
        },
        dominant: {
          triad: ["bVII", "V"], // bVII very common in rock
          "7th": ["bVII7", "V7"],
          "9th": ["bVIIadd9"],
          augmented: ["V+"],
          suspended: ["bVIIsus4", "Vsus4"],
        },
        other: {
          triad: ["bIII", "bVI"],
          "7th": ["bIIIM7", "bVIM7"],
          "9th": ["bIIIadd9", "bVIadd9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
      Dorian: {
        tonic: {
          triad: ["i"], // Dorian for progressive rock
          "7th": ["i7"],
          "9th": ["iadd9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["iiadd9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["bVII", "V"],
          "7th": ["bVII7", "V7"],
          "9th": ["bVIIadd9", "Vadd9"],
          augmented: ["V+"],
          suspended: ["bVIIsus4", "Vsus4"],
        },
        other: {
          triad: ["bIII", "vi"],
          "7th": ["bIIIM7", "vi7"],
          "9th": ["bIIIadd9", "viadd9"],
          augmented: ["bIII+"],
          suspended: ["bIIIsus4", "visus4"],
        },
      },
    };
  }

  // Get chord characteristics for rock analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    // Rock-specific characteristics
    if (chord.includes("sus")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Suspended chord - creates tension and release in rock riffs";
    }

    if (chord === "bVII" || chord === "bVII7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Flat VII - classic rock sound, avoids resolution to tonic";
    }

    if (chord === "I7" && mode === "Mixolydian") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Dominant tonic - gives rock its edgy, unresolved sound";
    }

    if (chord === "IV") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "IV chord - backbone of rock progressions, strong and stable";
    }

    if (chord === "vi" && mode === "Ionian") {
      characteristics.isTonicSubstitution = true;
      characteristics.tooltip =
        "Relative minor - used for emotional contrast in rock ballads";
    }

    // Power chord implications
    if (chord === "I" || chord === "IV" || chord === "V" || chord === "bVII") {
      characteristics.tooltip =
        (characteristics.tooltip || "") + " Perfect for power chords (1-5-8)";
    }

    return characteristics;
  }

  // Render legend for rock syntax
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
      <b>Rock Map Key</b><br>
      <span style="font-size:14px;">🎸</span> Power chord ready<br>
      <span style="font-size:14px;">🔥</span> Distortion friendly<br>
      <span style="font-size:14px;">⭐</span> Classic rock moves
    `;
    container.appendChild(legend);
  }

  // Rock progression validation
  validateProgressionMove(fromChord, toChord, mode) {
    // Rock is quite flexible with voice leading
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the rock syntax
export { RockSyntax };
