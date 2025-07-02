// R&B/NeoSoul Syntax - contains R&B/NeoSoul harmony rules, chord maps, and progressions
class RnBSyntax {
  constructor() {
    this.name = "R&B / NeoSoul";
    this.id = "rnb_neosoul";
  }

  // Get mode-specific chord data for R&B/NeoSoul harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"],
          "7th": ["IM7", "I6"], // Both maj7 and 6th chords
          "9th": ["IM9", "I6/9", "IM7#11"], // Extended harmony essential
          augmented: ["I+", "IM7#5"],
          suspended: ["Isus4", "Isus2"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "ii9", "IVM7"], // ii9 very common
          "9th": ["ii9", "ii11", "ii13", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V"],
          "7th": ["V7", "V13"], // Extended dominants
          "9th": ["V9", "V13", "V7alt", "V7#11"], // Altered chords
          augmented: ["V+", "V7#5"],
          suspended: ["Vsus4", "V7sus4"],
        },
        other: {
          triad: ["iii", "vi"],
          "7th": ["iii7", "vi7", "vi9"], // vi9 very soulful
          "9th": ["iii9", "vi9", "vi11"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Dorian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7", "i9"], // Extended minor chords
          "9th": ["i9", "i11", "i6/9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["ii9", "ii11", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V", "bVII"],
          "7th": ["V7", "bVII7", "bVIIM7"],
          "9th": ["V9", "V13", "bVII9"],
          augmented: ["V+"],
          suspended: ["Vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "vi"],
          "7th": ["bIIIM7", "vi7"],
          "9th": ["bIIIM9", "vi9"],
          augmented: ["bIII+"],
          suspended: ["bIIIsus4", "visus4"],
        },
      },
      Aeolian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7", "i6"], // Minor 6 very soulful
          "9th": ["i9", "i6/9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv", "IV"], // Both natural and borrowed
          "7th": ["iiø7", "iv7", "IVM7"],
          "9th": ["iv9", "IVM9"],
          augmented: ["iv+", "IV+"],
          suspended: ["iisus4", "ivsus4", "IVsus4"],
        },
        dominant: {
          triad: ["V", "v", "bVII"],
          "7th": ["V7", "v7", "bVII7"],
          "9th": ["V9", "v9", "bVII9"],
          augmented: ["V+", "v+"],
          suspended: ["Vsus4", "vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "bVI"],
          "7th": ["bIIIM7", "bVIM7"],
          "9th": ["bIIIM9", "bVIM9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
      Mixolydian: {
        tonic: {
          triad: ["I"],
          "7th": ["I7", "I13"], // Dominant 7th and 13th on tonic
          "9th": ["I9", "I13"],
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
          triad: ["bVII", "V"],
          "7th": ["bVII7", "bVIIM7", "V7"],
          "9th": ["bVII9", "V9"],
          augmented: ["V+"],
          suspended: ["bVIIsus4", "Vsus4"],
        },
        other: {
          triad: ["iii", "vi"],
          "7th": ["iii7", "vi7"],
          "9th": ["iii9", "vi9"],
          augmented: ["iii+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
    };
  }

  // Get chord characteristics for R&B/NeoSoul analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    // R&B/NeoSoul-specific characteristics
    if (chord.includes("9") || chord.includes("11") || chord.includes("13")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Extended harmony - essential for that smooth R&B/NeoSoul color";
    }

    if (chord.includes("6") && !chord.includes("6/")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "6th chord - creates the warm, soulful R&B sound";
    }

    if (chord.includes("6/9")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip = "6/9 chord - quintessential NeoSoul harmony";
    }

    if (chord === "ii9" || chord === "ii11") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Extended ii chord - backbone of R&B progressions";
    }

    if (
      chord.includes("alt") ||
      chord.includes("#11") ||
      chord.includes("b9")
    ) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Altered harmony - adds sophisticated jazz influence to R&B";
    }

    if (chord === "vi9" || chord === "vi7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip = "Extended vi - creates emotional depth in R&B";
    }

    // Modal characteristics
    if (chord === "bVII7" && (mode === "Dorian" || mode === "Mixolydian")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Flat VII7 - adds modal flavor to contemporary R&B";
    }

    // Borrowed chords
    if (chord === "IV" && mode === "Aeolian") {
      characteristics.isBorrowed = true;
      characteristics.tooltip =
        "Major IV in minor - borrowed chord popular in emotional R&B";
    }

    return characteristics;
  }

  // Render legend for R&B/NeoSoul syntax
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
      <b>R&B/NeoSoul Map Key</b><br>
      <span style="font-size:14px;">🎤</span> Extended harmony<br>
      <span style="font-size:14px;">✨</span> Soulful colors<br>
      <span style="font-size:14px;">⭐</span> Smooth grooves
    `;
    container.appendChild(legend);
  }

  // R&B/NeoSoul progression validation
  validateProgressionMove(fromChord, toChord, mode) {
    // R&B/NeoSoul is very flexible harmonically, influenced by jazz
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the R&B/NeoSoul syntax
export { RnBSyntax };
