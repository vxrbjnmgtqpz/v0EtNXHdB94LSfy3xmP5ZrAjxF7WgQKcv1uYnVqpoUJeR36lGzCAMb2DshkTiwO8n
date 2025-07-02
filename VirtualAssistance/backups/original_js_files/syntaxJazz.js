// Jazz Syntax - contains jazz harmony rules, chord maps, and progressions
class JazzSyntax {
  constructor() {
    this.name = "Jazz";
    this.id = "jazz";
  }

  // Get mode-specific chord data for jazz harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"],
          "7th": ["IM7"], // Jazz emphasizes maj7 on tonic
          "9th": ["IM9", "I6/9"],
          augmented: ["I+", "IM7#5"],
          suspended: ["Isus4", "Isus2"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"], // ii7 is crucial in jazz
          "9th": ["ii9", "ii11", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V"],
          "7th": ["V7", "V7alt"], // Altered dominants common
          "9th": ["V9", "V13", "V7b9", "V7#11"],
          augmented: ["V+", "V7#5"],
          suspended: ["Vsus4", "V7sus4"],
        },
        other: {
          triad: ["iii", "vi", "vii°"],
          "7th": ["iii7", "vi7", "viiø7"], // Half-diminished important
          "9th": ["iii9", "vi9", "viiø9"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Dorian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7", "i6"], // Minor 6 chord very jazzy
          "9th": ["i9", "i6/9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["ii9", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V", "bVII"],
          "7th": ["V7", "bVII7"],
          "9th": ["V9", "bVII9"],
          augmented: ["V+"],
          suspended: ["Vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "vi"],
          "7th": ["bIIIM7", "vi7"],
          "9th": ["bIIIM9", "vi9"],
          augmented: ["bIII+"],
          suspended: ["bIIIsus4"],
        },
      },
      Mixolydian: {
        tonic: {
          triad: ["I"],
          "7th": ["I7"], // Mixolydian I7 for jazz
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
          triad: ["V", "bVII"],
          "7th": ["V7", "bVII7"],
          "9th": ["V9", "bVII9"],
          augmented: ["V+"],
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
    };
  }

  // Get chord characteristics for jazz analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    // Jazz-specific characteristics
    if (chord.includes("M7") || chord.includes("maj7")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Major 7th - creates jazz sophistication and color";
    }

    if (chord === "ii7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "ii7 - the workhorse of jazz harmony, setup for V7";
    }

    if (
      chord.includes("alt") ||
      chord.includes("b9") ||
      chord.includes("#11")
    ) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Altered harmony - sophisticated jazz color and tension";
    }

    if (chord.includes("ø7")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Half-diminished 7th - essential jazz chord for ii in minor keys";
    }

    if (chord.includes("6/9")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip = "6/9 chord - classic jazz tonic sound";
    }

    // Tritone substitutions
    if (chord.includes("bII7") || chord.includes("bV7")) {
      characteristics.isBorrowed = true;
      characteristics.tooltip = "Tritone substitution - replaces V7 with bII7";
    }

    // Secondary dominants
    if (chord.includes("V7/") || chord.match(/[IVivi]+7$/)) {
      characteristics.isBorrowed = true;
      characteristics.tooltip =
        "Secondary dominant - tonicizes the target chord";
    }

    return characteristics;
  }

  // Render legend for jazz syntax
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
      <b>Jazz Map Key</b><br>
      <span style="font-size:14px;">🎷</span> Extended harmony<br>
      <span style="font-size:14px;">🎹</span> Altered chords<br>
      <span style="font-size:14px;">⭐</span> ii-V-I motion
    `;
    container.appendChild(legend);
  }

  // Jazz progression validation (very flexible)
  validateProgressionMove(fromChord, toChord, mode) {
    // Jazz allows nearly anything with proper voice leading
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the jazz syntax
export { JazzSyntax };
