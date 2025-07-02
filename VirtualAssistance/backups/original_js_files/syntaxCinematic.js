// Cinematic Syntax - contains film music harmony rules, chord maps, and progressions
class CinematicSyntax {
  constructor() {
    this.name = "Cinematic";
    this.id = "cinematic";
  }

  // Get mode-specific chord data for cinematic harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"],
          "7th": ["IM7"], // Orchestral sophistication
          "9th": ["IM9", "IM7#11"], // Extended harmony for color
          augmented: ["I+", "IM7#5"],
          suspended: ["Isus4", "Isus2"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7", "ii7b5"], // Half-diminished for tension
          "9th": ["ii9", "ii11", "IVM9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["V"],
          "7th": ["V7", "V7sus4"], // Orchestral dominants
          "9th": ["V9", "V13", "V7#11"], // Complex dominants
          augmented: ["V+", "V7#5"],
          suspended: ["Vsus4", "Vsus2"],
        },
        other: {
          triad: ["iii", "vi", "vii°"],
          "7th": ["iii7", "vi7", "viiø7"],
          "9th": ["iii9", "vi9", "viiø9"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Aeolian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7", "iM7"], // Both minor and minor-major
          "9th": ["i9", "iM9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv", "IV"], // Both natural and raised
          "7th": ["iiø7", "iv7", "IVM7"],
          "9th": ["iv9", "IVM9"],
          augmented: ["iv+", "IV+"],
          suspended: ["iisus4", "ivsus4", "IVsus4"],
        },
        dominant: {
          triad: ["V", "v", "bVII"], // All three options
          "7th": ["V7", "v7", "bVII7"],
          "9th": ["V9", "v9", "bVII9"],
          augmented: ["V+", "v+"],
          suspended: ["Vsus4", "vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "bVI", "VII°"], // Includes raised 7
          "7th": ["bIIIM7", "bVIM7", "VIIø7"],
          "9th": ["bIIIM9", "bVIM9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
      Dorian: {
        tonic: {
          triad: ["i"],
          "7th": ["i7", "i6"], // Minor 6 for color
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
          suspended: ["bIIIsus4", "visus4"],
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
          triad: ["bII", "iv"], // bII very dramatic
          "7th": ["bIIM7", "iv7"],
          "9th": ["bIIM9", "iv9"],
          augmented: ["bII+", "iv+"],
          suspended: ["bIIsus4", "ivsus4"],
        },
        dominant: {
          triad: ["bVII", "V"],
          "7th": ["bVII7", "V7"],
          "9th": ["bVII9", "V9"],
          augmented: ["V+"],
          suspended: ["bVIIsus4", "Vsus4"],
        },
        other: {
          triad: ["bIII", "bVI"],
          "7th": ["bIIIM7", "bVIM7"],
          "9th": ["bIIIM9", "bVIM9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
    };
  }

  // Get chord characteristics for cinematic analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    // Cinematic-specific characteristics
    if (chord.includes("M7") || chord.includes("maj7")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Major 7th - creates lush, orchestral cinematic sound";
    }

    if (chord.includes("#11") || chord.includes("b5")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Altered harmony - adds tension and mystery for film";
    }

    if (chord.includes("sus")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Suspended chord - creates floating, ethereal cinematic texture";
    }

    if (chord === "bII" && mode === "Phrygian") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Neapolitan chord - dramatic and exotic, perfect for film scores";
    }

    if (chord.includes("ø7")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Half-diminished 7th - creates sophisticated tension in film music";
    }

    // Modal borrowing common in film music
    if (chord === "IV" && mode === "Aeolian") {
      characteristics.isBorrowed = true;
      characteristics.tooltip =
        "Picardy third region - borrowed major IV for hope/triumph";
    }

    if (chord === "V" && mode === "Aeolian") {
      characteristics.isBorrowed = true;
      characteristics.tooltip =
        "Raised 7th - borrowed major V for dramatic resolution";
    }

    // Extended/altered chords
    if (chord.includes("13") || chord.includes("9") || chord.includes("11")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Extended harmony - creates rich orchestral colors";
    }

    return characteristics;
  }

  // Render legend for cinematic syntax
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
      <b>Cinematic Map Key</b><br>
      <span style="font-size:14px;">🎬</span> Orchestral colors<br>
      <span style="font-size:14px;">🎭</span> Dramatic tension<br>
      <span style="font-size:14px;">⭐</span> Film score moves
    `;
    container.appendChild(legend);
  }

  // Cinematic progression validation
  validateProgressionMove(fromChord, toChord, mode) {
    // Film music is extremely flexible harmonically
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the cinematic syntax
export { CinematicSyntax };
