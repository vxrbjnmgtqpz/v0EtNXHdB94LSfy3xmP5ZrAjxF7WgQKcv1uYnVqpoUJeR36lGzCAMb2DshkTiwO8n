// Folk Syntax - contains folk harmony rules, chord maps, and progressions
class FolkSyntax {
  constructor() {
    this.name = "Folk";
    this.id = "folk";
  }

  // Get mode-specific chord data for folk harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"], // Folk emphasizes simple triads
          "7th": ["IM7"],
          "9th": ["Iadd9"],
          augmented: ["I+"],
          suspended: ["Isus4", "Isus2"], // Sus chords common in folk
        },
        subdominant: {
          triad: ["ii", "IV"], // ii and IV very common
          "7th": ["ii7", "IVM7"],
          "9th": ["iiadd9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4", "IVsus2"],
        },
        dominant: {
          triad: ["V", "vii°"],
          "7th": ["V7", "viiø7"],
          "9th": ["Vadd9"],
          augmented: ["V+"],
          suspended: ["Vsus4"],
        },
        other: {
          triad: ["iii", "vi"], // vi especially common
          "7th": ["iii7", "vi7"],
          "9th": ["iiiadd9", "viadd9"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Dorian: {
        tonic: {
          triad: ["i"], // Modal folk often uses Dorian
          "7th": ["i7"],
          "9th": ["iadd9"],
          augmented: ["i+"],
          suspended: ["isus4", "isus2"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["iiadd9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["bVII", "V"], // bVII very characteristic of folk
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
      Mixolydian: {
        tonic: {
          triad: ["I"],
          "7th": ["I7"], // I7 in folk/country style
          "9th": ["Iadd9"],
          augmented: ["I+"],
          suspended: ["Isus4"],
        },
        subdominant: {
          triad: ["ii", "IV"],
          "7th": ["ii7", "IVM7"],
          "9th": ["iiadd9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4"],
        },
        dominant: {
          triad: ["bVII", "V"], // Both used in Mixolydian folk
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
          triad: ["i"], // Natural minor common in folk
          "7th": ["i7"],
          "9th": ["iadd9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv"], // ii° and iv typical
          "7th": ["iiø7", "iv7"],
          "9th": ["ivadd9"],
          augmented: ["iv+"],
          suspended: ["iisus4", "ivsus4"],
        },
        dominant: {
          triad: ["bVII", "V"], // bVII very common, V less so
          "7th": ["bVII7", "V7"],
          "9th": ["bVIIadd9"],
          augmented: ["V+"],
          suspended: ["bVIIsus4", "Vsus4"],
        },
        other: {
          triad: ["bIII", "bVI"], // Typical minor mode chords
          "7th": ["bIIIM7", "bVIM7"],
          "9th": ["bIIIadd9", "bVIadd9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
    };
  }

  // Get chord characteristics for folk analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    // Folk-specific characteristics
    if (chord.includes("sus")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Suspended chord - creates the open, ringing sound of folk guitar";
    }

    if (chord.includes("add9")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip = "Add9 chord - folk color tone without the 7th";
    }

    if (chord === "bVII" || chord === "bVII7") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Flat VII - creates the modal folk sound, avoids leading tone";
    }

    if (chord === "vi" && mode === "Ionian") {
      characteristics.isTonicSubstitution = true;
      characteristics.tooltip =
        "Relative minor - very common in folk for emotional contrast";
    }

    if (chord === "IV" && (mode === "Dorian" || mode === "Mixolydian")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Major IV - brightens the modal sound in folk music";
    }

    // Open tuning implications
    if (chord === "I" || chord === "V") {
      characteristics.tooltip =
        "Perfect for open tunings and drone strings in folk guitar";
    }

    return characteristics;
  }

  // Render legend for folk syntax
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
      <b>Folk Map Key</b><br>
      <span style="font-size:14px;">🪕</span> Open tuning friendly<br>
      <span style="font-size:14px;">🎸</span> Suspended colors<br>
      <span style="font-size:14px;">⭐</span> Modal harmony
    `;
    container.appendChild(legend);
  }

  // Folk progression validation (simple traditional rules)
  validateProgressionMove(fromChord, toChord, mode) {
    // Folk generally follows traditional voice leading but is quite flexible
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the folk syntax
export { FolkSyntax };
