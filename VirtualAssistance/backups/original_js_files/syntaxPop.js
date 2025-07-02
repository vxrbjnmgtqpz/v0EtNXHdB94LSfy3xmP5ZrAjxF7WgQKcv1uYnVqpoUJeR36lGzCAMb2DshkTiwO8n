// Pop Syntax - contains pop harmony rules, chord maps, and progressions
class PopSyntax {
  constructor() {
    this.name = "Pop";
    this.id = "pop";
  }

  // Get mode-specific chord data for pop harmony
  getModeChordData() {
    return {
      Ionian: {
        tonic: {
          triad: ["I"], // Simple and catchy
          "7th": ["IM7"], // Pop uses maj7 for sophistication
          "9th": ["Iadd9", "IM9"], // Add9 very common in pop
          augmented: ["I+"],
          suspended: ["Isus4", "Isus2"], // Sus chords for color
        },
        subdominant: {
          triad: ["ii", "IV"], // IV extremely common
          "7th": ["ii7", "IVM7"],
          "9th": ["iiadd9", "ii9", "IVadd9"],
          augmented: ["ii+", "IV+"],
          suspended: ["iisus4", "IVsus4", "IVsus2"],
        },
        dominant: {
          triad: ["V"],
          "7th": ["V7"], // Standard pop dominant
          "9th": ["Vadd9", "V9"],
          augmented: ["V+"],
          suspended: ["Vsus4"], // Very common in pop
        },
        other: {
          triad: ["iii", "vi"], // vi for sad/emotional sections
          "7th": ["iii7", "vi7"],
          "9th": ["iiiadd9", "viadd9"],
          augmented: ["iii+", "vi+"],
          suspended: ["iiisus4", "visus4"],
        },
      },
      Aeolian: {
        tonic: {
          triad: ["i"], // Minor pop ballads
          "7th": ["i7"],
          "9th": ["iadd9"],
          augmented: ["i+"],
          suspended: ["isus4"],
        },
        subdominant: {
          triad: ["ii°", "iv", "IV"], // Both minor and major IV
          "7th": ["iiø7", "iv7", "IVM7"],
          "9th": ["ivadd9", "IVadd9"],
          augmented: ["iv+", "IV+"],
          suspended: ["iisus4", "ivsus4", "IVsus4"],
        },
        dominant: {
          triad: ["V", "bVII"], // Both major V and bVII
          "7th": ["V7", "bVII7"],
          "9th": ["Vadd9", "bVIIadd9"],
          augmented: ["V+"],
          suspended: ["Vsus4", "bVIIsus4"],
        },
        other: {
          triad: ["bIII", "bVI"],
          "7th": ["bIIIM7", "bVIM7"],
          "9th": ["bIIIadd9", "bVIadd9"],
          augmented: ["bIII+", "bVI+"],
          suspended: ["bIIIsus4", "bVIsus4"],
        },
      },
      Mixolydian: {
        tonic: {
          triad: ["I"],
          "7th": ["I7"], // Mixolydian adds edge to pop
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
          triad: ["bVII", "V"],
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
    };
  }

  // Get chord characteristics for pop analysis
  getChordCharacteristics(chord, mode) {
    const characteristics = {
      isBorrowed: false,
      isModalSignature: false,
      isTonicSubstitution: false,
      tooltip: null,
    };

    // Pop-specific characteristics
    if (chord.includes("add9")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Add9 chord - creates that modern pop sparkle and color";
    }

    if (chord === "vi" && mode === "Ionian") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "vi chord - the heart of emotional pop, relative minor";
    }

    if (chord === "IV") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "IV chord - pop's most beloved chord, warm and uplifting";
    }

    if (chord.includes("sus")) {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Suspended chord - creates anticipation and modern pop texture";
    }

    if (chord === "V" || chord === "Vsus4") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Pop dominant - drives the hook and creates momentum";
    }

    // Borrowed chords common in pop
    if (chord === "IV" && mode === "Aeolian") {
      characteristics.isBorrowed = true;
      characteristics.tooltip =
        "Major IV in minor - borrowed from parallel major, very popular in pop";
    }

    if (chord === "bVII") {
      characteristics.isModalSignature = true;
      characteristics.tooltip =
        "Flat VII - adds modal flavor while staying accessible";
    }

    return characteristics;
  }

  // Render legend for pop syntax
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
      <b>Pop Map Key</b><br>
      <span style="font-size:14px;">✨</span> Modern color tones<br>
      <span style="font-size:14px;">💫</span> Catchy progressions<br>
      <span style="font-size:14px;">⭐</span> Hook-friendly
    `;
    container.appendChild(legend);
  }

  // Pop progression validation
  validateProgressionMove(fromChord, toChord, mode) {
    // Pop is very flexible - almost anything works if it sounds good
    return {
      isLegal: true,
      fromChord,
      toChord,
      legalOptions: [],
    };
  }
}

// Export the pop syntax
export { PopSyntax };
