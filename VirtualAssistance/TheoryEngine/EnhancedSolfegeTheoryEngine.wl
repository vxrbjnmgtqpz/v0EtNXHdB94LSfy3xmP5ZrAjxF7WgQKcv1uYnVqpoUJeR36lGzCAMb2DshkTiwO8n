(* Enhanced Solfege Theory Engine for MIDI Chord Generation *)
(* Complete multi-style theory engine using Wolfram Language *)

ClearAll["Global`*"];
SetDirectory[NotebookDirectory[]];

(* Load all JSON data files *)
Print["=== Loading Theory Engine Database ==="];

solfegeChords = Import["solfegeChords.json", "JSON"];
legalityAll = Import["legalityAll.json", "JSON"];
modulation = Import["modulation.json", "JSON"];

(* Load all style syntax data from consolidated file *)
syntaxAll = Import["syntaxAll.json", "JSON"];

Print["✓ Loaded solfege chord database with ", Length[Keys[solfegeChords]], " modes"];
Print["✓ Loaded comprehensive progression rules for ", Length[Keys[legalityAll]], " styles"];
Print["✓ Loaded ", Length[Keys[modulation]], " modulation/theory objects"];
Print["✓ Loaded consolidated syntax database with ", Length[Keys[syntaxAll]], " styles"];

(* Available styles and modes *)
availableStyles = {"Blues", "Jazz", "Classical", "Pop", "Rock", "Folk", "RnB", "Cinematic"};
availableModes = {"Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"};

Print["Available styles: ", availableStyles];
Print["Available modes: ", availableModes];

(* Solfege to MIDI note conversion functions *)
solfegeToMIDI[solfege_String, rootNote_Integer: 60] := Module[{solfegeMap, interval},
  solfegeMap = Association[
    "Do" -> 0,   (* Root *)
    "Ra" -> 1,   (* b2 in chromatic solfege *)
    "Re" -> 2,   (* 2nd *)
    "Me" -> 3,   (* b3 minor third *)
    "Mi" -> 4,   (* Major third *)
    "Fa" -> 5,   (* Perfect fourth *)
    "Fi" -> 6,   (* #4 tritone *)
    "Sol" -> 7,  (* Perfect fifth *)
    "Se" -> 8,   (* #5 augmented fifth *)
    "Le" -> 8,   (* b6 *)
    "La" -> 9,   (* Major sixth *)
    "Te" -> 10,  (* b7 *)
    "Ti" -> 11   (* Major seventh *)
  ];
  interval = Lookup[solfegeMap, solfege, 0];
  rootNote + interval
];

(* Generate MIDI chord from solfege representation *)
generateMIDIChord[mode_String, chordSymbol_String, rootNote_Integer: 60] := Module[{solfegeNotes, midiNotes},
  solfegeNotes = Lookup[Lookup[solfegeChords, mode, Association[]], chordSymbol, {}];
  If[Length[solfegeNotes] == 0,
    Print["Warning: Chord ", chordSymbol, " not found in mode ", mode];
    Return[{}]
  ];
  midiNotes = Table[solfegeToMIDI[note, rootNote], {note, solfegeNotes}];
  Sort[midiNotes]
];

(* Get style-specific chord data *)
getStyleChordData[style_String, mode_String] := Module[{styleData, modeData},
  (* Access style data from consolidated syntax file *)
  styleData = Lookup[syntaxAll, style, Lookup[syntaxAll, "Classical", <||>]];
  
  (* Extract mode-specific chord data *)
  modeData = Lookup[styleData, mode, <||>];
  modeData
];

(* Generate style-appropriate chord progression *)
generateStyleProgression[style_String, mode_String, length_Integer: 4, rootNote_Integer: 60] := Module[{
  chordData, functionGroups, progression, currentFunction, nextFunction, chordOptions, selectedChord
},
  chordData = getStyleChordData[style, mode];
  
  If[Length[Keys[chordData]] == 0,
    Print["Warning: No chord data available for style ", style, " in mode ", mode];
    Return[{}]
  ];
  
  functionGroups = Keys[chordData];
  progression = {};
  
  (* Start with tonic if available, otherwise first available function *)
  currentFunction = If[MemberQ[functionGroups, "tonic"], "tonic", First[functionGroups]];
  
  Do[
    (* Get chord options for current function *)
    chordOptions = Flatten[Values[Lookup[chordData, currentFunction, Association[]]]];
    
    If[Length[chordOptions] > 0,
      (* Select a random chord from available options *)
      selectedChord = RandomChoice[chordOptions];
      AppendTo[progression, selectedChord];
      
      (* Choose next function based on common progressions *)
      nextFunction = Switch[currentFunction,
        "tonic", RandomChoice[{"subdominant", "dominant", "other"}],
        "subdominant", RandomChoice[{"dominant", "tonic"}],
        "dominant", RandomChoice[{"tonic", "subdominant"}],
        "other", RandomChoice[{"tonic", "subdominant", "dominant"}],
        _, RandomChoice[functionGroups]
      ];
      
      (* Ensure the next function exists in this style/mode *)
      If[!MemberQ[functionGroups, nextFunction],
        nextFunction = RandomChoice[functionGroups]
      ];
      
      currentFunction = nextFunction;
    ],
    {i, length}
  ];
  
  progression
];

(* Convert chord progression to MIDI *)
progressionToMIDI[progression_List, mode_String, rootNote_Integer: 60] := Module[{midiProgression},
  midiProgression = Table[
    generateMIDIChord[mode, chord, rootNote],
    {chord, progression}
  ];
  midiProgression
];

(* Emotion to style mapping *)
emotionToStyle[emotion_String] := Module[{styleMap},
  styleMap = Association[
    "happy" -> "Pop",
    "sad" -> "Folk",
    "energetic" -> "Rock", 
    "peaceful" -> "Classical",
    "romantic" -> "Jazz",
    "melancholy" -> "Blues",
    "dramatic" -> "Cinematic",
    "soulful" -> "RnB",
    "nostalgic" -> "Folk",
    "uplifting" -> "Pop",
    "contemplative" -> "Classical",
    "passionate" -> "Jazz"
  ];
  
  Lookup[styleMap, ToLowerCase[emotion], "Classical"]
];

(* Main generation function with emotion support *)
generateEmotionalProgression[emotion_String, mode_String: "Ionian", length_Integer: 4, rootNote_Integer: 60] := Module[{
  style, progression, midiProgression, result
},
  style = emotionToStyle[emotion];
  progression = generateStyleProgression[style, mode, length, rootNote];
  midiProgression = progressionToMIDI[progression, mode, rootNote];
  
  result = Association[
    "emotion" -> emotion,
    "style" -> style,
    "mode" -> mode,
    "rootNote" -> rootNote,
    "chordSymbols" -> progression,
    "midiChords" -> midiProgression,
    "length" -> length
  ];
  
  Print["Generated ", emotion, " progression in ", style, " style (", mode, " mode):"];
  Print["Chords: ", progression];
  
  result
];

(* Analysis functions *)
analyzeChordProgression[progression_List, mode_String] := Module[{
  analysis, chordFunctions, functionalAnalysis
},
  chordFunctions = Association[
    "I" -> "Tonic", "i" -> "Tonic",
    "ii" -> "Subdominant", "ii7" -> "Subdominant", "II" -> "Subdominant",
    "iii" -> "Mediant", "iii7" -> "Mediant", "III" -> "Mediant", 
    "IV" -> "Subdominant", "iv" -> "Subdominant", "IVM7" -> "Subdominant",
    "V" -> "Dominant", "V7" -> "Dominant", "v" -> "Dominant",
    "vi" -> "Submediant", "VI" -> "Submediant", "vi7" -> "Submediant",
    "vii°" -> "Leading Tone", "vii\[Degree]7" -> "Leading Tone"
  ];
  
  functionalAnalysis = Table[
    Lookup[chordFunctions, chord, "Other"],
    {chord, progression}
  ];
  
  analysis = Association[
    "progression" -> progression,
    "mode" -> mode,
    "functions" -> functionalAnalysis,
    "cadences" -> findCadences[progression],
    "modulations" -> findModulations[progression]
  ];
  
  analysis
];

(* Helper functions for analysis *)
findCadences[progression_List] := Module[{cadences, i},
  cadences = {};
  For[i = 1, i <= Length[progression] - 1, i++,
    If[MatchQ[{progression[[i]], progression[[i+1]]}, {"V" | "V7", "I" | "i"}],
      AppendTo[cadences, {"Authentic", i, i+1}]
    ];
    If[MatchQ[{progression[[i]], progression[[i+1]]}, {"IV" | "iv", "I" | "i"}],
      AppendTo[cadences, {"Plagal", i, i+1}]
    ];
  ];
  cadences
];

findModulations[progression_List] := Module[{modulations},
  (* Simple modulation detection - this could be enhanced *)
  modulations = {};
  (* TODO: Implement sophisticated modulation detection *)
  modulations
];

(* Export functions *)
exportProgressionToMIDI[progression_Association, filename_String: "progression.mid"] := Module[{
  midiData, tracks
},
  (* TODO: Implement MIDI file export using Wolfram's MIDI capabilities *)
  Print["MIDI export functionality to be implemented"];
  Print["Progression data: ", progression];
];

(* Demo and test functions *)
demonstrateTheoryEngine[] := Module[{emotions, modes, results},
  Print["=== Theory Engine Demonstration ==="];
  
  emotions = {"happy", "sad", "energetic", "peaceful"};
  modes = {"Ionian", "Dorian", "Aeolian", "Mixolydian"};
  
  results = Table[
    generateEmotionalProgression[emotion, mode, 4, 60],
    {emotion, emotions[[1;;2]]}, {mode, modes[[1;;2]]}
  ];
  
  Print["Generated ", Length[Flatten[results]], " progressions"];
  results
];

Print["\n=== Solfege Theory Engine Loaded Successfully ==="];
Print["Available functions:"];
Print["• generateMIDIChord[mode, chord, rootNote]"];
Print["• generateStyleProgression[style, mode, length, rootNote]"];
Print["• generateEmotionalProgression[emotion, mode, length, rootNote]"];
Print["• progressionToMIDI[progression, mode, rootNote]"];
Print["• analyzeChordProgression[progression, mode]"];
Print["• demonstrateTheoryEngine[]"];
Print["\nReady for music generation!"];

(* Generate style-aware progression using legality rules *)
generateLegalProgression[style_String, mode_String, length_Integer: 4, startChord_String: "auto"] := Module[{
  legalityRules, availableChords, progression, currentChord, nextOptions, nextChord
},
  (* Get legality rules for this style and mode *)
  legalityRules = Lookup[Lookup[legalityAll, style, Association[]], mode, Association[]];
  
  If[Length[Keys[legalityRules]] == 0,
    Print["Warning: No legality rules available for style ", style, " in mode ", mode];
    Return[{}]
  ];
  
  availableChords = Keys[legalityRules];
  
  (* Choose starting chord *)
  currentChord = If[startChord == "auto",
    (* Try to start with a tonic chord *)
    If[MemberQ[availableChords, "I"], "I",
      If[MemberQ[availableChords, "i"], "i",
        RandomChoice[availableChords]
      ]
    ],
    startChord
  ];
  
  progression = {currentChord};
  
  (* Generate remaining chords based on legality rules *)
  Do[
    nextOptions = Lookup[legalityRules, currentChord, availableChords];
    
    If[Length[nextOptions] > 0,
      (* Weight the choices based on style preferences *)
      nextChord = weightedChordChoice[style, nextOptions, currentChord, i == length];
      AppendTo[progression, nextChord];
      currentChord = nextChord,
      
      (* Fallback if no legal options *)
      nextChord = RandomChoice[availableChords];
      AppendTo[progression, nextChord];
      currentChord = nextChord
    ],
    {i, 2, length}
  ];
  
  progression
];

(* Weighted chord choice based on style characteristics *)
weightedChordChoice[style_String, options_List, currentChord_String, isLast_: False] := Module[{
  weights, chordWeights
},
  (* Default equal weights *)
  weights = ConstantArray[1.0, Length[options]];
  
  (* Style-specific weighting *)
  Switch[style,
    "Jazz",
      (* Prefer extended chords and ii-V-I movement *)
      chordWeights = Table[
        Which[
          StringContainsQ[chord, "7"] || StringContainsQ[chord, "9"], 1.5,
          StringMatchQ[chord, "ii*"] && StringMatchQ[currentChord, "V*"], 2.0,
          StringMatchQ[chord, "I*"] && StringMatchQ[currentChord, "V*"], 2.0,
          True, 1.0
        ],
        {chord, options}
      ];
      weights = chordWeights,
      
    "Blues",
      (* Prefer 7th chords and traditional blues progressions *)
      chordWeights = Table[
        Which[
          StringContainsQ[chord, "7"], 2.0,
          StringMatchQ[chord, "IV*"] && StringMatchQ[currentChord, "I*"], 1.8,
          StringMatchQ[chord, "V*"] && StringMatchQ[currentChord, "IV*"], 1.8,
          StringMatchQ[chord, "I*"] && StringMatchQ[currentChord, "V*"], 1.8,
          True, 1.0
        ],
        {chord, options}
      ];
      weights = chordWeights,
      
    "Pop",
      (* Prefer simple, catchy progressions *)
      chordWeights = Table[
        Which[
          StringMatchQ[chord, "I"] || StringMatchQ[chord, "V"] || StringMatchQ[chord, "vi"] || StringMatchQ[chord, "IV"], 1.8,
          StringContainsQ[chord, "sus"] || StringContainsQ[chord, "add"], 1.3,
          True, 1.0
        ],
        {chord, options}
      ];
      weights = chordWeights,
      
    "Rock",
      (* Prefer power chords and modal progressions *)
      chordWeights = Table[
        Which[
          StringContainsQ[chord, "b"], 1.5, (* Modal chords *)
          StringMatchQ[chord, "I"] || StringMatchQ[chord, "IV"] || StringMatchQ[chord, "V"], 1.4,
          True, 1.0
        ],
        {chord, options}
      ];
      weights = chordWeights,
      
    _,
      (* Default weighting *)
      weights = ConstantArray[1.0, Length[options]]
  ];
  
  (* If it's the last chord, prefer tonic resolution *)
  If[isLast,
    Do[
      If[StringMatchQ[options[[i]], "I*"] || StringMatchQ[options[[i]], "i*"],
        weights[[i]] *= 2.0
      ],
      {i, Length[options]}
    ]
  ];
  
  RandomChoice[weights -> options]
];

(* Generate and compare progressions across multiple styles *)
compareStyleProgressions[mode_String: "Ionian", length_Integer: 4] := Module[{
  results, style
},
  results = Association[];
  
  Do[
    results[style] = generateLegalProgression[style, mode, length];
    Print[style, ": ", StringRiffle[results[style], " → "]],
    {style, availableStyles}
  ];
  
  results
];
