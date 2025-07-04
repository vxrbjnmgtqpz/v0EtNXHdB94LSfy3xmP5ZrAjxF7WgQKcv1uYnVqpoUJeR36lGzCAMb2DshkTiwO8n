(* Voice Leading & Register Mapping Engine *)
(* Advanced harmonic progression with smooth voice leading and emotional register mapping *)

BeginPackage["VoiceLeadingEngine`"]

(* Export functions *)
OptimizeVoiceLeading::usage = "OptimizeVoiceLeading[chords, emotions, key] optimizes chord voicings for smooth voice leading"
MapEmotionToRegister::usage = "MapEmotionToRegister[emotion] maps emotional content to appropriate octave registers"
CalculateVoiceDistance::usage = "CalculateVoiceDistance[voicing1, voicing2] calculates voice movement distance"
HandleKeyChange::usage = "HandleKeyChange[fromKey, toKey, progression] manages smooth key modulation"

Begin["`Private`"]

(* Chromatic pitch classes with octave information *)
chromaticPitchClasses = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

(* Convert note to MIDI number for precise calculation *)
noteToMIDI[note_String, octave_Integer] := 
  Module[{pitchClass, chromaticIndex},
    pitchClass = StringReplace[note, {"b" -> "♭", "#" -> "♯"}];
    chromaticIndex = FirstPosition[chromaticPitchClasses, pitchClass][[1]] - 1;
    12 * (octave + 1) + chromaticIndex
  ]

(* Convert MIDI back to note for output *)
midiToNote[midiNumber_Integer] := 
  Module[{octave, pitchClass},
    octave = Floor[midiNumber/12] - 1;
    pitchClass = chromaticPitchClasses[[Mod[midiNumber, 12] + 1]];
    {pitchClass, octave}
  ]

(* Emotional Register Mapping - Core Innovation *)
emotionRegisterMap = <|
  (* Aggressive/Dark emotions - Lower registers *)
  "Anger" -> {2, 3, 4},
  "Metal" -> {1, 2, 3},
  "Malice" -> {2, 3},
  "Disgust" -> {2, 3, 4},
  
  (* Transcendent/Ethereal emotions - Higher registers *)
  "Transcendence" -> {5, 6, 7},
  "Aesthetic Awe" -> {5, 6, 7},
  "Wonder" -> {5, 6},
  "Reverence" -> {4, 5, 6},
  
  (* Bright/Positive emotions - Mid-high registers *)
  "Joy" -> {4, 5, 6},
  "Empowerment" -> {4, 5},
  "Gratitude" -> {4, 5},
  "Trust" -> {4, 5},
  
  (* Introspective emotions - Mid registers *)
  "Sadness" -> {3, 4, 5},
  "Love" -> {4, 5},
  "Shame" -> {3, 4},
  "Guilt" -> {3, 4},
  
  (* Tension/Anxiety - Higher registers *)
  "Fear" -> {5, 6, 7},
  "Anticipation" -> {4, 5, 6},
  "Surprise" -> {5, 6},
  
  (* Complex emotions - Extended range *)
  "Envy" -> {3, 4, 5},
  "Arousal" -> {4, 5, 6},
  "Belonging" -> {4, 5},
  "Ideology" -> {3, 4, 5},
  "Dissociation" -> {2, 3, 6, 7} (* Extreme registers for disconnection *)
|>;

(* Map emotional weights to target register preferences *)
MapEmotionToRegister[emotionWeights_Association] := 
  Module[{weightedRegisters, targetRange},
    weightedRegisters = KeyValueMap[
      Function[{emotion, weight},
        If[KeyExistsQ[emotionRegisterMap, emotion],
          weight * emotionRegisterMap[emotion],
          weight * {4, 5} (* Default mid-range *)
        ]
      ],
      emotionWeights
    ];
    
    (* Calculate weighted average of preferred registers *)
    targetRange = Mean[Flatten[weightedRegisters]];
    
    (* Return optimal register range based on emotional content *)
    {Floor[targetRange] - 1, Floor[targetRange], Ceiling[targetRange]}
  ]

(* Roman numeral to semitone intervals mapping *)
romanNumeralToIntervals = <|
  (* Major triads *)
  "I" -> {0, 4, 7}, "II" -> {2, 6, 9}, "III" -> {4, 8, 11}, 
  "IV" -> {5, 9, 0}, "V" -> {7, 11, 2}, "VI" -> {9, 1, 4}, "VII" -> {11, 3, 6},
  
  (* Minor triads *)
  "i" -> {0, 3, 7}, "ii" -> {2, 5, 9}, "iii" -> {4, 7, 11}, 
  "iv" -> {5, 8, 0}, "v" -> {7, 10, 2}, "vi" -> {9, 0, 4}, "vii" -> {11, 2, 6},
  
  (* Seventh chords *)
  "I7" -> {0, 4, 7, 10}, "V7" -> {7, 11, 2, 5}, "IM7" -> {0, 4, 7, 11},
  "ii7" -> {2, 5, 9, 0}, "vim7" -> {9, 0, 4, 7}, "vim7" -> {9, 0, 4, 7},
  
  (* Diminished chords *)
  "vii°" -> {11, 2, 5}, "ii°" -> {2, 5, 8}, "vi°" -> {9, 0, 3},
  
  (* Altered chords *)
  "♭VII" -> {10, 2, 5}, "♭III" -> {3, 7, 10}, "♭VI" -> {8, 0, 3}
|>;

(* Generate all possible inversions for a chord *)
generateAllInversions[chordIntervals_List, targetRegisters_List] := 
  Module[{baseOctave, inversions, allVoicings},
    baseOctave = Mean[targetRegisters];
    
    (* Generate root position and all inversions *)
    inversions = Table[
      RotateLeft[chordIntervals, i],
      {i, 0, Length[chordIntervals] - 1}
    ];
    
    (* Map each inversion to MIDI numbers in target registers *)
    allVoicings = Map[
      Function[inversion,
        MapIndexed[
          Function[{interval, pos},
            (* Calculate MIDI number with appropriate octave *)
            noteToMIDI["C", baseOctave] + interval + 
            If[pos[[1]] > 1 && interval < inversion[[pos[[1]] - 1]], 12, 0]
          ],
          inversion
        ]
      ],
      inversions
    ];
    
    (* Filter to keep voicings within target register range *)
    Select[allVoicings, 
      AllTrue[#, (Min[targetRegisters] * 12 <= # <= (Max[targetRegisters] + 1) * 12) &] &
    ]
  ]

(* Calculate voice movement distance between two voicings *)
CalculateVoiceDistance[voicing1_List, voicing2_List] := 
  Module[{pairedVoices, distances},
    (* Handle different chord sizes by optimal pairing *)
    pairedVoices = If[Length[voicing1] == Length[voicing2],
      Transpose[{voicing1, voicing2}],
      (* For different sizes, find optimal voice pairing *)
      findOptimalVoicePairing[voicing1, voicing2]
    ];
    
    (* Calculate total semitone movement *)
    distances = Map[Abs[#[[2]] - #[[1]]] &, pairedVoices];
    Total[distances]
  ]

(* Find optimal voice pairing for different chord sizes *)
findOptimalVoicePairing[voicing1_List, voicing2_List] := 
  Module[{shorter, longer, assignments, minDistanceAssignment},
    {shorter, longer} = If[Length[voicing1] <= Length[voicing2], 
      {voicing1, voicing2}, {voicing2, voicing1}];
    
    (* Find assignment that minimizes total voice movement *)
    assignments = Permutations[Take[longer, Length[shorter]]];
    minDistanceAssignment = MinimalBy[assignments, 
      Function[assignment, Total[MapThread[Abs[#1 - #2] &, {shorter, assignment}]]]
    ][[1]];
    
    Transpose[{shorter, minDistanceAssignment}]
  ]

(* Optimize voice leading for an entire progression *)
OptimizeVoiceLeading[chordProgression_List, emotionWeights_Association, key_String: "C"] := 
  Module[{targetRegisters, voicedProgression, currentVoicing, nextChordOptions, optimalVoicing},
    
    (* Map emotions to target registers *)
    targetRegisters = MapEmotionToRegister[emotionWeights];
    
    (* Initialize with first chord in optimal register *)
    currentVoicing = optimizeInitialVoicing[chordProgression[[1]], targetRegisters, key];
    voicedProgression = {currentVoicing};
    
    (* Optimize each subsequent chord for smooth voice leading *)
    Do[
      nextChordOptions = generateAllInversions[
        romanNumeralToIntervals[chordProgression[[i]]], 
        targetRegisters
      ];
      
      (* Find voicing with minimal voice movement *)
      optimalVoicing = MinimalBy[nextChordOptions, 
        CalculateVoiceDistance[currentVoicing, #] &
      ][[1]];
      
      AppendTo[voicedProgression, optimalVoicing];
      currentVoicing = optimalVoicing;
      ,
      {i, 2, Length[chordProgression]}
    ];
    
    (* Convert MIDI numbers back to note names with octaves *)
    Map[Map[midiToNote, #] &, voicedProgression]
  ]

(* Optimize initial chord voicing based on emotional register preference *)
optimizeInitialVoicing[firstChord_String, targetRegisters_List, key_String] := 
  Module[{chordIntervals, allVoicings, registerScores, optimalVoicing},
    chordIntervals = romanNumeralToIntervals[firstChord];
    allVoicings = generateAllInversions[chordIntervals, targetRegisters];
    
    (* Score each voicing based on how well it fits target registers *)
    registerScores = Map[
      Function[voicing,
        Mean[Map[
          Function[midiNote,
            (* Higher score for notes in preferred registers *)
            If[MemberQ[targetRegisters, Floor[midiNote/12] - 1], 1.0, 0.5]
          ],
          voicing
        ]]
      ],
      allVoicings
    ];
    
    (* Return voicing with best register fit *)
    allVoicings[[Position[registerScores, Max[registerScores]][[1, 1]]]]
  ]

(* Handle key changes with smooth voice leading *)
HandleKeyChange[fromKey_String, toKey_String, progression_List, emotionWeights_Association] := 
  Module[{fromKeyNumber, toKeyNumber, semitoneShift, pivotChords, modulatedProgression},
    
    (* Calculate semitone difference between keys *)
    fromKeyNumber = Position[chromaticPitchClasses, fromKey][[1, 1]] - 1;
    toKeyNumber = Position[chromaticPitchClasses, toKey][[1, 1]] - 1;
    semitoneShift = Mod[toKeyNumber - fromKeyNumber, 12];
    
    (* Find pivot chords that work in both keys *)
    pivotChords = findPivotChords[fromKey, toKey];
    
    (* Insert pivot chord if beneficial for smooth modulation *)
    modulatedProgression = If[Length[pivotChords] > 0,
      insertPivotChord[progression, pivotChords[[1]]],
      progression
    ];
    
    (* Transpose and optimize voice leading *)
    OptimizeVoiceLeading[modulatedProgression, emotionWeights, toKey]
  ]

(* Find chords that function in both keys for smooth modulation *)
findPivotChords[fromKey_String, toKey_String] := 
  Module[{fromKeyChords, toKeyChords, commonChords},
    (* This would contain sophisticated key relationship analysis *)
    (* For now, simplified common pivot relationships *)
    <|
      {"C", "G"} -> {"I", "vi", "IV"}, (* C to G: common chords *)
      {"C", "F"} -> {"I", "V", "vi"},  (* C to F: common chords *)
      {"C", "Am"} -> {"vi", "I", "IV"} (* C to Am: relative major/minor *)
    |>[Sort[{fromKey, toKey}]]
  ]

(* Insert pivot chord strategically in progression *)
insertPivotChord[progression_List, pivotChord_String] := 
  Module[{insertionPoint},
    (* Insert pivot chord before last chord for smooth resolution *)
    insertionPoint = Max[1, Length[progression] - 1];
    Insert[progression, pivotChord, insertionPoint]
  ]

(* Analyze harmonic rhythm and suggest optimal chord durations *)
analyzeHarmonicRhythm[voicedProgression_List, emotionWeights_Association] := 
  Module[{tensionCurve, chordDurations},
    (* Calculate tension based on voice leading complexity and emotional content *)
    tensionCurve = MapIndexed[
      Function[{voicing, pos},
        If[pos[[1]] == 1, 0.5, (* First chord baseline *)
          CalculateVoiceDistance[voicedProgression[[pos[[1]] - 1]], voicing] / 12.0
        ]
      ],
      voicedProgression
    ];
    
    (* Higher tension = shorter duration for dramatic effect *)
    chordDurations = Map[
      Function[tension,
        Which[
          tension > 0.7, 0.5, (* High tension: half note *)
          tension > 0.4, 1.0, (* Medium tension: whole note *)
          True, 2.0           (* Low tension: longer duration *)
        ]
      ],
      tensionCurve
    ];
    
    <|"tensions" -> tensionCurve, "durations" -> chordDurations|>
  ]

End[]
EndPackage[]

(* Usage Examples *)
(*
emotionalContext = <|"Anger" -> 0.8, "Transcendence" -> 0.2|>;
progression = {"i", "♭VII", "♭VI", "♯iv°"};
voicedResult = OptimizeVoiceLeading[progression, emotionalContext, "Am"];
*) 