#!/usr/bin/env python3
"""
Script to generate the expanded emotion progression database
Based on emotionexpansion2.md schema with sub-emotions
"""

import json

def create_expanded_database():
    """Generate the complete expanded emotion database"""
    
    database = {
        "database_info": {
            "version": "2.0",
            "description": "Expanded emotion-to-sub-emotion-to-progression mapping database with detailed musical analysis",
            "total_emotions": 12,
            "total_sub_emotions": 32,
            "progressions_per_sub_emotion": 3,
            "total_progressions": 96,
            "created": "2025-07-02",
            "architecture_notes": [
                "Enhanced database with granular sub-emotion mapping for deeper musical expression",
                "Each main emotion contains multiple sub-emotions with specific modal characteristics",
                "Sub-emotions include detailed musical explanations and theoretical foundations",
                "Genre weights enable probabilistic selection and style-aware generation",
                "Supports advanced emotional granularity for AI-driven composition"
            ]
        },
        "emotions": {
            "Joy": {
                "mode": "Ionian",
                "description": "Bright, uplifting emotions with positive energy",
                "sub_emotions": {
                    "Excitement": {
                        "mode": "Ionian with secondary dominant (II7)",
                        "description": "Using the major scale's supertonic as a dominant (II7 chord) before the V creates an expectant, giddy build-up fitting for excitement.",
                        "progression_pool": [
                            {
                                "chords": ["I", "II7", "V", "I"],
                                "genres": { "Pop": 0.9, "Gospel": 0.8, "Rock": 0.7 },
                                "progression_id": "excitement_001"
                            },
                            {
                                "chords": ["I", "vi", "II7", "V"],
                                "genres": { "Pop": 0.8, "Jazz": 0.7, "R&B": 0.6 },
                                "progression_id": "excitement_002"
                            },
                            {
                                "chords": ["IV", "V", "I", "II7"],
                                "genres": { "Gospel": 0.9, "Soul": 0.7, "Funk": 0.6 },
                                "progression_id": "excitement_003"
                            }
                        ]
                    },
                    "Contentment": {
                        "mode": "Ionian",
                        "description": "A simple loop of major chords avoids strong cadences, producing a neutral, soothing atmosphere that mirrors contentment.",
                        "progression_pool": [
                            {
                                "chords": ["I", "IV", "V", "IV"],
                                "genres": { "Ambient": 0.9, "New Age": 0.8, "Classical": 0.7 },
                                "progression_id": "contentment_001"
                            },
                            {
                                "chords": ["I", "IV", "I", "IV"],
                                "genres": { "Folk": 0.8, "Acoustic": 0.7, "Indie": 0.6 },
                                "progression_id": "contentment_002"
                            },
                            {
                                "chords": ["I", "V", "IV", "I"],
                                "genres": { "Country": 0.8, "Americana": 0.7, "Singer-Songwriter": 0.6 },
                                "progression_id": "contentment_003"
                            }
                        ]
                    },
                    "Pride": {
                        "mode": "Ionian",
                        "description": "A classic ii–V–I cadence (borrowed from jazz) offers a satisfying resolution, conveying confidence and triumph appropriate for pride.",
                        "progression_pool": [
                            {
                                "chords": ["ii", "V", "I"],
                                "genres": { "Jazz": 0.9, "Big Band": 0.8, "Anthemic": 0.7 },
                                "progression_id": "pride_001"
                            },
                            {
                                "chords": ["I", "vi", "ii", "V", "I"],
                                "genres": { "Jazz": 0.9, "Musical Theatre": 0.8, "Orchestral": 0.7 },
                                "progression_id": "pride_002"
                            },
                            {
                                "chords": ["I", "IV", "ii", "V"],
                                "genres": { "Pop": 0.8, "Rock": 0.7, "Anthem": 0.9 },
                                "progression_id": "pride_003"
                            }
                        ]
                    },
                    "Romantic Longing": {
                        "mode": "Ionian with modal mixture",
                        "description": "Shifting from a major tonic to a minor iv chord introduces a colorful, bittersweet sense of longing, famously used in love themes.",
                        "progression_pool": [
                            {
                                "chords": ["I", "iv"],
                                "genres": { "Soundtrack": 0.9, "Romantic": 0.9, "Classical": 0.8 },
                                "progression_id": "romantic_longing_001"
                            },
                            {
                                "chords": ["I", "vi", "iv", "I"],
                                "genres": { "Film Score": 0.9, "Ballad": 0.8, "Piano": 0.7 },
                                "progression_id": "romantic_longing_002"
                            },
                            {
                                "chords": ["I", "IV", "iv", "I"],
                                "genres": { "Romantic": 0.9, "Jazz": 0.7, "Bossa Nova": 0.6 },
                                "progression_id": "romantic_longing_003"
                            }
                        ]
                    }
                }
            },
            "Sadness": {
                "mode": "Aeolian",
                "description": "Melancholic emotions with introspective qualities",
                "sub_emotions": {
                    "Melancholy": {
                        "mode": "Aeolian",
                        "description": "A rise from minor i to major III and VII before descending back creates a melancholic, bittersweet sound.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭VI", "♭III", "♭VII"],
                                "genres": { "Ballad": 0.9, "Classical": 0.8, "Indie": 0.7 },
                                "progression_id": "melancholy_001"
                            },
                            {
                                "chords": ["i", "♭III", "♭VI", "♭VII"],
                                "genres": { "Post-Rock": 0.8, "Cinematic": 0.9, "Alternative": 0.7 },
                                "progression_id": "melancholy_002"
                            },
                            {
                                "chords": ["i", "iv", "♭VI", "♭III"],
                                "genres": { "Singer-Songwriter": 0.8, "Folk": 0.7, "Acoustic": 0.8 },
                                "progression_id": "melancholy_003"
                            }
                        ]
                    },
                    "Longing": {
                        "mode": "Aeolian",
                        "description": "Alternating minor and major chords yields a bittersweet, yearning effect, perfectly capturing a sense of longing.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭III", "♭VII", "iv"],
                                "genres": { "Ballad": 0.9, "Pop": 0.8, "R&B": 0.7 },
                                "progression_id": "longing_001"
                            },
                            {
                                "chords": ["i", "♭VI", "iv", "♭VII"],
                                "genres": { "Indie": 0.8, "Alternative": 0.7, "Dream Pop": 0.8 },
                                "progression_id": "longing_002"
                            },
                            {
                                "chords": ["i", "v", "♭VI", "♭III"],
                                "genres": { "Lo-fi": 0.8, "Chillhop": 0.7, "Neo-Soul": 0.6 },
                                "progression_id": "longing_003"
                            }
                        ]
                    },
                    "Nostalgia": {
                        "mode": "Ionian",
                        "description": "The '50s progression' (I–vi–IV–V) evokes a classic sadness and nostalgia reminiscent of mid-century pop ballads.",
                        "progression_pool": [
                            {
                                "chords": ["I", "vi", "IV", "V"],
                                "genres": { "50s Pop": 0.9, "Doo-wop": 0.9, "Oldies": 0.8 },
                                "progression_id": "nostalgia_001"
                            },
                            {
                                "chords": ["vi", "IV", "I", "V"],
                                "genres": { "Retro": 0.8, "Vintage": 0.8, "Easy Listening": 0.7 },
                                "progression_id": "nostalgia_002"
                            },
                            {
                                "chords": ["I", "vi", "ii", "V"],
                                "genres": { "Jazz Standards": 0.8, "Swing": 0.7, "Crooner": 0.8 },
                                "progression_id": "nostalgia_003"
                            }
                        ]
                    },
                    "Despair": {
                        "mode": "Aeolian",
                        "description": "A somber minor progression like VI–iv–i–v sustains a haunting, reflective mood, expressing deep despair.",
                        "progression_pool": [
                            {
                                "chords": ["♭VI", "iv", "i", "v"],
                                "genres": { "Dramatic Score": 0.9, "Classical": 0.8, "Metal": 0.7 },
                                "progression_id": "despair_001"
                            },
                            {
                                "chords": ["i", "♭VI", "♭III", "iv"],
                                "genres": { "Gothic": 0.9, "Darkwave": 0.8, "Doom": 0.7 },
                                "progression_id": "despair_002"
                            },
                            {
                                "chords": ["i", "iv", "♭VII", "♭VI"],
                                "genres": { "Post-Metal": 0.8, "Atmospheric": 0.8, "Ambient": 0.7 },
                                "progression_id": "despair_003"
                            }
                        ]
                    },
                    "Loneliness": {
                        "mode": "Aeolian",
                        "description": "A descending minor sequence (i–VII–VI–VII) creates a cyclical feeling of melancholy and isolation, echoing loneliness.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭VII", "♭VI", "♭VII"],
                                "genres": { "Alternative": 0.8, "Ambient": 0.8, "Shoegaze": 0.7 },
                                "progression_id": "loneliness_001"
                            },
                            {
                                "chords": ["i", "♭VI", "♭VII", "i"],
                                "genres": { "Lo-fi": 0.9, "Chill": 0.8, "Trip-hop": 0.7 },
                                "progression_id": "loneliness_002"
                            },
                            {
                                "chords": ["i", "iv", "♭VI", "♭VII"],
                                "genres": { "Indie Rock": 0.8, "Post-Punk": 0.7, "Slowcore": 0.8 },
                                "progression_id": "loneliness_003"
                            }
                        ]
                    }
                }
            },
            "Anger": {
                "mode": "Phrygian",
                "description": "Aggressive emotions with intense energy",
                "sub_emotions": {
                    "Rage": {
                        "mode": "Phrygian",
                        "description": "Phrygian mode's signature ♭II chord delivers an extra dark, aggressive sound suited for unbridled rage.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭II", "i"],
                                "genres": { "Metal": 0.9, "Industrial": 0.8, "Orchestral": 0.7 },
                                "progression_id": "rage_001"
                            },
                            {
                                "chords": ["♭II", "i", "♭VII", "i"],
                                "genres": { "Death Metal": 0.9, "Hardcore": 0.8, "Thrash": 0.8 },
                                "progression_id": "rage_002"
                            },
                            {
                                "chords": ["i", "♭II", "♭VI", "♭II"],
                                "genres": { "Black Metal": 0.9, "Doom": 0.8, "Sludge": 0.7 },
                                "progression_id": "rage_003"
                            }
                        ]
                    },
                    "Resentment": {
                        "mode": "Aeolian",
                        "description": "A brooding minor pattern that never fully resolves lends a haunting, introspective edge, matching the simmering nature of resentment.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭VI", "i", "♭VII"],
                                "genres": { "Alternative Rock": 0.8, "Cinematic": 0.8, "Grunge": 0.7 },
                                "progression_id": "resentment_001"
                            },
                            {
                                "chords": ["i", "iv", "♭VI", "v"],
                                "genres": { "Post-Rock": 0.8, "Metal": 0.7, "Progressive": 0.7 },
                                "progression_id": "resentment_002"
                            },
                            {
                                "chords": ["i", "♭VII", "iv", "♭VI"],
                                "genres": { "Dark Alternative": 0.8, "Industrial Rock": 0.7, "Gothic": 0.8 },
                                "progression_id": "resentment_003"
                            }
                        ]
                    },
                    "Hatred": {
                        "mode": "Double Harmonic (Byzantine)",
                        "description": "The Double Harmonic scale (with its two augmented seconds) is considered one of the darkest musical modes. Its sharp, exotic intervals intensify a feeling of deep hatred.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭II", "i"],
                                "genres": { "Horror Score": 0.9, "World": 0.8, "Experimental": 0.7 },
                                "progression_id": "hatred_001"
                            },
                            {
                                "chords": ["i", "♭II", "III", "i"],
                                "genres": { "Middle Eastern": 0.9, "Dark Ambient": 0.8, "Avant-garde": 0.7 },
                                "progression_id": "hatred_002"
                            },
                            {
                                "chords": ["♭II", "III", "i", "♭VII"],
                                "genres": { "World Fusion": 0.8, "Film Score": 0.8, "Ethnic": 0.9 },
                                "progression_id": "hatred_003"
                            }
                        ]
                    }
                }
            },
            "Fear": {
                "mode": "Locrian",
                "description": "Tense, anxious emotions with unresolved harmony",
                "sub_emotions": {
                    "Anxiety": {
                        "mode": "Locrian / Diminished",
                        "description": "Revolving through diminished chords and tense intervals amplifies an unresolved tension, keeping the listener on edge—much like anxiety.",
                        "progression_pool": [
                            {
                                "chords": ["i°", "ii°", "i°"],
                                "genres": { "Horror Score": 0.9, "Ambient": 0.8, "Experimental": 0.7 },
                                "progression_id": "anxiety_001"
                            },
                            {
                                "chords": ["i°", "♭v°", "i°", "♭ii°"],
                                "genres": { "Thriller": 0.9, "Dark Ambient": 0.8, "Soundscape": 0.7 },
                                "progression_id": "anxiety_002"
                            },
                            {
                                "chords": ["i°", "iv°", "♭vii°", "i°"],
                                "genres": { "Modern Classical": 0.8, "Atonal": 0.9, "Avant-garde": 0.8 },
                                "progression_id": "anxiety_003"
                            }
                        ]
                    },
                    "Dread": {
                        "mode": "Harmonic Minor",
                        "description": "The harmonic minor scale's augmented second adds dramatic, uneasy gravity, fitting the slow-burning, ominous tension of dread.",
                        "progression_pool": [
                            {
                                "chords": ["i", "iv", "♭II", "V7", "i"],
                                "genres": { "Horror": 0.9, "Classical": 0.8, "Gothic": 0.7 },
                                "progression_id": "dread_001"
                            },
                            {
                                "chords": ["i", "♭VI", "♭II", "V7"],
                                "genres": { "Film Score": 0.9, "Orchestral": 0.8, "Suspense": 0.8 },
                                "progression_id": "dread_002"
                            },
                            {
                                "chords": ["♭II", "V7", "i", "iv"],
                                "genres": { "Baroque": 0.8, "Neo-Classical": 0.8, "Chamber": 0.7 },
                                "progression_id": "dread_003"
                            }
                        ]
                    },
                    "Suspense": {
                        "mode": "Aeolian",
                        "description": "Combining minor and major chords without resolving (e.g. i–VI–iv–V) sustains tension and keeps the listener on edge, perfect for suspenseful scenes.",
                        "progression_pool": [
                            {
                                "chords": ["i", "♭VI", "iv", "V"],
                                "genres": { "Thriller Score": 0.9, "Cinematic": 0.8, "TV Drama": 0.8 },
                                "progression_id": "suspense_001"
                            },
                            {
                                "chords": ["i", "♭VII", "♭VI", "V"],
                                "genres": { "Mystery": 0.9, "Film Noir": 0.8, "Crime": 0.8 },
                                "progression_id": "suspense_002"
                            },
                            {
                                "chords": ["i", "iv", "V", "♭VI"],
                                "genres": { "Action": 0.8, "Adventure": 0.8, "Spy": 0.9 },
                                "progression_id": "suspense_003"
                            }
                        ]
                    }
                }
            }
        },
        "parser_rules": {
            "emotion_keywords": {
                "Joy": {
                    "primary": ["happy", "joy", "joyful", "elated", "cheerful", "uplifted", "bright", "celebratory"],
                    "sub_emotion_keywords": {
                        "Excitement": ["excited", "thrilled", "energetic", "pumped", "exhilarated", "animated"],
                        "Contentment": ["content", "satisfied", "peaceful", "serene", "calm", "relaxed"],
                        "Pride": ["proud", "accomplished", "triumphant", "confident", "victorious", "successful"],
                        "Romantic Longing": ["romantic", "longing", "yearning", "tender", "loving", "affectionate"]
                    }
                },
                "Sadness": {
                    "primary": ["sad", "depressed", "grieving", "blue", "mournful", "melancholy", "sorrowful"],
                    "sub_emotion_keywords": {
                        "Melancholy": ["melancholic", "bittersweet", "wistful", "pensive", "reflective"],
                        "Longing": ["yearning", "longing", "aching", "craving", "pining"],
                        "Nostalgia": ["nostalgic", "reminiscent", "wistful", "sentimental", "remembering"],
                        "Despair": ["despairing", "hopeless", "devastated", "crushed", "broken"],
                        "Loneliness": ["lonely", "isolated", "alone", "abandoned", "solitary"]
                    }
                },
                "Anger": {
                    "primary": ["angry", "furious", "frustrated", "pissed", "irritated", "rage", "aggressive"],
                    "sub_emotion_keywords": {
                        "Rage": ["raging", "furious", "livid", "enraged", "seething", "violent"],
                        "Resentment": ["resentful", "bitter", "grudging", "spiteful", "vindictive"],
                        "Hatred": ["hate", "loathe", "despise", "abhor", "detest", "revile"]
                    }
                },
                "Fear": {
                    "primary": ["afraid", "scared", "anxious", "nervous", "terrified", "worried", "tense"],
                    "sub_emotion_keywords": {
                        "Anxiety": ["anxious", "worried", "nervous", "uneasy", "apprehensive", "restless"],
                        "Dread": ["dreading", "ominous", "foreboding", "apprehensive", "fearful"],
                        "Suspense": ["suspenseful", "tense", "on edge", "uncertain", "anticipating"]
                    }
                }
            }
        }
    }
    
    return database

if __name__ == "__main__":
    print("🎼 Generating expanded emotion progression database...")
    
    # Generate the database
    db = create_expanded_database()
    
    # Write to file
    with open("emotion_progression_database.json", "w", encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Database created with {len(db['emotions'])} main emotions")
    
    # Count sub-emotions
    total_sub_emotions = 0
    for emotion_name, emotion_data in db['emotions'].items():
        if 'sub_emotions' in emotion_data:
            sub_count = len(emotion_data['sub_emotions'])
            total_sub_emotions += sub_count
            print(f"   📋 {emotion_name}: {sub_count} sub-emotions")
    
    print(f"✅ Total sub-emotions: {total_sub_emotions}")
    print("✅ Database file written to: emotion_progression_database.json") 