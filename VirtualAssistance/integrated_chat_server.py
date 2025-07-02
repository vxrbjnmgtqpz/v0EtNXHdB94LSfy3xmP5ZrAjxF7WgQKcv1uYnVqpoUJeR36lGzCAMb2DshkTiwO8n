#!/usr/bin/env python3
"""
Integrated Music Chat Server
Combines all three music generation models into a unified chatbot interface
"""

from flask import Flask, request, jsonify, Response, send_from_directory, session
from flask_cors import CORS
import json
import time
import os
import re
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import our three models
from chord_progression_model import ChordProgressionModel
from individual_chord_model import IndividualChordModel
import sys
sys.path.append('./TheoryEngine')
from enhanced_solfege_theory_engine import EnhancedSolfegeTheoryEngine

app = Flask(__name__)
app.secret_key = 'music_theory_chat_secret_key'  # For session management
CORS(app, supports_credentials=True)

@dataclass
class ConversationContext:
    """Stores conversation context for follow-up requests"""
    last_response: Dict[str, Any]
    last_progression: List[str]
    last_emotion: str
    last_style: str
    last_mode: str
    session_id: str
    timestamp: float

class ConversationMemory:
    """Manages conversation context across requests"""
    
    def __init__(self):
        self.sessions = {}
        self.max_session_age = 3600  # 1 hour
    
    def get_session_id(self):
        """Get or create session ID"""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        return session['session_id']
    
    def store_context(self, session_id: str, response_data: Dict[str, Any]):
        """Store conversation context"""
        print(f"DEBUG: Storing context for session {session_id}")
        print(f"DEBUG: Response data keys: {list(response_data.keys())}")
        print(f"DEBUG: Chords in response: {response_data.get('chords', [])}")
        
        context = ConversationContext(
            last_response=response_data,
            last_progression=response_data.get('chords', []),
            last_emotion=response_data.get('primary_emotion', ''),
            last_style=response_data.get('style', ''),
            last_mode=response_data.get('mode', ''),
            session_id=session_id,
            timestamp=time.time()
        )
        self.sessions[session_id] = context
        print(f"DEBUG: Stored progression: {context.last_progression}")
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation context"""
        print(f"DEBUG: Getting context for session {session_id}")
        print(f"DEBUG: Available sessions: {list(self.sessions.keys())}")
        
        context = self.sessions.get(session_id)
        if context and (time.time() - context.timestamp) < self.max_session_age:
            print(f"DEBUG: Found valid context with progression: {context.last_progression}")
            return context
        elif context:
            # Clean up expired session
            print(f"DEBUG: Context expired, cleaning up")
            del self.sessions[session_id]
        else:
            print(f"DEBUG: No context found for session")
        return None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, ctx in self.sessions.items()
            if (current_time - ctx.timestamp) > self.max_session_age
        ]
        for sid in expired_sessions:
            del self.sessions[sid]

@dataclass
class UserIntent:
    """Represents the classified intent of a user message"""
    primary_intent: str
    confidence: float
    extracted_params: Dict[str, Any]
    suggested_models: List[str]

class IntentClassifier:
    """Classifies user intents and determines which models to use"""
    
    def __init__(self):
        self.intent_patterns = {
            "emotional_progression": {
                "patterns": [
                    r"i\s+(feel|am\s+feeling|want\s+something)",
                    r"(happy|sad|angry|romantic|nostalgic|excited|melancholy|joyful|peaceful|dramatic|energetic|soulful).*progression",
                    r"make.*sound|want.*that.*sounds|create.*mood",
                    r"feeling\s+(like|very|really|quite|somewhat)",
                    r"(uplifting|depressing|calming|aggressive|tender|mysterious|bright|dark)"
                ],
                "models": ["progression", "individual", "theory"],
                "params": ["emotion", "style", "length"]
            },
            "individual_chord": {
                "patterns": [
                    r"what\s+chord|which\s+chord|chord\s+for|chord\s+that",
                    r"single\s+chord|one\s+chord|individual\s+chord",
                    r"best\s+chord\s+for|chord\s+represents"
                ],
                "models": ["individual", "theory"],
                "params": ["emotion", "context"]
            },
            "individual_analysis": {
                "patterns": [
                    r"show.*me.*individual.*chord.*emotions?",
                    r"individual.*chord.*emotions?",
                    r"analyze.*each.*chord",
                    r"emotions?.*for.*each.*chord",
                    r"breakdown.*of.*chord.*emotions?",
                    r"ask.*for.*details"
                ],
                "models": ["individual"],
                "params": ["progression_context"]
            },
            "theory_request": {
                "patterns": [
                    r"(jazz|blues|classical|rock|pop|folk|rnb|cinematic)\s+(progression|chord|style)",
                    r"(ionian|dorian|phrygian|lydian|mixolydian|aeolian|locrian)\s+(mode|scale)",
                    r"theory|analysis|explain|why|how.*work",
                    r"generate.*in\s+(jazz|blues|classical|rock|pop|folk)",
                    r"show.*me.*in\s+(major|minor|dorian|phrygian)"
                ],
                "models": ["theory", "progression"],
                "params": ["style", "mode", "length", "analysis_type"]
            },
            "comparison": {
                "patterns": [
                    r"compare|versus|vs|different.*style",
                    r"how.*sound.*in|what.*if.*were",
                    r"show.*me.*different|try.*in.*style",
                    r"(jazz|blues|classical).*vs.*(jazz|blues|classical)"
                ],
                "models": ["theory", "progression", "individual"],
                "params": ["styles", "emotion", "comparison_type"]
            },
            "educational": {
                "patterns": [
                    r"explain|why.*does|how.*work|what.*makes",
                    r"teach.*me|learn.*about|understand",
                    r"difference.*between|what.*is.*the",
                    r"music.*theory|harmonic.*function"
                ],
                "models": ["theory", "individual"],
                "params": ["concept", "depth_level"]
            }
        }
        
        self.emotion_keywords = [
            "happy", "sad", "angry", "fearful", "disgusted", "surprised", 
            "trusting", "anticipation", "joy", "sadness", "fear", "anger",
            "disgust", "surprise", "trust", "shame", "love", "envy",
            "romantic", "melancholy", "energetic", "peaceful", "dramatic",
            "nostalgic", "uplifting", "mysterious", "bright", "dark",
            "tender", "aggressive", "calming", "exciting", "soulful"
        ]
        
        self.style_keywords = [
            "jazz", "blues", "classical", "pop", "rock", "folk", "rnb", "cinematic"
        ]
        
        self.mode_keywords = [
            "ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian",
            "major", "minor"
        ]
    
    def classify(self, text: str, context: Optional[ConversationContext] = None) -> UserIntent:
        """Classify user intent and extract parameters with conversation context"""
        text_lower = text.lower().strip()
        
        # Score each intent type
        intent_scores = {}
        extracted_params = {}
        
        for intent_type, intent_data in self.intent_patterns.items():
            score = 0
            for pattern in intent_data["patterns"]:
                if re.search(pattern, text_lower):
                    score += 1
            intent_scores[intent_type] = score
        
        # Boost individual_analysis if context suggests it and patterns match
        if context and context.last_progression and len(context.last_progression) > 0:
            if intent_scores.get("individual_analysis", 0) > 0:
                intent_scores["individual_analysis"] += 2  # Boost this intent
                extracted_params["context_progression"] = context.last_progression
                extracted_params["context_emotion"] = context.last_emotion
        
        # Determine primary intent
        if not intent_scores or max(intent_scores.values()) == 0:
            primary_intent = "emotional_progression"  # Default fallback
            confidence = 0.3
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[primary_intent] / 3.0, 1.0)
        
        # Extract parameters
        extracted_params.update(self._extract_parameters(text_lower))
        
        # Get suggested models
        suggested_models = self.intent_patterns[primary_intent]["models"]
        
        return UserIntent(
            primary_intent=primary_intent,
            confidence=confidence,
            extracted_params=extracted_params,
            suggested_models=suggested_models
        )
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract relevant parameters from user text"""
        params = {}
        
        # Extract emotions
        found_emotions = []
        for emotion in self.emotion_keywords:
            if emotion in text:
                found_emotions.append(emotion)
        if found_emotions:
            params["emotions"] = found_emotions
            params["primary_emotion"] = found_emotions[0]
        
        # Extract styles
        found_styles = []
        for style in self.style_keywords:
            if style in text:
                found_styles.append(style.title())
        if found_styles:
            params["styles"] = found_styles
            params["primary_style"] = found_styles[0]
        
        # Extract modes
        found_modes = []
        for mode in self.mode_keywords:
            if mode in text:
                if mode == "major":
                    found_modes.append("Ionian")
                elif mode == "minor":
                    found_modes.append("Aeolian") 
                else:
                    found_modes.append(mode.title())
        if found_modes:
            params["modes"] = found_modes
            params["primary_mode"] = found_modes[0]
        
        # Extract length preferences
        length_match = re.search(r'(\d+)\s*(chord|note)', text)
        if length_match:
            params["length"] = int(length_match.group(1))
        
        return params

class ResponseSynthesizer:
    """Combines results from multiple models into unified responses"""
    
    def __init__(self):
        self.response_templates = {
            "emotional_progression": self._synthesize_emotional_progression,
            "individual_chord": self._synthesize_single_chord,
            "individual_analysis": self._synthesize_individual_analysis,
            "theory_request": self._synthesize_theory_request,
            "comparison": self._synthesize_comparison,
            "educational": self._synthesize_educational
        }
    
    def synthesize(self, intent: UserIntent, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize a unified response from model results"""
        synthesizer = self.response_templates.get(intent.primary_intent, self._synthesize_default)
        return synthesizer(intent, model_results)
    
    def _synthesize_emotional_progression(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response for emotional progression requests"""
        progression_result = results.get("progression", {})
        individual_result = results.get("individual", {})
        theory_result = results.get("theory", {})
        
        # Build primary response
        chords = progression_result.get("chords", [])
        emotions = progression_result.get("emotion_weights", {})
        
        # Create comprehensive message
        message_parts = []
        
        if chords:
            message_parts.append(f"🎼 **{' → '.join(chords)}**")
        
        if emotions:
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
            emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions])
            message_parts.append(f"🎭 Emotions: {emotion_text}")
        
        if progression_result.get("primary_mode"):
            message_parts.append(f"🎵 Mode: {progression_result['primary_mode']}")
        
        # Add individual chord insights if available
        individual_available = False
        if isinstance(individual_result, list) and len(individual_result) > 0:
            individual_available = True
        elif isinstance(individual_result, dict) and not individual_result.get("error"):
            individual_available = True
            
        if individual_available:
            message_parts.append("🔍 Individual chord emotions available - ask for details!")
        
        # Add theory context if available
        if theory_result and theory_result.get("style_alternatives"):
            alt_count = len(theory_result["style_alternatives"])
            message_parts.append(f"🎶 {alt_count} style alternatives available")
        
        return {
            "message": "\n\n".join(message_parts),
            "chords": chords,
            "emotions": emotions,
            "primary_result": progression_result,
            "alternatives": theory_result.get("style_alternatives", {}),
            "individual_analysis": individual_result,
            "suggestions": self._generate_suggestions(intent, "emotional_progression")
        }
    
    def _synthesize_single_chord(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response for single chord requests"""
        individual_result = results.get("individual", {})
        theory_result = results.get("theory", {})
        
        # Handle individual chord model result (which returns a list)
        chord_data = None
        if isinstance(individual_result, list) and len(individual_result) > 0:
            chord_data = individual_result[0]  # Take the first result
        elif isinstance(individual_result, dict) and not individual_result.get("error"):
            chord_data = individual_result.get("chord", individual_result)
        
        message_parts = []
        if chord_data:
            symbol = chord_data.get("chord_symbol", chord_data.get("symbol", "Unknown"))
            roman = chord_data.get("roman_numeral", "N/A")
            message_parts.append(f"🎵 **{symbol} ({roman})**")
            
            if chord_data.get("emotion_weights"):
                emotions = chord_data["emotion_weights"]
                top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions if v > 0])
                if emotion_text:
                    message_parts.append(f"🎭 Primary emotions: {emotion_text}")
            
            if chord_data.get("mode_context") or chord_data.get("style_context"):
                mode = chord_data.get("mode_context", "Unknown")
                style = chord_data.get("style_context", "Unknown")
                message_parts.append(f"🎼 Context: {mode} ({style})")
                
            if chord_data.get("emotional_score"):
                score = chord_data["emotional_score"]
                message_parts.append(f"🎯 Emotional fit: {score:.2f}")
        else:
            # Handle error case
            error_msg = individual_result.get("error", "No chord data available")
            message_parts.append(f"❌ {error_msg}")
        
        # Add theory alternatives
        if theory_result.get("style_alternatives"):
            alt_count = len(theory_result["style_alternatives"])
            message_parts.append(f"🎶 {alt_count} style alternatives available")
        
        return {
            "message": "\n\n".join(message_parts),
            "chord": chord_data or {},
            "primary_result": individual_result,
            "theory_context": theory_result,
            "suggestions": self._generate_suggestions(intent, "single_chord")
        }
    
    def _synthesize_theory_request(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response for music theory requests"""
        theory_result = results.get("theory", {})
        progression_result = results.get("progression", {})
        
        message_parts = []
        
        if theory_result.get("progression"):
            chords = theory_result["progression"]
            message_parts.append(f"🎼 **{' → '.join(chords)}**")
        
        style = intent.extracted_params.get("primary_style", "Classical")
        mode = intent.extracted_params.get("primary_mode", "Ionian")
        message_parts.append(f"🎵 Style: {style} | Mode: {mode}")
        
        if theory_result.get("analysis"):
            message_parts.append("🔍 Theoretical analysis available")
        
        if progression_result.get("emotion_weights"):
            emotions = progression_result["emotion_weights"]
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            message_parts.append(f"🎭 Emotional character: {top_emotion[0]} ({top_emotion[1]:.2f})")
        
        return {
            "message": "\n\n".join(message_parts),
            "chords": theory_result.get("progression", []),
            "style": style,
            "mode": mode,
            "primary_result": theory_result,
            "emotional_context": progression_result,
            "suggestions": self._generate_suggestions(intent, "theory_request")
        }
    
    def _synthesize_comparison(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response for comparison requests"""
        theory_result = results.get("theory", {})
        progression_result = results.get("progression", {})
        
        message_parts = ["🎼 **Style Comparison**"]
        
        comparisons = theory_result.get("style_comparison", {})
        if comparisons:
            for style, progression in comparisons.items():
                if progression:
                    message_parts.append(f"• **{style}**: {' → '.join(progression)}")
        
        # Add emotional context if available
        if progression_result.get("emotion_weights"):
            emotions = progression_result["emotion_weights"]
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
            emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions])
            message_parts.append(f"\n🎭 Emotional foundation: {emotion_text}")
        
        return {
            "message": "\n\n".join(message_parts),
            "comparisons": comparisons,
            "primary_result": theory_result,
            "emotional_context": progression_result,
            "suggestions": self._generate_suggestions(intent, "comparison")
        }
    
    def _synthesize_educational(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response for educational requests"""
        theory_result = results.get("theory", {})
        individual_result = results.get("individual", {})
        
        message_parts = ["📚 **Music Theory Explanation**"]
        
        # Add theory explanation if available
        if theory_result.get("explanation"):
            message_parts.append(theory_result["explanation"])
        
        # Add practical examples
        if theory_result.get("examples"):
            message_parts.append("🎵 **Examples:**")
            for example in theory_result["examples"]:
                message_parts.append(f"• {example}")
        
        return {
            "message": "\n\n".join(message_parts),
            "primary_result": theory_result,
            "supporting_data": individual_result,
            "suggestions": self._generate_suggestions(intent, "educational")
        }
    
    def _synthesize_individual_analysis(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response for individual chord analysis of a progression"""
        individual_results = results.get("individual", [])
        progression = intent.extracted_params.get("context_progression", [])
        
        if not progression:
            return {
                "message": "❌ No progression found in conversation context. Please generate a progression first.",
                "suggestions": ["Generate a new progression", "Ask for a specific chord"]
            }
        
        message_parts = [f"🎼 **Individual Chord Analysis for: {' → '.join(progression)}**"]
        
        # Process each chord analysis
        if isinstance(individual_results, list) and len(individual_results) >= len(progression):
            for i, (chord, analysis) in enumerate(zip(progression, individual_results)):
                message_parts.append(f"\\n**{i+1}. {chord}**")
                
                # Handle errors in analysis
                if analysis.get("error"):
                    message_parts.append(f"  ❌ Error: {analysis['error']}")
                    continue
                
                # Display emotion weights
                if analysis.get("emotion_weights"):
                    emotions = analysis["emotion_weights"]
                    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions if v > 0])
                    if emotion_text:
                        message_parts.append(f"  🎭 Emotions: {emotion_text}")
                    else:
                        message_parts.append(f"  🎭 Neutral emotional content")
                
                # Display harmonic context
                if analysis.get("mode_context") or analysis.get("style_context"):
                    mode = analysis.get("mode_context", "Unknown")
                    style = analysis.get("style_context", "Unknown")
                    message_parts.append(f"  🎼 Context: {mode} ({style})")
                    
                # Display emotional fit score
                if analysis.get("emotional_score") is not None:
                    message_parts.append(f"  🎯 Emotional fit: {analysis['emotional_score']:.2f}")
                
                # Display chord notes if available
                if analysis.get("notes"):
                    message_parts.append(f"  🎵 Notes: {analysis['notes']}")
        else:
            # Fallback if we don't have proper analysis
            message_parts.append("\\n🔍 **Basic chord breakdown:**")
            for i, chord in enumerate(progression):
                message_parts.append(f"\\n**{i+1}. {chord}**")
                message_parts.append(f"  🎵 Roman numeral chord")
                message_parts.append(f"  🎼 Part of progression context")
        
        return {
            "message": "\\n".join(message_parts),
            "chords": progression,
            "individual_analysis": individual_results,
            "progression_breakdown": True,
            "suggestions": [
                "Play this progression",
                "Try in a different style", 
                "Compare with other progressions",
                "Explain the harmonic functions"
            ]
        }
    
    def _synthesize_default(self, intent: UserIntent, results: Dict[str, Any]) -> Dict[str, Any]:
        """Default synthesizer for unrecognized intents"""
        # Fall back to emotional progression
        return self._synthesize_emotional_progression(intent, results)
    
    def _generate_suggestions(self, intent: UserIntent, response_type: str) -> List[str]:
        """Generate contextual follow-up suggestions"""
        base_suggestions = {
            "emotional_progression": [
                "Try this in a different style",
                "Show me individual chord emotions", 
                "Compare across musical styles",
                "Explain the music theory"
            ],
            "single_chord": [
                "Show me similar chords",
                "Create a progression with this chord",
                "Explain why this chord fits",
                "Try in different modes"
            ],
            "theory_request": [
                "Add emotional context",
                "Compare with other styles",
                "Show me variations",
                "Explain the harmonic functions"
            ],
            "comparison": [
                "Try a different emotion",
                "Add more styles to compare",
                "Explain the differences",
                "Show me individual chord analysis"
            ],
            "educational": [
                "Show me practical examples",
                "Try this concept in practice",
                "Compare with other concepts",
                "Test my understanding"
            ]
        }
        
        suggestions = base_suggestions.get(response_type, base_suggestions["emotional_progression"])
        
        # Customize based on extracted parameters
        if intent.extracted_params.get("styles"):
            suggestions.append("Try other musical styles")
        
        if intent.extracted_params.get("emotions"):
            suggestions.append("Explore different emotions")
        
        return suggestions[:4]  # Limit to 4 suggestions

class IntegratedMusicChatServer:
    """Main server class that coordinates all three models"""
    
    def __init__(self):
        print("Initializing Integrated Music Chat Server...")
        
        # Initialize models
        print("Loading Chord Progression Model...")
        self.progression_model = ChordProgressionModel()
        
        print("Loading Individual Chord Model...")
        self.individual_model = IndividualChordModel()
        
        print("Loading Enhanced Solfege Theory Engine...")
        self.theory_engine = EnhancedSolfegeTheoryEngine()
        
        # Initialize support systems
        self.intent_classifier = IntentClassifier()
        self.response_synthesizer = ResponseSynthesizer()
        self.conversation_memory = ConversationMemory()
        
        print("✓ All models loaded successfully!")
    
    def process_message(self, user_input: str, context: Dict[str, Any] = None, session_id: str = None) -> Dict[str, Any]:
        """Process a user message and return integrated response with conversation context"""
        try:
            # Get conversation context
            conversation_context = None
            if session_id:
                conversation_context = self.conversation_memory.get_context(session_id)
            
            # Classify intent with context
            intent = self.intent_classifier.classify(user_input, conversation_context)
            
            # Route to appropriate models
            model_results = {}
            
            # For individual analysis, we might need to analyze each chord in the progression
            if intent.primary_intent == "individual_analysis" and conversation_context:
                progression = conversation_context.last_progression
                if progression:
                    # Generate individual chord analysis for each chord in the progression
                    individual_analyses = []
                    for chord in progression:
                        try:
                            # Use the chord symbol as emotion prompt to get detailed analysis
                            chord_analysis = self.individual_model.generate_chord_from_prompt(f"chord {chord}")
                            if isinstance(chord_analysis, list) and len(chord_analysis) > 0:
                                individual_analyses.append(chord_analysis[0])
                            elif isinstance(chord_analysis, dict):
                                individual_analyses.append(chord_analysis)
                            else:
                                # Fallback: create basic analysis
                                individual_analyses.append({
                                    "chord_symbol": chord,
                                    "emotion_weights": {},
                                    "mode_context": "Unknown",
                                    "style_context": "Unknown",
                                    "emotional_score": 0.0
                                })
                        except Exception as e:
                            print(f"Error analyzing chord {chord}: {e}")
                            individual_analyses.append({
                                "chord_symbol": chord,
                                "error": str(e),
                                "emotion_weights": {},
                                "mode_context": "Error",
                                "style_context": "Error",
                                "emotional_score": 0.0
                            })
                    model_results["individual"] = individual_analyses
            elif "progression" in intent.suggested_models:
                model_results["progression"] = self._call_progression_model(user_input, intent)
            
            if "individual" in intent.suggested_models and intent.primary_intent != "individual_analysis":
                model_results["individual"] = self._call_individual_model(user_input, intent)
            
            if "theory" in intent.suggested_models:
                model_results["theory"] = self._call_theory_engine(user_input, intent)
            
            # Synthesize unified response
            response = self.response_synthesizer.synthesize(intent, model_results)
            
            # Add metadata
            response.update({
                "intent": intent.primary_intent,
                "confidence": intent.confidence,
                "models_used": intent.suggested_models,
                "timestamp": time.time()
            })
            
            # Store conversation context for follow-up requests
            if session_id:
                self.conversation_memory.store_context(session_id, response)
            
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "I encountered an error processing your request. Please try again.",
                "timestamp": time.time()
            }
    
    def _call_progression_model(self, user_input: str, intent: UserIntent) -> Dict[str, Any]:
        """Call the chord progression model"""
        try:
            # Use primary emotion or default to the input text
            emotion_text = intent.extracted_params.get("primary_emotion", user_input)
            style_preference = intent.extracted_params.get("primary_style", "Pop")
            
            results = self.progression_model.generate_from_prompt(
                emotion_text, 
                genre_preference=style_preference, 
                num_progressions=1
            )
            
            if results:
                return results[0]
            else:
                return {"error": "No progression generated"}
                
        except Exception as e:
            return {"error": f"Progression model error: {str(e)}"}
    
    def _call_individual_model(self, user_input: str, intent: UserIntent) -> Dict[str, Any]:
        """Call the individual chord model"""
        try:
            # Use primary emotion or parse from text
            emotion_text = intent.extracted_params.get("primary_emotion", user_input)
            
            result = self.individual_model.generate_chord_from_prompt(emotion_text)
            return result
            
        except Exception as e:
            return {"error": f"Individual chord model error: {str(e)}"}
    
    def _call_theory_engine(self, user_input: str, intent: UserIntent) -> Dict[str, Any]:
        """Call the enhanced solfege theory engine"""
        try:
            style = intent.extracted_params.get("primary_style", "Classical")
            mode = intent.extracted_params.get("primary_mode", "Ionian")
            length = intent.extracted_params.get("length", 4)
            
            if intent.primary_intent == "comparison":
                # Generate style comparison
                comparison = self.theory_engine.compare_style_progressions(mode, length)
                return {"style_comparison": comparison}
            
            elif intent.primary_intent == "theory_request":
                # Generate specific style progression
                progression = self.theory_engine.generate_legal_progression(style, mode, length)
                return {"progression": progression, "style": style, "mode": mode}
            
            else:
                # Generate style alternatives for emotion
                emotion = intent.extracted_params.get("primary_emotion", "happy")
                comparison = self.theory_engine.compare_style_progressions(mode, length)
                return {"style_alternatives": comparison}
                
        except Exception as e:
            return {"error": f"Theory engine error: {str(e)}"}

# Initialize the integrated server
print("Starting Integrated Music Chat Server...")
integrated_server = IntegratedMusicChatServer()

# Flask routes
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': True})

@app.route('/chat/integrated', methods=['POST'])
def chat_integrated():
    """Main integrated chat endpoint with enhanced validation"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_message = data.get('message', '').strip()
        context = data.get('context', {})
        
        # Input validation
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        if len(user_message) > 1000:  # Reasonable limit
            return jsonify({'error': 'Message too long (max 1000 characters)'}), 400
            
        # Basic sanitization - remove potential HTML/script tags
        import re
        user_message = re.sub(r'<[^>]*>', '', user_message)
        user_message = re.sub(r'[^\w\s\-\.\,\!\?\:\;\(\)\'\"#@&+/]', '', user_message)
        
        if not user_message.strip():
            return jsonify({'error': 'Message contains no valid content'}), 400
        
        # Process with integrated server and session management
        session_id = integrated_server.conversation_memory.get_session_id()
        response = integrated_server.process_message(user_message, context, session_id)
        
        # Ensure response has required fields
        if 'error' not in response:
            response.setdefault('message', 'No response generated')
            response.setdefault('intent', 'unknown')
            response.setdefault('confidence', 0.0)
            response.setdefault('models_used', [])
            response.setdefault('suggestions', [])
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if app.debug else 'Contact administrator'
        }), 500

@app.route('/chat/analyze', methods=['POST'])
def analyze_progression():
    """Analyze a specific chord progression"""
    try:
        data = request.json
        progression = data.get('progression', [])
        mode = data.get('mode', 'Ionian')
        
        if not progression:
            return jsonify({'error': 'No progression provided'}), 400
        
        # Use theory engine for analysis
        analysis = integrated_server.theory_engine.analyze_progression(progression, mode)
        
        return jsonify({
            'analysis': analysis,
            'progression': progression,
            'mode': mode,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_chat():
    """Serve the chat interface"""
    return send_from_directory('.', 'chord_chat.html')

if __name__ == '__main__':
    print("🎼 Integrated Music Chat Server is ready!")
    print("Available endpoints:")
    print("  GET  /              - Chat interface")
    print("  GET  /health        - Health check")
    print("  POST /chat/integrated - Integrated chat")
    print("  POST /chat/analyze   - Progression analysis")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5003)
