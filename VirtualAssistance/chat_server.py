#!/usr/bin/env python3
"""
Simple Flask server for the chord progression chat app
"""

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import json
import time
import os
from chord_progression_model import ChordProgressionModel

app = Flask(__name__)
CORS(app)  # Enable CORS for the HTML file

# Initialize the model once
print("Loading chord progression model...")
model = ChordProgressionModel()
print("✓ Model loaded!")

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return chord progressions"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate chord progression
        results = model.generate_from_prompt(user_message, genre_preference="Pop", num_progressions=1)
        result = results[0]
        
        # Format response
        top_emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:2]
        emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions])
        
        response = {
            'message': f"🎼 **{' → '.join(result['chords'])}**\n\n🎭 Emotions: {emotion_text}\n🎵 Mode: {result['primary_mode']}",
            'chords': result['chords'],
            'emotions': dict(top_emotions),
            'mode': result['primary_mode'],
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream', methods=['POST'])
def stream_chat():
    """Server-sent events streaming endpoint"""
    def generate():
        try:
            data = request.json
            user_message = data.get('message', '').strip()
            
            if not user_message:
                yield f"data: {json.dumps({'error': 'No message provided'})}\n\n"
                return
            
            # Stream thinking message
            yield f"data: {json.dumps({'type': 'thinking', 'message': '🎵 Analyzing emotions...'})}\n\n"
            time.sleep(0.5)
            
            # Generate chord progression
            results = model.generate_from_prompt(user_message, genre_preference="Pop", num_progressions=1)
            result = results[0]
            
            # Stream partial results
            yield f"data: {json.dumps({'type': 'partial', 'message': '🎭 Detected emotions...'})}\n\n"
            time.sleep(0.3)
            
            # Stream final result
            top_emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:2]
            emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions])
            
            response = {
                'type': 'complete',
                'message': f"🎼 **{' → '.join(result['chords'])}**\n\n🎭 Emotions: {emotion_text}\n🎵 Mode: {result['primary_mode']}",
                'chords': result['chords'],
                'emotions': dict(top_emotions),
                'mode': result['primary_mode'],
                'timestamp': time.time()
            }
            
            yield f"data: {json.dumps(response)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/')
def index():
    """Serve the chat HTML file"""
    return send_from_directory('.', 'chord_chat.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    print("Starting chord progression chat server...")
    print("Server will be available at http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
