#!/usr/bin/env python3
"""
Test the chat server's access to Transcendence chord progressions
"""

import requests
import json
import sys

def test_server_transcendence():
    """Test server endpoints for Transcendence emotions"""
    
    base_url = "http://localhost:5004"
    
    try:
        # Test 1: Check if server is running
        print("🔍 Testing server connection...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
        print("✅ Server is running")
        
        # Test 2: Try to access transcendence via chat
        print("\n🔍 Testing Transcendence access via chat...")
        test_request = {
            "message": "Generate a chord progression for transcendence with lucid wonder",
            "style": "Cinematic"
        }
        response = requests.post(f"{base_url}/chat/integrated", json=test_request, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to get chat response: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        print("✅ Successfully got chat response for Transcendence:")
        print(f"   Response: {result.get('response', 'No response')[:200]}...")
        
        # Test 3: Try chord generation
        print("\n🔍 Testing chord generation...")
        chord_request = {
            "emotion": "transcendence",
            "sub_emotion": "cosmic_unity",
            "complexity": "intermediate"
        }
        
        response = requests.post(
            f"{base_url}/chord/generate", 
            json=chord_request,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to generate chord: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        print("✅ Successfully generated chord for Transcendence:")
        print(f"   Chord: {result.get('chord', 'No chord')}")
        print(f"   Description: {result.get('description', 'No description')}")
        
        # Test 4: Try progression analysis 
        print("\n🔍 Testing progression analysis...")
        analysis_request = {
            "chords": ["I", "bVII", "IV", "I"],
            "key": "C"
        }
        
        response = requests.post(
            f"{base_url}/progression/analyze",
            json=analysis_request,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to analyze progression: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        print("✅ Successfully analyzed progression:")
        print(f"   Analysis: {result.get('analysis', 'No analysis')[:100]}...")
        
        print("\n🎉 ALL TRANSCENDENCE SERVER TESTS PASSED!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on port 5004?")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_server_transcendence()
    sys.exit(0 if success else 1)
