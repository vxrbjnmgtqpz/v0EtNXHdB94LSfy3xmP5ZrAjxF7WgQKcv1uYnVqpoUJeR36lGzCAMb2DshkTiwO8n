#!/usr/bin/env python3

# Simple test to debug the integration issues

try:
    print("Testing individual chord integration...")
    from neural_progression_analyzer import NeuralProgressionAnalyzer
    analyzer = NeuralProgressionAnalyzer()
    
    # Check if the method exists
    if hasattr(analyzer, '_get_individual_chord_data'):
        print("✅ _get_individual_chord_data method exists")
        # Test the method
        try:
            chord_data, cd_value = analyzer._get_individual_chord_data("I")
            print(f"✅ Method works: chord_data={type(chord_data)}, cd_value={cd_value}")
        except Exception as e:
            print(f"❌ Method error: {e}")
    else:
        print("❌ _get_individual_chord_data method missing")
        print(f"Available methods: {[m for m in dir(analyzer) if not m.startswith('__')]}")

except Exception as e:
    print(f"❌ Error importing/testing neural analyzer: {e}")

try:
    print("\nTesting integration layer...")
    from chord_progression_model import ChordProgressionModel
    model = ChordProgressionModel()
    
    print("✅ Model created")
    
    # Test integration layer
    integration_result = model.emotion_integration_layer.process_emotion_input("happy")
    print(f"✅ Integration layer works: {list(integration_result.keys())}")
    
    # Test simple generation
    results = model.generate_from_prompt("happy")
    print(f"✅ Generation works: got {len(results)} results")
    
except Exception as e:
    print(f"❌ Error testing main model: {e}")
    import traceback
    traceback.print_exc() 