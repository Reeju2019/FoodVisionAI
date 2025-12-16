"""
Complete FoodVisionAI Pipeline Test

This script tests the entire 3-stage academic pipeline:
1. Stage 1: Recipe1M CNN ingredient detection
2. Stage 2: Gemini Pro dish identification
3. Stage 3: Nutrition database lookup

Usage:
    python test_full_pipeline.py <path_to_food_image.jpg>
"""

import sys
import os
from pathlib import Path


def check_prerequisites():
    """Check if all required files and dependencies are present."""
    print("=" * 70)
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 70)
    
    issues = []
    
    # Check model files
    model_dir = Path("foodvision_ai/models/recipe1m")
    required_files = [
        "recipe1m_best_model.pth",
        "recipe1m_ingredient_vocab.json",
        "recipe1m_deployment_info.json"
    ]
    
    print("\n📦 Checking Recipe1M model files...")
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file} - NOT FOUND")
            issues.append(f"Missing: {file_path}")
    
    # Check dependencies
    print("\n📚 Checking Python dependencies...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError:
        print(f"   ❌ PyTorch - NOT INSTALLED")
        issues.append("Install: pip install torch torchvision")
    
    try:
        import timm
        print(f"   ✅ timm")
    except ImportError:
        print(f"   ❌ timm - NOT INSTALLED")
        issues.append("Install: pip install timm")
    
    try:
        from PIL import Image
        print(f"   ✅ Pillow")
    except ImportError:
        print(f"   ❌ Pillow - NOT INSTALLED")
        issues.append("Install: pip install pillow")
    
    try:
        import google.generativeai as genai
        print(f"   ✅ Google Generative AI")
    except ImportError:
        print(f"   ⚠️  Google Generative AI - NOT INSTALLED (Stage 2 will use fallback)")
    
    # Check .env file
    print("\n⚙️  Checking configuration...")
    env_file = Path(".env")
    if env_file.exists():
        print(f"   ✅ .env file found")
        # Check for Gemini API key
        with open(env_file) as f:
            content = f.read()
            if "GEMINI_API_KEY" in content and "your-gemini-api-key-here" not in content:
                print(f"   ✅ Gemini API key configured")
            else:
                print(f"   ⚠️  Gemini API key not configured (Stage 2 will use fallback)")
    else:
        print(f"   ⚠️  .env file not found (using defaults)")
    
    print("\n" + "=" * 70)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Please fix these issues before running the pipeline.")
        return False
    else:
        print("✅ ALL PREREQUISITES MET!")
        return True


def test_stage1_only(image_path):
    """Test Stage 1: Recipe1M ingredient detection."""
    print("\n" + "=" * 70)
    print("🧪 TESTING STAGE 1: Recipe1M Ingredient Detection")
    print("=" * 70)
    
    try:
        from foodvision_ai.models.recipe1m_loader import load_recipe1m_model
        from PIL import Image
        
        print("\n📦 Loading Recipe1M model...")
        loader = load_recipe1m_model("foodvision_ai/models/recipe1m")
        
        print(f"\n📷 Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        print(f"   Image size: {image.size}")
        
        print(f"\n🔮 Running ingredient detection...")
        results = loader.predict(image, threshold=0.2, top_k=10)
        
        print(f"\n✅ STAGE 1 RESULTS:")
        print(f"   Detected {results['metadata']['num_predictions_threshold']} ingredients")
        
        print(f"\n🥘 Top 10 Ingredients:")
        for i, (ingredient, prob) in enumerate(zip(
            results['top_k_predictions']['ingredients'],
            results['top_k_predictions']['probabilities']
        ), 1):
            print(f"   {i:2d}. {ingredient:<30} {prob:>6.1%}")
        
        print(f"\n✅ Confident Predictions (≥20% confidence):")
        if results['threshold_predictions']['ingredients']:
            for ingredient, prob in zip(
                results['threshold_predictions']['ingredients'],
                results['threshold_predictions']['probabilities']
            ):
                print(f"   • {ingredient:<30} {prob:>6.1%}")
        else:
            print(f"   (No predictions above threshold)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ STAGE 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline(image_path):
    """Test the complete 3-stage pipeline."""
    print("\n" + "=" * 70)
    print("🚀 TESTING FULL 3-STAGE PIPELINE")
    print("=" * 70)
    
    try:
        from foodvision_ai.models.academic_pipeline import AcademicPipeline
        
        print("\n📦 Initializing Academic Pipeline...")
        pipeline = AcademicPipeline()
        
        print(f"\n📷 Processing image: {image_path}")
        print("\n" + "-" * 70)
        
        # Run full pipeline
        result = pipeline.process_image(image_path)
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 PIPELINE RESULTS")
        print("=" * 70)
        
        print("\n🔹 STAGE 1: Ingredient Detection")
        if 'stage1' in result:
            stage1 = result['stage1']
            print(f"   Model: {stage1.get('model', 'Unknown')}")
            print(f"   Ingredients detected: {len(stage1.get('ingredients', []))}")
            print(f"   Top ingredients:")
            for ing in stage1.get('ingredients', [])[:5]:
                print(f"      • {ing}")
        
        print("\n🔹 STAGE 2: Dish Identification")
        if 'stage2' in result:
            stage2 = result['stage2']
            print(f"   Dish: {stage2.get('dish_name', 'Unknown')}")
            print(f"   Description: {stage2.get('description', 'N/A')}")
            print(f"   Cuisine: {stage2.get('cuisine', 'Unknown')}")
        
        print("\n🔹 STAGE 3: Nutrition Information")
        if 'stage3' in result:
            stage3 = result['stage3']
            print(f"   Calories: {stage3.get('calories', 0)} kcal")
            print(f"   Protein: {stage3.get('protein', 0)}g")
            print(f"   Fat: {stage3.get('fat', 0)}g")
            print(f"   Carbohydrates: {stage3.get('carbohydrates', 0)}g")
        
        print("\n" + "=" * 70)
        print("✅ FULL PIPELINE TEST COMPLETE!")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("🧪 FoodVisionAI Complete Pipeline Test")
    print("=" * 70)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check if image provided
    if len(sys.argv) < 2:
        print("\n⚠️  No test image provided!")
        print("\n📖 Usage:")
        print("   python test_full_pipeline.py <path_to_food_image.jpg>")
        print("\n💡 Example:")
        print("   python test_full_pipeline.py test_images/pizza.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: {image_path}")
        sys.exit(1)
    
    # Test Stage 1 only first
    stage1_results = test_stage1_only(image_path)
    
    if stage1_results is None:
        print("\n❌ Stage 1 failed. Cannot proceed with full pipeline test.")
        sys.exit(1)
    
    # Ask user if they want to test full pipeline
    print("\n" + "=" * 70)
    response = input("🤔 Test full 3-stage pipeline? (y/n): ").strip().lower()
    
    if response == 'y':
        test_full_pipeline(image_path)
    else:
        print("\n✅ Stage 1 test complete. Skipping full pipeline test.")
    
    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    main()

