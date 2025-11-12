#!/usr/bin/env python3
"""
Comprehensive code review script to find issues and missing functionality.
"""
import os
import sys
import ast
import importlib.util
from pathlib import Path

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
github_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(github_dir, "src"))

issues = []
warnings = []

def check_imports(module_path, module_name):
    """Check if module can be imported and has required functions."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            issues.append(f"❌ {module_name}: Cannot create spec from file")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for common issues
        if not hasattr(module, '__file__'):
            warnings.append(f"⚠️  {module_name}: No __file__ attribute")
        
        return True
    except Exception as e:
        issues.append(f"❌ {module_name}: Import error - {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if not os.path.exists(filepath):
        issues.append(f"❌ Missing file: {filepath} ({description})")
        return False
    return True

def check_function_exists(module, func_name, description=""):
    """Check if a function exists in module."""
    if not hasattr(module, func_name):
        issues.append(f"❌ Missing function: {module.__name__}.{func_name} {description}")
        return False
    return True

def main():
    print("=" * 60)
    print("🔍 Comprehensive Code Review")
    print("=" * 60)
    print()
    
    # Check core modules
    modules_to_check = [
        ("remixmate.config", "src/remixmate/config.py"),
        ("remixmate.remix_core", "src/remixmate/remix_core.py"),
        ("remixmate.dj_mixing", "src/remixmate/dj_mixing.py"),
        ("remixmate.recommendations", "src/remixmate/recommendations.py"),
        ("remixmate.structure_detection", "src/remixmate/structure_detection.py"),
        ("remixmate.timeline_planner", "src/remixmate/timeline_planner.py"),
        ("remixmate.timeline_renderer", "src/remixmate/timeline_renderer.py"),
        ("remixmate.ml_audio_features", "src/remixmate/ml_audio_features.py"),
        ("remixmate.auto_mode_selector", "src/remixmate/auto_mode_selector.py"),
        ("remixmate.playlist_manager", "src/remixmate/playlist_manager.py"),
        ("remixmate.lyrics_extraction", "src/remixmate/lyrics_extraction.py"),
        ("remixmate.advanced_arrangement", "src/remixmate/advanced_arrangement.py"),
    ]
    
    print("📦 Checking module imports...")
    imported_modules = {}
    for module_name, file_path in modules_to_check:
        full_path = os.path.join(github_dir, file_path)
        if check_file_exists(full_path, module_name):
            if check_imports(full_path, module_name):
                try:
                    mod = __import__(module_name, fromlist=[''])
                    imported_modules[module_name] = mod
                    print(f"  ✅ {module_name}")
                except Exception as e:
                    print(f"  ❌ {module_name}: {e}")
    
    print()
    print("🔍 Checking function availability...")
    
    # Check critical functions
    critical_functions = {
        "remixmate.remix_core": [
            "remix_two_files",
            "convert_audio_to_wav",
            "separate_stems",
        ],
        "remixmate.dj_mixing": [
            "balance_stem_volumes",
            "detect_bpm",
            "estimate_key",
            "create_crossfade_curve",
            "time_stretch_audio",
        ],
        "remixmate.recommendations": [
            "analyze_track_characteristics",
            "analyze_track_for_display",
            "find_compatible_songs",
            "format_recommendations",
        ],
        "remixmate.structure_detection": [
            "detect_beat_grid",
            "detect_sections",
        ],
        "remixmate.timeline_planner": [
            "TimelinePlanner",
        ],
        "remixmate.timeline_renderer": [
            "TimelineRenderer",
        ],
        "remixmate.ml_audio_features": [
            "get_strategy_engine",
            "classify_genre",
            "predict_energy",
        ],
        "remixmate.auto_mode_selector": [
            "select_auto_mode",
        ],
    }
    
    for module_name, funcs in critical_functions.items():
        if module_name in imported_modules:
            mod = imported_modules[module_name]
            for func_name in funcs:
                if check_function_exists(mod, func_name):
                    print(f"  ✅ {module_name}.{func_name}")
                else:
                    print(f"  ❌ {module_name}.{func_name}")
    
    print()
    print("📁 Checking file structure...")
    
    # Check important files
    important_files = [
        ("app.py", "Main web application"),
        ("requirements.txt", "Dependencies"),
        ("models/song_embeddings.json", "Song database (optional)"),
    ]
    
    for filename, description in important_files:
        filepath = os.path.join(github_dir, filename)
        if check_file_exists(filepath, description):
            print(f"  ✅ {filename}")
        else:
            print(f"  ⚠️  {filename} (optional)")
    
    print()
    print("=" * 60)
    print("📊 Review Summary")
    print("=" * 60)
    
    if issues:
        print(f"\n❌ Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No critical issues found!")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  {warning}")
    
    # Additional checks
    print("\n🔍 Additional Checks:")
    
    # Check if app.py imports work
    try:
        app_path = os.path.join(github_dir, "app.py")
        if os.path.exists(app_path):
            with open(app_path, 'r') as f:
                app_code = f.read()
                if "from remixmate" in app_code:
                    print("  ✅ app.py uses remixmate imports")
                else:
                    warnings.append("app.py might not be using remixmate imports correctly")
    except Exception as e:
        warnings.append(f"Could not check app.py: {e}")
    
    # Check config.py has required constants
    if "remixmate.config" in imported_modules:
        config = imported_modules["remixmate.config"]
        required_configs = ["SAMPLE_RATE", "AUDIO_OUTPUT_DIR", "CONVERTED_DIR"]
        for cfg in required_configs:
            if hasattr(config, cfg):
                print(f"  ✅ config.{cfg}")
            else:
                issues.append(f"Missing config.{cfg}")
    
    print()
    if not issues and not warnings:
        print("🎉 Code review passed! No issues found.")
    elif issues:
        print(f"⚠️  Found {len(issues)} issue(s) that need attention.")
    else:
        print("✅ Code review passed with warnings.")

if __name__ == "__main__":
    main()

