#!/usr/bin/env python3
"""
Export Manifest for AI RemixMate Bridge

This module creates manifest.json files for Python → Logic Pro communication.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
import datetime


def build_manifest(
    project_path: str,
    session_id: str,
    out_dir: str,
    stems: Dict[str, Any],
    project_bpm: float,
    project_key: str,
    mix_params: Dict[str, float]
) -> Dict[str, Any]:
    """
    Build a manifest.json for Logic Pro bridge communication.
    
    Args:
        project_path: Path to Logic Pro template project (.logicx file)
        session_id: Unique session identifier
        out_dir: Output directory for this session
        stems: Dictionary containing stem information
        project_bpm: Project tempo in BPM
        project_key: Project key in Camelot notation
        mix_params: Mix parameters dictionary
        
    Returns:
        Manifest dictionary ready for JSON serialization
    """
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Build bounce path
    bounce_path = str(Path(out_dir) / "AI_RemixMate_Bounce.wav")
    
    manifest = {
        "version": "1.0",
        "project_path": str(Path(project_path).resolve()),
        "session": {
            "id": session_id,
            "out_dir": str(Path(out_dir).resolve())
        },
        "tempo_key": {
            "project_bpm": float(project_bpm),
            "project_key": project_key,
            "base_track": {
                "bpm": float(stems.get("instrumental_bpm", project_bpm)),
                "key": stems.get("instrumental_key", project_key),
                "path": str(Path(stems["instrumental_path"]).resolve())
            },
            "vocal_track": {
                "bpm": float(stems.get("vocals_bpm", project_bpm)),
                "key": stems.get("vocals_key", project_key),
                "path": str(Path(stems["vocals_path"]).resolve())
            }
        },
        "mix_params": {
            "vocal_gain_db": float(mix_params.get("vocal_gain_db", 0.0)),
            "inst_gain_db": float(mix_params.get("inst_gain_db", 0.0)),
            "sidechain_amount": float(mix_params.get("sidechain_amount", 0.0)),
            "hp_filter_hz": float(mix_params.get("hp_filter_hz", 100.0)),
            "reverb_send": float(mix_params.get("reverb_send", 0.0))
        },
        "import": {
            "tracks": [
                {
                    "logic_track_name": "AI_INSTRUMENTAL",
                    "audio_path": str(Path(stems["instrumental_path"]).resolve())
                },
                {
                    "logic_track_name": "AI_VOCALS",
                    "audio_path": str(Path(stems["vocals_path"]).resolve())
                }
            ]
        },
        "bounce": {
            "filename": "AI_RemixMate_Bounce.wav",
            "path": bounce_path,
            "sample_rate": 44100,
            "bit_depth": 24,
            "normalize": False
        },
        "smart_tempo": {
            "mode": "ADAPT",
            "analyze_on_import": True
        }
    }
    
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> bool:
    """
    Validate manifest against schema.
    
    Args:
        manifest: Manifest dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import jsonschema
        
        # Load schema
        schema_path = Path(__file__).parent / "bridge" / "manifest_schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        
        # Validate
        jsonschema.validate(manifest, schema)
        return True
        
    except ImportError:
        print("⚠️ jsonschema not available, skipping validation")
        return True
    except Exception as e:
        print(f"❌ Manifest validation failed: {e}")
        return False


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """
    Write manifest to JSON file.
    
    Args:
        manifest: Manifest dictionary
        path: Output file path
    """
    # Validate before writing
    if not validate_manifest(manifest):
        raise ValueError("Invalid manifest")
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📋 Manifest written to: {path}")


def create_session_id() -> str:
    """Create a unique session ID with timestamp."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def main():
    """Example usage of manifest creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create manifest for Logic Pro bridge")
    parser.add_argument("--project", required=True, help="Path to Logic Pro template")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--vocals", required=True, help="Path to vocals file")
    parser.add_argument("--instrumental", required=True, help="Path to instrumental file")
    parser.add_argument("--bpm", type=float, default=120.0, help="Project BPM")
    parser.add_argument("--key", default="8A", help="Project key (Camelot)")
    
    args = parser.parse_args()
    
    # Create session ID
    session_id = create_session_id()
    
    # Build stems dictionary
    stems = {
        "vocals_path": args.vocals,
        "instrumental_path": args.instrumental,
        "vocals_bpm": args.bpm,
        "instrumental_bpm": args.bpm,
        "vocals_key": args.key,
        "instrumental_key": args.key
    }
    
    # Default mix parameters
    mix_params = {
        "vocal_gain_db": 0.0,
        "inst_gain_db": 0.0,
        "sidechain_amount": 0.0,
        "hp_filter_hz": 100.0,
        "reverb_send": 0.0
    }
    
    # Build manifest
    manifest = build_manifest(
        project_path=args.project,
        session_id=session_id,
        out_dir=args.out_dir,
        stems=stems,
        project_bpm=args.bpm,
        project_key=args.key,
        mix_params=mix_params
    )
    
    # Write manifest
    manifest_path = Path(args.out_dir) / "manifest.json"
    write_manifest(manifest, manifest_path)
    
    print(f"✅ Manifest created successfully")
    print(f"   Session ID: {session_id}")
    print(f"   Project: {args.project}")
    print(f"   Output: {args.out_dir}")


if __name__ == "__main__":
    main()
