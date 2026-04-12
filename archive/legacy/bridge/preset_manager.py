#!/usr/bin/env python3
"""
Preset Manager for AI RemixMate Bridge

This module manages presets for different remix styles:
- "radio": -14 LUFS target, clean limiter
- "club": -9 LUFS target, different limiter setting  
- "ambient": -16 LUFS target, subtle processing
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PresetConfig:
    """Configuration for a remix preset."""
    name: str
    description: str
    lufs_target: float
    true_peak_ceiling: float
    mix_params: Dict[str, float]
    template_variant: Optional[str] = None


class PresetManager:
    """Manages remix presets for different styles."""
    
    def __init__(self):
        self.presets = self._load_default_presets()
    
    def _load_default_presets(self) -> Dict[str, PresetConfig]:
        """Load default presets."""
        return {
            "radio": PresetConfig(
                name="Radio Edit",
                description="Clean, broadcast-ready mix with -14 LUFS target",
                lufs_target=-14.0,
                true_peak_ceiling=-1.0,
                mix_params={
                    "vocal_gain_db": 0.0,
                    "inst_gain_db": -1.0,
                    "sidechain_amount": 0.15,
                    "hp_filter_hz": 100.0,
                    "reverb_send": 0.08
                },
                template_variant="radio"
            ),
            "club": PresetConfig(
                name="Club Edit",
                description="Loud, punchy mix optimized for club systems",
                lufs_target=-9.0,
                true_peak_ceiling=-0.5,
                mix_params={
                    "vocal_gain_db": 1.0,
                    "inst_gain_db": 0.0,
                    "sidechain_amount": 0.25,
                    "hp_filter_hz": 80.0,
                    "reverb_send": 0.12
                },
                template_variant="club"
            ),
            "ambient": PresetConfig(
                name="Ambient Edit",
                description="Subtle, atmospheric mix with gentle processing",
                lufs_target=-16.0,
                true_peak_ceiling=-1.5,
                mix_params={
                    "vocal_gain_db": -1.0,
                    "inst_gain_db": -2.0,
                    "sidechain_amount": 0.05,
                    "hp_filter_hz": 120.0,
                    "reverb_send": 0.20
                },
                template_variant="ambient"
            )
        }
    
    def get_preset(self, preset_name: str) -> Optional[PresetConfig]:
        """Get preset by name."""
        return self.presets.get(preset_name.lower())
    
    def list_presets(self) -> Dict[str, str]:
        """List available presets."""
        return {name: preset.description for name, preset in self.presets.items()}
    
    def apply_preset_to_manifest(self, manifest: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """
        Apply preset to manifest.
        
        Args:
            manifest: Original manifest
            preset_name: Name of preset to apply
            
        Returns:
            Modified manifest with preset applied
        """
        preset = self.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        # Create modified manifest
        modified_manifest = manifest.copy()
        
        # Apply mix parameters
        modified_manifest["mix_params"] = preset.mix_params.copy()
        
        # Update bounce settings for preset
        modified_manifest["bounce"]["normalize"] = False  # Presets handle their own levels
        
        # Add preset metadata
        modified_manifest["preset"] = {
            "name": preset.name,
            "description": preset.description,
            "lufs_target": preset.lufs_target,
            "true_peak_ceiling": preset.true_peak_ceiling
        }
        
        return modified_manifest
    
    def get_template_path(self, preset_name: str, base_template_path: str) -> str:
        """
        Get template path for preset (with variant if available).
        
        Args:
            preset_name: Name of preset
            base_template_path: Base template path
            
        Returns:
            Template path (may be variant or base)
        """
        preset = self.get_preset(preset_name)
        if not preset or not preset.template_variant:
            return base_template_path
        
        # Look for variant template
        base_path = Path(base_template_path)
        variant_path = base_path.parent / f"AI_RemixMate_{preset.template_variant.title()}.logicx"
        
        if variant_path.exists():
            return str(variant_path)
        else:
            print(f"⚠️ Variant template not found: {variant_path}, using base template")
            return base_template_path
    
    def create_export_pack(self, session_dir: Path, preset_name: str) -> Path:
        """
        Create export pack for delivery.
        
        Args:
            session_dir: Session directory with stems and reports
            preset_name: Preset used for this session
            
        Returns:
            Path to export pack directory
        """
        preset = self.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        # Create export pack directory
        pack_name = f"AI_RemixMate_{preset_name.title()}_{session_dir.name}"
        pack_dir = session_dir.parent / pack_name
        pack_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        import shutil
        
        # Copy stems
        stems_dir = pack_dir / "stems"
        stems_dir.mkdir(exist_ok=True)
        
        for stem_file in session_dir.glob("*.wav"):
            shutil.copy2(stem_file, stems_dir)
        
        # Copy reports
        for report_file in session_dir.glob("*.json"):
            shutil.copy2(report_file, pack_dir)
        
        # Create README
        readme_path = pack_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"""AI RemixMate Export Pack
========================

Preset: {preset.name}
Description: {preset.description}
Session: {session_dir.name}

Contents:
- stems/: Audio stems (vocals, instrumental)
- *.json: Session reports and manifests
- README.txt: This file

Technical Details:
- LUFS Target: {preset.lufs_target:.1f}
- True Peak Ceiling: {preset.true_peak_ceiling:.1f} dBFS
- Mix Parameters: {json.dumps(preset.mix_params, indent=2)}

Usage:
1. Import stems into your DAW
2. Apply the mix parameters from the manifest
3. Use the Logic Pro template for consistent results

Legal Notice:
This pack contains AI-generated remixes. Ensure you have rights to the source material.
Do not distribute copyrighted content without proper licensing.

Generated by AI RemixMate
""")
        
        print(f"📦 Export pack created: {pack_dir}")
        return pack_dir


def main():
    """Command-line interface for preset management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI RemixMate Preset Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List presets
    list_parser = subparsers.add_parser('list', help='List available presets')
    
    # Apply preset
    apply_parser = subparsers.add_parser('apply', help='Apply preset to manifest')
    apply_parser.add_argument('manifest_path', help='Path to manifest.json')
    apply_parser.add_argument('preset_name', help='Preset name to apply')
    apply_parser.add_argument('--output', help='Output manifest path')
    
    # Create export pack
    pack_parser = subparsers.add_parser('pack', help='Create export pack')
    pack_parser.add_argument('session_dir', help='Session directory')
    pack_parser.add_argument('preset_name', help='Preset name used')
    pack_parser.add_argument('--output', help='Output pack directory')
    
    args = parser.parse_args()
    
    manager = PresetManager()
    
    if args.command == 'list':
        print("🎛️ Available Presets:")
        print("=" * 40)
        for name, description in manager.list_presets().items():
            preset = manager.get_preset(name)
            print(f"{name}: {description}")
            print(f"   LUFS Target: {preset.lufs_target:.1f}")
            print(f"   True Peak: {preset.true_peak_ceiling:.1f} dBFS")
            print()
    
    elif args.command == 'apply':
        manifest_path = Path(args.manifest_path)
        if not manifest_path.exists():
            print(f"❌ Manifest not found: {manifest_path}")
            return 1
        
        # Read manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Apply preset
        try:
            modified_manifest = manager.apply_preset_to_manifest(manifest, args.preset_name)
            
            # Save modified manifest
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = manifest_path.parent / f"manifest_{args.preset_name}.json"
            
            with open(output_path, 'w') as f:
                json.dump(modified_manifest, f, indent=2)
            
            print(f"✅ Preset '{args.preset_name}' applied to manifest")
            print(f"   Output: {output_path}")
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            return 1
    
    elif args.command == 'pack':
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            print(f"❌ Session directory not found: {session_dir}")
            return 1
        
        try:
            pack_dir = manager.create_export_pack(session_dir, args.preset_name)
            print(f"✅ Export pack created: {pack_dir}")
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
