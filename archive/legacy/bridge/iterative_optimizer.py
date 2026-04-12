#!/usr/bin/env python3
"""
Iterative Optimizer for AI RemixMate Bridge

This module implements the iterative optimization loop:
1. Generate initial remix with Logic Pro
2. Compute metrics and check constraints
3. If constraints not satisfied, adjust parameters and re-render
4. Repeat up to 3 iterations, pick best result
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import copy

from .logic_bridge import main as logic_main
from .logic_bridge import read_json, write_manifest
from scripts.export_manifest import build_manifest, create_session_id


class IterativeOptimizer:
    """Iterative optimizer for bridge constraint satisfaction."""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.iteration_results = []
    
    def adjust_parameters(self, manifest: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust mix parameters based on constraint violations.
        
        Args:
            manifest: Current manifest
            report: Current metrics report
            
        Returns:
            Adjusted manifest with new parameters
        """
        adjusted_manifest = copy.deepcopy(manifest)
        mix_params = adjusted_manifest["mix_params"]
        metrics = report["metrics"]
        
        print(f"🔧 Adjusting parameters based on constraints...")
        
        # Adjust vocal gain if true peak is too high
        if metrics["true_peak_dbfs"] > -1.0:
            reduction = min(2.0, metrics["true_peak_dbfs"] + 1.0)
            mix_params["vocal_gain_db"] -= reduction
            print(f"   Reducing vocal gain by {reduction:.1f} dB (true peak: {metrics['true_peak_dbfs']:.1f} dBFS)")
        
        # Adjust instrumental gain if clipping
        if metrics["clipping_ratio"] > 0.005:
            reduction = min(3.0, metrics["clipping_ratio"] * 100)
            mix_params["inst_gain_db"] -= reduction
            print(f"   Reducing instrumental gain by {reduction:.1f} dB (clipping: {metrics['clipping_ratio']:.3f})")
        
        # Increase sidechain if intelligibility is low
        if metrics["vocal_intelligibility_proxy"] is not None and metrics["vocal_intelligibility_proxy"] < 0.6:
            increase = min(0.2, 0.6 - metrics["vocal_intelligibility_proxy"])
            mix_params["sidechain_amount"] += increase
            print(f"   Increasing sidechain by {increase:.2f} (intelligibility: {metrics['vocal_intelligibility_proxy']:.3f})")
        
        # Adjust high-pass filter if needed
        if metrics["beat_alignment_ms"] > 40:
            mix_params["hp_filter_hz"] = min(200, mix_params["hp_filter_hz"] + 20)
            print(f"   Increasing HP filter to {mix_params['hp_filter_hz']:.0f} Hz (beat alignment: {metrics['beat_alignment_ms']:.1f} ms)")
        
        # Adjust reverb if LUFS is off target
        lufs = metrics["lufs_integrated"]
        if lufs < -15.0:  # Too quiet
            mix_params["reverb_send"] = min(0.3, mix_params["reverb_send"] + 0.05)
            print(f"   Increasing reverb send (LUFS: {lufs:.1f})")
        elif lufs > -13.0:  # Too loud
            mix_params["reverb_send"] = max(0.0, mix_params["reverb_send"] - 0.05)
            print(f"   Decreasing reverb send (LUFS: {lufs:.1f})")
        
        return adjusted_manifest
    
    def score_iteration(self, report: Dict[str, Any]) -> float:
        """
        Score an iteration based on constraint satisfaction and quality.
        
        Args:
            report: Metrics report
            
        Returns:
            Score (0-1, higher is better)
        """
        metrics = report["metrics"]
        
        # Base score from constraint satisfaction
        base_score = 1.0 if report["constraints_satisfied"] else 0.5
        
        # Quality bonuses
        quality_bonus = 0.0
        
        # LUFS bonus (closer to -14 is better)
        lufs_error = abs(metrics["lufs_integrated"] + 14.0)
        lufs_bonus = max(0, 0.1 - lufs_error * 0.01)
        quality_bonus += lufs_bonus
        
        # Intelligibility bonus
        if metrics["vocal_intelligibility_proxy"] is not None:
            quality_bonus += metrics["vocal_intelligibility_proxy"] * 0.1
        
        # Beat alignment bonus
        if metrics["beat_alignment_ms"] <= 20:
            quality_bonus += 0.1
        
        # Key compatibility bonus
        if metrics["key_compatible"]:
            quality_bonus += 0.05
        
        return min(1.0, base_score + quality_bonus)
    
    def optimize(self, initial_manifest: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run iterative optimization.
        
        Args:
            initial_manifest: Initial manifest to optimize
            
        Returns:
            Tuple of (best_manifest, best_report)
        """
        print(f"🔄 Starting iterative optimization (max {self.max_iterations} iterations)...")
        
        current_manifest = copy.deepcopy(initial_manifest)
        best_score = -1.0
        best_manifest = None
        best_report = None
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Update session ID for this iteration
            current_manifest["session"]["id"] = f"{initial_manifest['session']['id']}_iter{iteration + 1}"
            
            # Write manifest for this iteration
            manifest_path = Path(current_manifest["session"]["out_dir"]) / f"manifest_iter_{iteration + 1}.json"
            write_manifest(current_manifest, manifest_path)
            
            # Run Logic Pro bridge
            try:
                # Set up sys.argv for logic_bridge
                import sys
                original_argv = sys.argv.copy()
                sys.argv = ["logic_bridge.py", str(manifest_path)]
                
                try:
                    logic_main()
                finally:
                    sys.argv = original_argv
                
                # Read report
                report_path = Path(current_manifest["session"]["out_dir"]) / "report.json"
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
                    
                    # Score this iteration
                    score = self.score_iteration(report)
                    print(f"   Score: {score:.3f}")
                    print(f"   Constraints satisfied: {'✅' if report['constraints_satisfied'] else '❌'}")
                    
                    # Store result
                    self.iteration_results.append({
                        "iteration": iteration + 1,
                        "manifest": copy.deepcopy(current_manifest),
                        "report": report,
                        "score": score
                    })
                    
                    # Update best if improved
                    if score > best_score:
                        best_score = score
                        best_manifest = copy.deepcopy(current_manifest)
                        best_report = report
                        print(f"   🎯 New best score!")
                    
                    # Check if we should continue
                    if report["constraints_satisfied"] and score > 0.8:
                        print(f"   ✅ Constraints satisfied with good quality, stopping early")
                        break
                    
                    # Adjust parameters for next iteration
                    if iteration < self.max_iterations - 1:
                        current_manifest = self.adjust_parameters(current_manifest, report)
                
                else:
                    print(f"   ❌ No report generated for iteration {iteration + 1}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Iteration {iteration + 1} failed: {e}")
                break
        
        print(f"\n🎯 Optimization completed!")
        print(f"   Best score: {best_score:.3f}")
        print(f"   Iterations run: {len(self.iteration_results)}")
        
        return best_manifest, best_report
    
    def save_optimization_report(self, output_path: Path) -> None:
        """Save detailed optimization report."""
        report = {
            "optimization_summary": {
                "total_iterations": len(self.iteration_results),
                "best_score": max([r["score"] for r in self.iteration_results]) if self.iteration_results else 0.0,
                "constraints_satisfied": any([r["report"]["constraints_satisfied"] for r in self.iteration_results])
            },
            "iterations": self.iteration_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Optimization report saved to: {output_path}")
    
    def print_optimization_summary(self) -> None:
        """Print optimization summary."""
        print("\n" + "="*60)
        print("🔄 ITERATIVE OPTIMIZATION SUMMARY")
        print("="*60)
        
        if not self.iteration_results:
            print("   No iterations completed")
            return
        
        print(f"\n📊 RESULTS")
        print(f"   Total iterations: {len(self.iteration_results)}")
        print(f"   Best score: {max([r['score'] for r in self.iteration_results]):.3f}")
        print(f"   Constraints satisfied: {'✅' if any([r['report']['constraints_satisfied'] for r in self.iteration_results]) else '❌'}")
        
        print(f"\n📋 ITERATION DETAILS")
        for result in self.iteration_results:
            metrics = result["report"]["metrics"]
            print(f"   Iteration {result['iteration']}:")
            print(f"      Score: {result['score']:.3f}")
            print(f"      LUFS: {metrics['lufs_integrated']:.1f}")
            print(f"      True Peak: {metrics['true_peak_dbfs']:.1f} dBFS")
            print(f"      Clipping: {metrics['clipping_ratio']:.3f}")
            print(f"      Beat Alignment: {metrics['beat_alignment_ms']:.1f} ms")
            print(f"      Constraints: {'✅' if result['report']['constraints_satisfied'] else '❌'}")
        
        print("\n" + "="*60)


def main():
    """Command-line interface for iterative optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Iterative optimization for bridge constraints")
    parser.add_argument("manifest_path", help="Path to initial manifest.json")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum optimization iterations")
    parser.add_argument("--output", help="Output directory for optimization results")
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return 1
    
    # Read initial manifest
    initial_manifest = read_json(manifest_path)
    
    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(initial_manifest["session"]["out_dir"]) / "optimization"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_manifest["session"]["out_dir"] = str(output_dir)
    
    # Run optimization
    optimizer = IterativeOptimizer(max_iterations=args.max_iterations)
    best_manifest, best_report = optimizer.optimize(initial_manifest)
    
    # Save results
    if best_manifest and best_report:
        # Save best manifest
        best_manifest_path = output_dir / "best_manifest.json"
        with open(best_manifest_path, 'w') as f:
            json.dump(best_manifest, f, indent=2)
        
        # Save best report
        best_report_path = output_dir / "best_report.json"
        with open(best_report_path, 'w') as f:
            json.dump(best_report, f, indent=2)
        
        # Save optimization report
        optimization_report_path = output_dir / "optimization_report.json"
        optimizer.save_optimization_report(optimization_report_path)
        
        # Print summary
        optimizer.print_optimization_summary()
        
        print(f"\n✅ Optimization completed!")
        print(f"   Best manifest: {best_manifest_path}")
        print(f"   Best report: {best_report_path}")
        print(f"   Optimization report: {optimization_report_path}")
        
        return 0
    else:
        print(f"❌ Optimization failed - no valid results")
        return 1


if __name__ == "__main__":
    exit(main())
