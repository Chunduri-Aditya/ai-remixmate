#!/usr/bin/env python3
"""
Performance Benchmarking Suite for AI RemixMate Bridge

This module benchmarks the performance of different components
across various scenarios and tracks performance metrics.
"""

from __future__ import annotations
import time
import psutil
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import sys
import librosa
from dataclasses import dataclass
from datetime import datetime


from scripts.bridge_metrics import lufs_metrics, beat_alignment_ms, key_compatible_camelot, vocal_intelligibility_proxy
from scripts.core.metrics import AudioMetrics
from scripts.core.musical_analysis import MusicalAnalyzer
from scripts.core.real_optimizer import RealOptimizer
from scripts.core.pro_audio_chain import ProfessionalAudioChain


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    input_size: int
    output_size: int
    success: bool
    error: str = ""


class PerformanceMonitor:
    """Monitors system performance during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.initial_cpu = self.process.cpu_percent()
    
    def get_current_stats(self) -> Tuple[float, float]:
        """Get current memory and CPU usage."""
        memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu = self.process.cpu_percent()
        return memory, cpu
    
    def get_peak_stats(self) -> Tuple[float, float]:
        """Get peak memory and CPU usage."""
        memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu = self.process.cpu_percent()
        return memory, cpu


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, output_dir: str = "runs/performance_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.audio_generator = self._create_audio_generator()
    
    def _create_audio_generator(self):
        """Create audio generator for testing."""
        class SimpleAudioGenerator:
            def __init__(self, sr=44100):
                self.sr = sr
            
            def generate_test_audio(self, duration: float, complexity: str = "simple") -> np.ndarray:
                """Generate test audio of specified complexity."""
                t = np.linspace(0, duration, int(self.sr * duration))
                
                if complexity == "simple":
                    # Simple sine wave
                    return 0.3 * np.sin(2 * np.pi * 440 * t)
                elif complexity == "medium":
                    # Multiple frequencies
                    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
                    audio += 0.1 * np.sin(2 * np.pi * 880 * t)
                    audio += 0.1 * np.sin(2 * np.pi * 1320 * t)
                    return audio
                elif complexity == "complex":
                    # Complex signal with noise
                    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
                    audio += 0.1 * np.sin(2 * np.pi * 880 * t)
                    audio += 0.1 * np.sin(2 * np.pi * 1320 * t)
                    audio += 0.05 * np.random.randn(len(t))
                    return audio
                else:
                    return np.zeros(len(t))
        
        return SimpleAudioGenerator()
    
    def benchmark_operation(self, operation_name: str, operation_func, *args, **kwargs) -> BenchmarkResult:
        """Benchmark a single operation."""
        monitor = PerformanceMonitor()
        
        start_time = time.time()
        start_memory, start_cpu = monitor.get_current_stats()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = ""
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory, end_cpu = monitor.get_current_stats()
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        
        # Estimate input/output sizes
        input_size = sum(len(str(arg)) for arg in args if isinstance(arg, (str, Path)))
        output_size = len(str(result)) if result is not None else 0
        
        benchmark_result = BenchmarkResult(
            operation=operation_name,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            input_size=input_size,
            output_size=output_size,
            success=success,
            error=error
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_audio_loading(self) -> List[BenchmarkResult]:
        """Benchmark audio loading performance."""
        print("🎵 Benchmarking audio loading...")
        
        durations = [5, 10, 30, 60]  # seconds
        complexities = ["simple", "medium", "complex"]
        
        results = []
        
        for duration in durations:
            for complexity in complexities:
                # Generate test audio
                audio = self.audio_generator.generate_test_audio(duration, complexity)
                
                # Save to temporary file
                temp_file = self.output_dir / f"temp_{duration}s_{complexity}.wav"
                sf.write(str(temp_file), audio, self.audio_generator.sr)
                
                # Benchmark loading with librosa
                def load_audio():
                    return librosa.load(str(temp_file), sr=44100)
                
                result = self.benchmark_operation(
                    f"audio_loading_{duration}s_{complexity}",
                    load_audio
                )
                results.append(result)
                
                # Clean up
                temp_file.unlink()
        
        return results
    
    def benchmark_metrics_computation(self) -> List[BenchmarkResult]:
        """Benchmark metrics computation performance."""
        print("📊 Benchmarking metrics computation...")
        
        durations = [10, 30, 60]
        complexities = ["simple", "medium", "complex"]
        
        results = []
        
        for duration in durations:
            for complexity in complexities:
                # Generate test audio
                audio1 = self.audio_generator.generate_test_audio(duration, complexity)
                audio2 = self.audio_generator.generate_test_audio(duration, complexity)
                
                # Save to temporary files
                temp1 = self.output_dir / f"temp1_{duration}s_{complexity}.wav"
                temp2 = self.output_dir / f"temp2_{duration}s_{complexity}.wav"
                sf.write(str(temp1), audio1, self.audio_generator.sr)
                sf.write(str(temp2), audio2, self.audio_generator.sr)
                
                # Benchmark LUFS metrics
                def compute_lufs():
                    return lufs_metrics(temp1)
                
                result = self.benchmark_operation(
                    f"lufs_metrics_{duration}s_{complexity}",
                    compute_lufs
                )
                results.append(result)
                
                # Benchmark beat alignment
                def compute_beat_alignment():
                    return beat_alignment_ms(temp1, temp2)
                
                result = self.benchmark_operation(
                    f"beat_alignment_{duration}s_{complexity}",
                    compute_beat_alignment
                )
                results.append(result)
                
                # Benchmark key compatibility
                def compute_key_compatibility():
                    return key_compatible_camelot("8A", "5A")
                
                result = self.benchmark_operation(
                    f"key_compatibility_{duration}s_{complexity}",
                    compute_key_compatibility
                )
                results.append(result)
                
                # Benchmark intelligibility
                def compute_intelligibility():
                    return vocal_intelligibility_proxy(temp1, temp2)
                
                result = self.benchmark_operation(
                    f"intelligibility_{duration}s_{complexity}",
                    compute_intelligibility
                )
                results.append(result)
                
                # Clean up
                temp1.unlink()
                temp2.unlink()
        
        return results
    
    def benchmark_core_components(self) -> List[BenchmarkResult]:
        """Benchmark core AI RemixMate components."""
        print("🧠 Benchmarking core components...")
        
        results = []
        
        # Generate test audio
        audio1 = self.audio_generator.generate_test_audio(30, "medium")
        audio2 = self.audio_generator.generate_test_audio(30, "medium")
        
        # Save to temporary files
        temp1 = self.output_dir / "temp_core1.wav"
        temp2 = self.output_dir / "temp_core2.wav"
        sf.write(str(temp1), audio1, self.audio_generator.sr)
        sf.write(str(temp2), audio2, self.audio_generator.sr)
        
        try:
            # Benchmark AudioMetrics
            def test_audio_metrics():
                metrics = AudioMetrics()
                return metrics.evaluate_remix(temp1, temp2, temp1)
            
            result = self.benchmark_operation("audio_metrics", test_audio_metrics)
            results.append(result)
            
            # Benchmark MusicalAnalyzer
            def test_musical_analyzer():
                analyzer = MusicalAnalyzer()
                return analyzer.analyze_musical_compatibility(audio1, audio2)
            
            result = self.benchmark_operation("musical_analyzer", test_musical_analyzer)
            results.append(result)
            
            # Benchmark RealOptimizer
            def test_real_optimizer():
                optimizer = RealOptimizer()
                return optimizer.optimize(audio1, audio2, max_iterations=10)
            
            result = self.benchmark_operation("real_optimizer", test_real_optimizer)
            results.append(result)
            
            # Benchmark ProfessionalAudioChain
            def test_audio_chain():
                chain = ProfessionalAudioChain()
                return chain.process_remix(audio1, audio2)
            
            result = self.benchmark_operation("audio_chain", test_audio_chain)
            results.append(result)
            
        except Exception as e:
            print(f"⚠️ Core component benchmarking failed: {e}")
        
        finally:
            # Clean up
            if temp1.exists():
                temp1.unlink()
            if temp2.exists():
                temp2.unlink()
        
        return results
    
    def benchmark_memory_usage(self) -> List[BenchmarkResult]:
        """Benchmark memory usage patterns."""
        print("💾 Benchmarking memory usage...")
        
        results = []
        
        # Test with different audio sizes
        sizes = [1024, 4096, 16384, 65536]  # samples
        
        for size in sizes:
            # Generate large audio arrays
            def create_large_audio():
                audio = np.random.randn(size)
                return audio
            
            result = self.benchmark_operation(
                f"memory_usage_{size}_samples",
                create_large_audio
            )
            results.append(result)
            
            # Test audio processing
            def process_large_audio():
                audio = np.random.randn(size)
                # Simulate processing
                processed = audio * 0.5
                return processed
            
            result = self.benchmark_operation(
                f"audio_processing_{size}_samples",
                process_large_audio
            )
            results.append(result)
        
        return results
    
    def benchmark_concurrent_operations(self) -> List[BenchmarkResult]:
        """Benchmark concurrent operations."""
        print("🔄 Benchmarking concurrent operations...")
        
        results = []
        
        # Test concurrent audio loading
        def concurrent_audio_loading():
            import concurrent.futures
            
            def load_audio(i):
                audio = self.audio_generator.generate_test_audio(10, "medium")
                temp_file = self.output_dir / f"temp_concurrent_{i}.wav"
                sf.write(str(temp_file), audio, self.audio_generator.sr)
                loaded_audio, sr = librosa.load(str(temp_file), sr=44100)
                temp_file.unlink()
                return loaded_audio
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(load_audio, i) for i in range(4)]
                results = [future.result() for future in futures]
            
            return results
        
        result = self.benchmark_operation("concurrent_audio_loading", concurrent_audio_loading)
        results.append(result)
        
        # Test concurrent metrics computation
        def concurrent_metrics():
            import concurrent.futures
            
            def compute_metrics(i):
                audio1 = self.audio_generator.generate_test_audio(10, "medium")
                audio2 = self.audio_generator.generate_test_audio(10, "medium")
                temp1 = self.output_dir / f"temp_metrics1_{i}.wav"
                temp2 = self.output_dir / f"temp_metrics2_{i}.wav"
                sf.write(str(temp1), audio1, self.audio_generator.sr)
                sf.write(str(temp2), audio2, self.audio_generator.sr)
                
                lufs, peak, clipping = lufs_metrics(temp1)
                alignment = beat_alignment_ms(temp1, temp2)
                
                temp1.unlink()
                temp2.unlink()
                
                return {"lufs": lufs, "peak": peak, "clipping": clipping, "alignment": alignment}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(compute_metrics, i) for i in range(4)]
                results = [future.result() for future in futures]
            
            return results
        
        result = self.benchmark_operation("concurrent_metrics", concurrent_metrics)
        results.append(result)
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        audio_loading_results = self.benchmark_audio_loading()
        metrics_results = self.benchmark_metrics_computation()
        core_results = self.benchmark_core_components()
        memory_results = self.benchmark_memory_usage()
        concurrent_results = self.benchmark_concurrent_operations()
        
        total_time = time.time() - start_time
        
        # Generate report
        report = self.generate_benchmark_report(total_time)
        
        return report
    
    def generate_benchmark_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        print("\n📊 Generating Performance Benchmark Report")
        print("=" * 50)
        
        # Calculate statistics
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if successful_results:
            durations = [r.duration for r in successful_results]
            memory_usage = [r.memory_usage for r in successful_results]
            cpu_usage = [r.cpu_usage for r in successful_results]
            
            stats = {
                "total_operations": len(self.results),
                "successful_operations": len(successful_results),
                "failed_operations": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) if self.results else 0,
                "average_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "std_duration": np.std(durations),
                "average_memory_usage": np.mean(memory_usage),
                "max_memory_usage": np.max(memory_usage),
                "average_cpu_usage": np.mean(cpu_usage),
                "max_cpu_usage": np.max(cpu_usage)
            }
        else:
            stats = {
                "total_operations": len(self.results),
                "successful_operations": 0,
                "failed_operations": len(failed_results),
                "success_rate": 0
            }
        
        # Group results by operation type
        operation_groups = {}
        for result in self.results:
            operation_type = result.operation.split('_')[0]
            if operation_type not in operation_groups:
                operation_groups[operation_type] = []
            operation_groups[operation_type].append(result)
        
        # Calculate group statistics
        group_stats = {}
        for group_name, group_results in operation_groups.items():
            successful_group = [r for r in group_results if r.success]
            if successful_group:
                group_durations = [r.duration for r in successful_group]
                group_stats[group_name] = {
                    "total": len(group_results),
                    "successful": len(successful_group),
                    "average_duration": np.mean(group_durations),
                    "max_duration": np.max(group_durations),
                    "min_duration": np.min(group_durations)
                }
        
        # Build report
        report = {
            "benchmark_summary": {
                "total_benchmark_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                }
            },
            "performance_statistics": stats,
            "operation_group_statistics": group_stats,
            "detailed_results": [
                {
                    "operation": r.operation,
                    "duration": r.duration,
                    "memory_usage": r.memory_usage,
                    "cpu_usage": r.cpu_usage,
                    "input_size": r.input_size,
                    "output_size": r.output_size,
                    "success": r.success,
                    "error": r.error
                }
                for r in self.results
            ],
            "failed_operations": [
                {
                    "operation": r.operation,
                    "error": r.error
                }
                for r in failed_results
            ]
        }
        
        # Save report
        report_path = self.output_dir / "performance_benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_benchmark_summary(report)
        
        return report
    
    def print_benchmark_summary(self, report: Dict[str, Any]) -> None:
        """Print benchmark summary."""
        stats = report["performance_statistics"]
        group_stats = report["operation_group_statistics"]
        
        print(f"\n🎯 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Operations: {stats['total_operations']}")
        print(f"Successful: {stats['successful_operations']} ({stats['success_rate']:.1%})")
        print(f"Failed: {stats['failed_operations']}")
        print(f"Total Benchmark Time: {report['benchmark_summary']['total_benchmark_time']:.2f}s")
        
        if stats['successful_operations'] > 0:
            print(f"\n⏱️ PERFORMANCE METRICS")
            print(f"Average Duration: {stats['average_duration']:.4f}s")
            print(f"Median Duration: {stats['median_duration']:.4f}s")
            print(f"Min Duration: {stats['min_duration']:.4f}s")
            print(f"Max Duration: {stats['max_duration']:.4f}s")
            print(f"Std Duration: {stats['std_duration']:.4f}s")
            
            print(f"\n💾 MEMORY USAGE")
            print(f"Average Memory: {stats['average_memory_usage']:.2f} MB")
            print(f"Max Memory: {stats['max_memory_usage']:.2f} MB")
            
            print(f"\n🖥️ CPU USAGE")
            print(f"Average CPU: {stats['average_cpu_usage']:.2f}%")
            print(f"Max CPU: {stats['max_cpu_usage']:.2f}%")
        
        print(f"\n📊 OPERATION GROUP PERFORMANCE")
        for group_name, group_stat in group_stats.items():
            print(f"  {group_name.title()}:")
            print(f"    Total: {group_stat['total']}, Successful: {group_stat['successful']}")
            print(f"    Avg Duration: {group_stat['average_duration']:.4f}s")
            print(f"    Max Duration: {group_stat['max_duration']:.4f}s")
        
        print(f"\n📁 Report saved to: {self.output_dir / 'performance_benchmark_report.json'}")


def main():
    """Main function to run performance benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Benchmarking Suite for AI RemixMate Bridge")
    parser.add_argument("--output-dir", default="runs/performance_benchmark", help="Output directory for benchmark results")
    parser.add_argument("--benchmark", choices=["audio_loading", "metrics", "core", "memory", "concurrent", "all"], 
                       default="all", help="Specific benchmark to run")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(args.output_dir)
    
    if args.benchmark == "audio_loading":
        results = benchmark.benchmark_audio_loading()
        print(f"✅ Audio loading benchmark completed: {len(results)} operations")
    elif args.benchmark == "metrics":
        results = benchmark.benchmark_metrics_computation()
        print(f"✅ Metrics benchmark completed: {len(results)} operations")
    elif args.benchmark == "core":
        results = benchmark.benchmark_core_components()
        print(f"✅ Core components benchmark completed: {len(results)} operations")
    elif args.benchmark == "memory":
        results = benchmark.benchmark_memory_usage()
        print(f"✅ Memory usage benchmark completed: {len(results)} operations")
    elif args.benchmark == "concurrent":
        results = benchmark.benchmark_concurrent_operations()
        print(f"✅ Concurrent operations benchmark completed: {len(results)} operations")
    else:
        # Run all benchmarks
        report = benchmark.run_comprehensive_benchmark()
        print(f"✅ Comprehensive benchmark completed")
    
    return 0


if __name__ == "__main__":
    exit(main())
