#!/usr/bin/env python3
"""
Master Test Runner for AI RemixMate Bridge

This module orchestrates all test suites and provides a unified interface
for comprehensive testing of the bridge system.
"""

from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse


from scripts.bridge.comprehensive_test_suite import ComprehensiveTestSuite
from scripts.bridge.performance_benchmark import PerformanceBenchmark
from scripts.bridge.real_music_test import RealMusicTestSuite


class MasterTestRunner:
    """Master test runner that orchestrates all test suites."""
    
    def __init__(self, output_dir: str = "runs/master_test"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, 
                     include_comprehensive: bool = True,
                     include_performance: bool = True,
                     include_real_music: bool = True,
                     create_music_library: bool = True) -> Dict[str, Any]:
        """Run all test suites."""
        print("🚀 Starting Master Test Suite for AI RemixMate Bridge")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Run comprehensive test suite
        if include_comprehensive:
            print("\n📊 Running Comprehensive Test Suite...")
            comprehensive_suite = ComprehensiveTestSuite(
                output_dir=str(self.output_dir / "comprehensive")
            )
            comprehensive_results = comprehensive_suite.run_comprehensive_tests()
            self.test_results["comprehensive"] = comprehensive_results
        
        # Run performance benchmark
        if include_performance:
            print("\n⚡ Running Performance Benchmark...")
            performance_suite = PerformanceBenchmark(
                output_dir=str(self.output_dir / "performance")
            )
            performance_results = performance_suite.run_comprehensive_benchmark()
            self.test_results["performance"] = performance_results
        
        # Run real music tests
        if include_real_music:
            print("\n🎵 Running Real Music Test Suite...")
            real_music_suite = RealMusicTestSuite(
                music_dir="test_music",
                output_dir=str(self.output_dir / "real_music")
            )
            
            if create_music_library:
                real_music_suite.create_test_music_library()
            
            real_music_suite.create_test_cases()
            real_music_results = real_music_suite.run_comprehensive_tests()
            self.test_results["real_music"] = real_music_results
        
        self.end_time = time.time()
        
        # Generate master report
        return self.generate_master_report()
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick tests for rapid validation."""
        print("🏃 Running Quick Test Suite...")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # Quick comprehensive tests
        print("\n📊 Running Quick Comprehensive Tests...")
        comprehensive_suite = ComprehensiveTestSuite(
            output_dir=str(self.output_dir / "comprehensive_quick")
        )
        # Limit to first 5 scenarios
        comprehensive_suite.test_scenarios = comprehensive_suite.test_scenarios[:5]
        comprehensive_results = comprehensive_suite.run_comprehensive_tests()
        self.test_results["comprehensive_quick"] = comprehensive_results
        
        # Quick performance tests
        print("\n⚡ Running Quick Performance Tests...")
        performance_suite = PerformanceBenchmark(
            output_dir=str(self.output_dir / "performance_quick")
        )
        # Run only audio loading and metrics benchmarks
        audio_results = performance_suite.benchmark_audio_loading()
        metrics_results = performance_suite.benchmark_metrics_computation()
        performance_results = performance_suite.generate_benchmark_report(0)
        self.test_results["performance_quick"] = performance_results
        
        # Quick real music tests
        print("\n🎵 Running Quick Real Music Tests...")
        real_music_suite = RealMusicTestSuite(
            music_dir="test_music",
            output_dir=str(self.output_dir / "real_music_quick")
        )
        real_music_suite.create_test_music_library()
        real_music_suite.create_test_cases()
        # Limit to first 3 test cases
        real_music_suite.test_cases = real_music_suite.test_cases[:3]
        real_music_results = real_music_suite.run_comprehensive_tests()
        self.test_results["real_music_quick"] = real_music_results
        
        self.end_time = time.time()
        
        # Generate master report
        return self.generate_master_report()
    
    def run_specific_tests(self, test_types: List[str]) -> Dict[str, Any]:
        """Run specific test types."""
        print(f"🎯 Running Specific Tests: {', '.join(test_types)}")
        print("=" * 50)
        
        self.start_time = time.time()
        
        for test_type in test_types:
            if test_type == "comprehensive":
                print("\n📊 Running Comprehensive Test Suite...")
                comprehensive_suite = ComprehensiveTestSuite(
                    output_dir=str(self.output_dir / "comprehensive")
                )
                comprehensive_results = comprehensive_suite.run_comprehensive_tests()
                self.test_results["comprehensive"] = comprehensive_results
            
            elif test_type == "performance":
                print("\n⚡ Running Performance Benchmark...")
                performance_suite = PerformanceBenchmark(
                    output_dir=str(self.output_dir / "performance")
                )
                performance_results = performance_suite.run_comprehensive_benchmark()
                self.test_results["performance"] = performance_results
            
            elif test_type == "real_music":
                print("\n🎵 Running Real Music Test Suite...")
                real_music_suite = RealMusicTestSuite(
                    music_dir="test_music",
                    output_dir=str(self.output_dir / "real_music")
                )
                real_music_suite.create_test_music_library()
                real_music_suite.create_test_cases()
                real_music_results = real_music_suite.run_comprehensive_tests()
                self.test_results["real_music"] = real_music_results
            
            else:
                print(f"⚠️ Unknown test type: {test_type}")
        
        self.end_time = time.time()
        
        # Generate master report
        return self.generate_master_report()
    
    def generate_master_report(self) -> Dict[str, Any]:
        """Generate master test report."""
        print("\n📊 Generating Master Test Report")
        print("=" * 50)
        
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate overall statistics
        total_tests = 0
        total_successful = 0
        total_failed = 0
        total_constraints_satisfied = 0
        
        for test_suite, results in self.test_results.items():
            if "test_summary" in results:
                summary = results["test_summary"]
                total_tests += summary.get("total_tests", 0)
                total_successful += summary.get("successful_tests", 0)
                total_failed += summary.get("failed_tests", 0)
                total_constraints_satisfied += summary.get("constraints_satisfied", 0)
        
        # Build master report
        master_report = {
            "master_test_summary": {
                "total_test_suites": len(self.test_results),
                "total_tests": total_tests,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "overall_success_rate": total_successful / total_tests if total_tests > 0 else 0,
                "total_constraints_satisfied": total_constraints_satisfied,
                "overall_constraint_satisfaction_rate": total_constraints_satisfied / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_time
            },
            "test_suite_results": self.test_results,
            "execution_timestamp": datetime.now().isoformat(),
            "test_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "output_directory": str(self.output_dir)
            }
        }
        
        # Save master report
        master_report_path = self.output_dir / "master_test_report.json"
        with open(master_report_path, 'w') as f:
            json.dump(master_report, f, indent=2)
        
        # Print master summary
        self.print_master_summary(master_report)
        
        return master_report
    
    def print_master_summary(self, report: Dict[str, Any]) -> None:
        """Print master test summary."""
        summary = report["master_test_summary"]
        
        print(f"\n🎯 MASTER TEST RESULTS")
        print("=" * 70)
        print(f"Test Suites Run: {summary['total_test_suites']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['total_successful']} ({summary['overall_success_rate']:.1%})")
        print(f"Failed: {summary['total_failed']}")
        print(f"Constraints Satisfied: {summary['total_constraints_satisfied']} ({summary['overall_constraint_satisfaction_rate']:.1%})")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        print(f"\n📊 TEST SUITE BREAKDOWN")
        for suite_name, suite_results in report["test_suite_results"].items():
            if "test_summary" in suite_results:
                suite_summary = suite_results["test_summary"]
                print(f"  {suite_name.title()}:")
                print(f"    Tests: {suite_summary.get('total_tests', 0)}")
                print(f"    Success Rate: {suite_summary.get('success_rate', 0):.1%}")
                print(f"    Constraint Satisfaction: {suite_summary.get('constraint_satisfaction_rate', 0):.1%}")
        
        print(f"\n📁 Master report saved to: {self.output_dir / 'master_test_report.json'}")
        
        # Print recommendations
        self.print_recommendations(report)
    
    def print_recommendations(self, report: Dict[str, Any]) -> None:
        """Print recommendations based on test results."""
        summary = report["master_test_summary"]
        
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 30)
        
        if summary["overall_success_rate"] >= 0.9:
            print("✅ Excellent! The bridge system is performing very well.")
        elif summary["overall_success_rate"] >= 0.8:
            print("✅ Good performance! Minor optimizations may be beneficial.")
        elif summary["overall_success_rate"] >= 0.7:
            print("⚠️ Moderate performance. Consider investigating failed tests.")
        else:
            print("❌ Poor performance. Significant issues need to be addressed.")
        
        if summary["overall_constraint_satisfaction_rate"] >= 0.8:
            print("✅ Constraint satisfaction is excellent.")
        elif summary["overall_constraint_satisfaction_rate"] >= 0.6:
            print("⚠️ Constraint satisfaction is moderate. Consider tuning parameters.")
        else:
            print("❌ Constraint satisfaction is poor. Review optimization logic.")
        
        # Performance recommendations
        if "performance" in report["test_suite_results"]:
            perf_results = report["test_suite_results"]["performance"]
            if "performance_statistics" in perf_results:
                perf_stats = perf_results["performance_statistics"]
                avg_duration = perf_stats.get("average_duration", 0)
                if avg_duration > 10:
                    print("⚠️ Average test duration is high. Consider performance optimization.")
                elif avg_duration > 5:
                    print("ℹ️ Average test duration is moderate. Monitor performance.")
                else:
                    print("✅ Test performance is excellent.")
        
        # Genre-specific recommendations
        if "real_music" in report["test_suite_results"]:
            real_music_results = report["test_suite_results"]["real_music"]
            if "genre_performance" in real_music_results:
                genre_perf = real_music_results["genre_performance"]
                for genre, stats in genre_perf.items():
                    success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                    if success_rate < 0.7:
                        print(f"⚠️ {genre.title()} genre tests show low success rate. Consider genre-specific tuning.")
        
        print(f"\n🎉 Master test suite completed successfully!")
        print(f"   Check individual test reports for detailed analysis.")
        print(f"   All results saved to: {self.output_dir}")


def main():
    """Main function to run master test suite."""
    parser = argparse.ArgumentParser(
        description="Master Test Runner for AI RemixMate Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/bridge/master_test_runner.py --all
  
  # Run quick tests
  python scripts/bridge/master_test_runner.py --quick
  
  # Run specific test types
  python scripts/bridge/master_test_runner.py --tests comprehensive performance
  
  # Run with custom output directory
  python scripts/bridge/master_test_runner.py --all --output-dir runs/custom_test
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--quick", action="store_true", help="Run quick tests for rapid validation")
    parser.add_argument("--tests", nargs="+", choices=["comprehensive", "performance", "real_music"], 
                       help="Run specific test types")
    parser.add_argument("--output-dir", default="runs/master_test", help="Output directory for test results")
    parser.add_argument("--no-music-library", action="store_true", help="Skip creating music library for real music tests")
    
    args = parser.parse_args()
    
    if not any([args.all, args.quick, args.tests]):
        parser.print_help()
        return 1
    
    # Create master test runner
    master_runner = MasterTestRunner(args.output_dir)
    
    try:
        if args.all:
            # Run all tests
            report = master_runner.run_all_tests(
                create_music_library=not args.no_music_library
            )
        elif args.quick:
            # Run quick tests
            report = master_runner.run_quick_tests()
        elif args.tests:
            # Run specific tests
            report = master_runner.run_specific_tests(args.tests)
        
        print(f"\n✅ Master test suite completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Master test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
