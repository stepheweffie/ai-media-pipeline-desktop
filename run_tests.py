#!/usr/bin/env python3
"""
Test Runner for Lecture Processing System
Runs various test suites with clear output and reporting
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def print_banner(title):
    """Print a nice banner for test sections"""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_summary(title, success, duration, details=None):
    """Print test summary"""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n📋 {title}: {status} ({duration:.1f}s)")
    if details:
        for detail in details:
            print(f"   {detail}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print_banner("Checking Dependencies")
    
    required_modules = [
        'unittest',
        'tempfile',
        'subprocess',
        'pathlib',
        'time',
        'psutil'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"❌ {module}")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install psutil")
        return False
    
    print("\n✅ All dependencies available")
    return True

def run_basic_tests():
    """Run basic functionality tests"""
    print_banner("Basic Functionality Tests")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, 'test_lecture_processor.py', '--basic'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        details = []
        if "Ran" in result.stdout:
            # Extract test count from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Ran" in line:
                    details.append(line.strip())
                    break
        
        print_summary("Basic Tests", success, duration, details)
        return success
        
    except FileNotFoundError:
        print("❌ test_lecture_processor.py not found")
        return False
    except Exception as e:
        print(f"❌ Error running basic tests: {e}")
        return False

def run_full_tests():
    """Run comprehensive test suite"""
    print_banner("Comprehensive Test Suite")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, 'test_lecture_processor.py'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        details = []
        if "Ran" in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if "Ran" in line or "FAILED" in line or "ERROR" in line:
                    details.append(line.strip())
        
        print_summary("Full Test Suite", success, duration, details)
        return success
        
    except Exception as e:
        print(f"❌ Error running full tests: {e}")
        return False

def run_performance_tests():
    """Run performance and benchmarking tests"""
    print_banner("Performance Tests")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, 'test_performance.py'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print_summary("Performance Tests", success, duration)
        return success
        
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        return False

def run_benchmark_only():
    """Run just the benchmark without unit tests"""
    print_banner("Performance Benchmark")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, 'test_performance.py', '--benchmark-only'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print_summary("Benchmark", success, duration)
        return success
        
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return False

def test_cli_commands():
    """Test CLI functionality"""
    print_banner("CLI Command Tests")
    
    tests = [
        ("Stats Command", ['python', 'lecture_cli.py', 'stats']),
        ("Help Command", ['python', 'lecture_cli.py', '--help'])
    ]
    
    results = []
    for test_name, command in tests:
        start_time = time.time()
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, 
                timeout=30, cwd=os.path.dirname(__file__)
            )
            duration = time.time() - start_time
            success = result.returncode == 0
            
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'} ({duration:.1f}s)")
            if not success and result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            
            results.append(success)
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name}: TIMEOUT")
            results.append(False)
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    overall_success = all(results)
    print_summary("CLI Tests", overall_success, sum([30 if not r else 1 for r in results]))
    return overall_success

def run_config_validation():
    """Test configuration validation"""
    print_banner("Configuration Tests")
    
    start_time = time.time()
    try:
        # Test config loading
        import config
        
        # Test basic config properties
        tests = [
            ("Config module loads", lambda: config is not None),
            ("Has OPENAI_API_KEY", lambda: hasattr(config, 'OPENAI_API_KEY')),
            ("Has validate_config", lambda: hasattr(config, 'validate_config')),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                print(f"{'✅' if result else '❌'} {test_name}")
                results.append(result)
            except Exception as e:
                print(f"❌ {test_name}: {e}")
                results.append(False)
        
        duration = time.time() - start_time
        success = all(results)
        
        print_summary("Configuration", success, duration, 
                     [f"Passed: {sum(results)}/{len(results)} tests"])
        return success
        
    except Exception as e:
        print(f"❌ Config validation error: {e}")
        return False

def system_info():
    """Display system information"""
    print_banner("System Information")
    
    try:
        import platform
        import sys
        
        print(f"🖥️  Platform: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version.split()[0]} ({sys.executable})")
        print(f"📁 Working Directory: {os.getcwd()}")
        
        # Check for FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"🎬 FFmpeg: {version_line}")
            else:
                print("⚠️  FFmpeg: Not found or not working")
        except:
            print("⚠️  FFmpeg: Not available")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free / (1024**3)
            print(f"💾 Free Disk Space: {free_gb:.1f}GB")
        except:
            print("💾 Disk Space: Unable to check")
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            print(f"🧠 Available Memory: {available_gb:.1f}GB")
        except:
            print("🧠 Memory: Unable to check")
            
    except Exception as e:
        print(f"❌ Error getting system info: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Runner for Lecture Processing System')
    parser.add_argument('--basic', action='store_true', 
                       help='Run only basic tests (fastest)')
    parser.add_argument('--full', action='store_true',
                       help='Run comprehensive test suite')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark only')
    parser.add_argument('--cli', action='store_true',
                       help='Test CLI commands')
    parser.add_argument('--config', action='store_true',
                       help='Test configuration')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--info', action='store_true',
                       help='Show system information only')
    
    args = parser.parse_args()
    
    # Show system info
    if args.info:
        system_info()
        return 0
    
    # Default to basic tests if no specific test chosen
    if not any([args.basic, args.full, args.performance, args.benchmark, 
               args.cli, args.config, args.all]):
        args.basic = True
    
    print("🧪 Lecture Processing Test Runner")
    print(f"📁 Working in: {os.path.dirname(__file__) or '.'}")
    
    system_info()
    
    # Check dependencies first
    if not check_dependencies():
        return 1
    
    results = []
    total_start = time.time()
    
    # Run selected tests
    if args.all or args.config:
        results.append(run_config_validation())
    
    if args.all or args.cli:
        results.append(test_cli_commands())
    
    if args.all or args.basic:
        results.append(run_basic_tests())
    
    if args.all or args.full:
        results.append(run_full_tests())
    
    if args.benchmark:
        results.append(run_benchmark_only())
    elif args.all or args.performance:
        results.append(run_performance_tests())
    
    total_duration = time.time() - total_start
    
    # Final summary
    print_banner("Test Summary")
    passed = sum(results)
    total = len(results)
    success = all(results)
    
    print(f"📊 Overall Results:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ⏱️  Total Time: {total_duration:.1f}s")
    print(f"   🎯 Status: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
    
    if success:
        print("\n🎉 All tests passed! Your lecture processing system is ready to go.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above for details.")
        print("   Common issues:")
        print("   - Missing API keys in .env file")
        print("   - Missing dependencies (run: pip install -r requirements.txt)")
        print("   - Configuration issues")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
