#!/usr/bin/env python3
"""
Convenience script for running common pipeline operations.

This script provides easy-to-use commands for running different parts
of the ML pipeline without remembering all the command-line arguments.
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description='ML Pipeline Runner')
    parser.add_argument('command', choices=[
        'setup', 'train', 'continuous', 'drift', 'quick-demo'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        # Just initialize the environment
        cmd = [sys.executable, 'main_pipeline.py', '--mode', 'init']
        run_command(cmd, "Setting up ML environment")
        
    elif args.command == 'train':
        # Initialize and train baseline model
        cmd = [sys.executable, 'main_pipeline.py', '--mode', 'train', '--months', '4']
        run_command(cmd, "Training baseline model")
        
    elif args.command == 'continuous':
        # Run full continuous learning pipeline
        cmd = [sys.executable, 'main_pipeline.py', '--mode', 'continuous', '--months', '4']
        run_command(cmd, "Running continuous learning pipeline")
        
    elif args.command == 'drift':
        # Run drift simulation
        cmd = [sys.executable, 'main_pipeline.py', '--mode', 'drift', '--months', '4']
        run_command(cmd, "Running data drift simulation")
        
    elif args.command == 'quick-demo':
        # Quick demo with limited iterations
        cmd = [sys.executable, 'main_pipeline.py', '--mode', 'continuous', 
               '--months', '3', '--max-iterations', '3']
        run_command(cmd, "Running quick demo (3 months, 3 iterations)")

if __name__ == "__main__":
    main()
