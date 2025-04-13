#!/usr/bin/env python3
"""
Test runner for AstroShield backend tests

This script runs all unit and integration tests for the AstroShield backend.
"""

import pytest
import os
import sys

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """Run all tests"""
    # Get the current directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the tests
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
        "-xvs",  # Exit on first failure, verbose, don't capture output
        test_dir  # Run all tests in the tests directory
    ]
    
    return pytest.main(args)

def run_unit_tests():
    """Run only unit tests"""
    # Get the unit tests directory
    unit_test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "unit"
    )
    
    # Run the tests
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
        "-xvs",  # Exit on first failure, verbose, don't capture output
        unit_test_dir  # Run all tests in the unit tests directory
    ]
    
    return pytest.main(args)

def run_integration_tests():
    """Run only integration tests"""
    # Get the integration tests directory
    integration_test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "integration"
    )
    
    # Run the tests
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
        "-xvs",  # Exit on first failure, verbose, don't capture output
        integration_test_dir  # Run all tests in the integration tests directory
    ]
    
    return pytest.main(args)

if __name__ == "__main__":
    print("===== RUNNING ASTROSHIELD BACKEND TESTS =====")
    
    # Check if only a specific test suite should be run
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            print("\n===== RUNNING UNIT TESTS =====")
            exit_code = run_unit_tests()
        elif test_type == "integration":
            print("\n===== RUNNING INTEGRATION TESTS =====")
            exit_code = run_integration_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python test_runner.py [unit|integration]")
            exit_code = 1
    else:
        # Run all tests by default
        print("\n===== RUNNING UNIT TESTS =====")
        exit_code_unit = run_unit_tests()
        
        print("\n===== RUNNING INTEGRATION TESTS =====")
        exit_code_integration = run_integration_tests()
        
        # Exit with non-zero code if any test suite failed
        exit_code = 1 if exit_code_unit != 0 or exit_code_integration != 0 else 0
    
    sys.exit(exit_code) 