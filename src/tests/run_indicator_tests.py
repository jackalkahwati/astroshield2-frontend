"""Test runner for indicator models."""

import os
import sys
import pytest
import json
from datetime import datetime
from pathlib import Path

def run_tests():
    """Run all indicator model tests and generate reports."""
    # Create reports directory if it doesn't exist
    reports_dir = Path(__file__).parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for report files
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Run tests with pytest
    test_args = [
        '--verbose',
        '--cov=src',
        f'--cov-report=html:reports/coverage_{timestamp}',
        f'--html=reports/report_{timestamp}.html',
        '--self-contained-html'
    ]
    
    result = pytest.main(test_args)
    
    # Generate JSON report
    report = {
        'timestamp': timestamp,
        'exit_code': result,
        'success': result == 0,
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # Save JSON report
    report_path = reports_dir / f'indicator_test_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nTest reports saved to {reports_dir}")
    return result

if __name__ == '__main__':
    sys.exit(run_tests()) 