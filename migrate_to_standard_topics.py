#!/usr/bin/env python3
"""
Migration script to update code from old topic names to standard SDA topic names.
This script can be used to scan and optionally update Python files.
"""

import os
import re
import argparse
from typing import Dict, List, Tuple

# Mapping of old topic patterns to new standard topics
TOPIC_MAPPINGS = {
    # Direct string replacements
    'welders.ss0.sensor.observations': 'ss2.data.observation-track',
    'welders.ss0.sensor.heartbeat': 'ss0.sensor.heartbeat',
    'welders.ss0.collect.request': 'ss0.data.manifold.request',
    'welders.ss0.collect.response': 'ss0.data.manifold.response',
    
    'welders.ss1.target.update.request': 'ss1.request.state-vector-prediction',
    'welders.ss1.target.update.response': 'ss1.response.state-vector-prediction',
    'welders.ss1.target.model.insert': 'ss1.tmdb.object-inserted',
    'welders.ss1.target.model.update': 'ss1.tmdb.object-updated',
    
    'welders.ss2.uct.tracks': 'ss2.data.observation-track.true-uct',
    'welders.ss2.state.vectors': 'ss2.data.state-vector',
    'welders.ss2.orbit.determination': 'ss2.data.orbit-determination',
    'welders.ss2.catalog.correlation': 'ss2.data.state-vector.catalog-nominee',
    
    'welders.ss3.sensor.schedule': 'ss3.data.accesswindow',
    'welders.ss3.collection.plan': 'ss3.data.detectionprobability',
    'welders.ss3.surveillance.task': 'ss3.data.accesswindow',
    'welders.ss3.custody.task': 'ss3.data.detectionprobability',
    
    'welders.ss4.ccdm.indicators': 'ss4.ccdm.ccdm-db',
    'welders.ss4.object.interest.list': 'ss4.ccdm.ooi',
    'welders.ss4.anomaly.detection': 'ss4.indicators.maneuvers-detected',
    'welders.ss4.pattern.violation': 'ss4.indicators.orbit-oof',
    
    'welders.ss5.wez.prediction': 'ss5.pez-wez-prediction.kkv',
    'welders.ss5.intent.assessment': 'ss5.launch.intent-assessment',
    'welders.ss5.pursuit.detection': 'ss5.rpo.intent',
    'welders.ss5.threat.warning': 'ss5.launch.intent-assessment',
    
    'welders.ss6.defensive.coa': 'ss6.response-recommendation.on-orbit',
    'welders.ss6.mitigation.plan': 'ss6.risk-mitigation.optimal-maneuver',
    'welders.ss6.alert.operator': 'ss6.response-recommendation.on-orbit',
    'welders.ss6.action.recommendation': 'ss6.response-recommendation.launch',
    
    'welders.event.launch.detection': 'ss5.launch.detection',
    'welders.event.maneuver.detection': 'ss4.indicators.maneuvers-detected',
    'welders.event.proximity.alert': 'ss4.indicators.proximity-events-valid-remote-sense',
    'welders.event.separation.detected': 'ss5.separation.detection',
    
    # Uppercase variations
    'SS5.launch.trajectory': 'ss5.launch.trajectory',
    'SS5.launch.asat.assessment': 'ss5.launch.asat-assessment',
    'SS5.reentry.prediction': 'ss5.reentry.prediction',
    'SS4.attribution.orbital_attributes': 'ss4.attributes.orbital-attribution',
    'SS4.attribution.shape_change': 'ss4.indicators.amr-changes',
    'SS4.attribution.thermal_signature': 'ss4.indicators.stability-changed',
    
    # Other common patterns
    'AstroShield.heartbeat': 'ss0.sensor.heartbeat',
    'ccdm-events': 'ss4.ccdm.ccdm-db',
    'ccdm-alerts': 'ss4.ccdm.ooi',
    'ccdm-reports': 'ss4.attributes.orbital-attribution',
    'sda-conjunction-data': 'ss2.data.state-vector',
    'sda-tle-updates': 'ss2.data.elset.sgp4',
    'sda-ephemerides': 'ss2.data.ephemeris',
    'sda-maneuvers': 'ss4.indicators.maneuvers-detected',
}


def find_topic_references(file_path: str) -> List[Tuple[int, str, str]]:
    """Find all topic references in a file."""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            for old_topic, new_topic in TOPIC_MAPPINGS.items():
                # Look for direct string references
                if old_topic in line:
                    findings.append((line_num, old_topic, new_topic))
                
                # Look for topic references in quotes
                pattern = rf'["\']({re.escape(old_topic)})["\']'
                if re.search(pattern, line):
                    findings.append((line_num, old_topic, new_topic))
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return findings


def update_file(file_path: str, findings: List[Tuple[int, str, str]], dry_run: bool = True):
    """Update a file with new topic names."""
    if not findings:
        return
        
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Updating {file_path}:")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        # Sort findings by old topic length (longest first) to avoid partial replacements
        sorted_mappings = sorted(TOPIC_MAPPINGS.items(), key=lambda x: len(x[0]), reverse=True)
        
        for old_topic, new_topic in sorted_mappings:
            if old_topic in content:
                content = content.replace(old_topic, new_topic)
                print(f"  Replaced: {old_topic} -> {new_topic}")
                
        if not dry_run and content != original_content:
            # Create backup
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"  Created backup: {backup_path}")
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Updated: {file_path}")
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")


def scan_directory(directory: str, extensions: List[str], exclude_dirs: List[str]):
    """Scan directory for files containing old topic names."""
    all_findings = {}
    
    for root, dirs, files in os.walk(directory):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                findings = find_topic_references(file_path)
                if findings:
                    all_findings[file_path] = findings
                    
    return all_findings


def main():
    parser = argparse.ArgumentParser(
        description='Migrate code to use standard SDA Kafka topic names'
    )
    parser.add_argument(
        'directory',
        help='Directory to scan for files'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Actually update files (default is dry run)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.py', '.yaml', '.yml', '.json', '.sh'],
        help='File extensions to scan (default: .py .yaml .yml .json .sh)'
    )
    parser.add_argument(
        '--exclude-dirs',
        nargs='+',
        default=['venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'],
        help='Directories to exclude from scan'
    )
    
    args = parser.parse_args()
    
    print(f"Scanning {args.directory} for old topic names...")
    print(f"File extensions: {args.extensions}")
    print(f"Excluding directories: {args.exclude_dirs}")
    print(f"Mode: {'UPDATE' if args.update else 'DRY RUN'}\n")
    
    # Scan for findings
    all_findings = scan_directory(args.directory, args.extensions, args.exclude_dirs)
    
    if not all_findings:
        print("No old topic names found!")
        return
        
    # Report findings
    total_files = len(all_findings)
    total_refs = sum(len(findings) for findings in all_findings.values())
    
    print(f"Found {total_refs} references in {total_files} files:\n")
    
    for file_path, findings in all_findings.items():
        print(f"\n{file_path}:")
        for line_num, old_topic, new_topic in findings:
            print(f"  Line {line_num}: {old_topic} -> {new_topic}")
            
    # Update files if requested
    if args.update:
        print("\n" + "="*60)
        print("UPDATING FILES")
        print("="*60)
        
        for file_path, findings in all_findings.items():
            update_file(file_path, findings, dry_run=False)
    else:
        print("\n" + "="*60)
        print("DRY RUN COMPLETE")
        print("To actually update files, run with --update flag")
        print("="*60)


if __name__ == '__main__':
    main() 