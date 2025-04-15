#!/usr/bin/env python3
"""
This script fixes backend issues in the AstroShield application:
1. Fixes AttributeError in main.py by ensuring shutdown_event is properly initialized
2. Fixes the trajectory router registration by updating the prefix
3. Ensures all dependencies are installed
"""

import os
import sys
import subprocess
import shutil

# ANSI color codes for better readability
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(message):
    print(f"\n{BOLD}{GREEN}=== {message} ==={RESET}\n")

def print_warning(message):
    print(f"{YELLOW}WARNING: {message}{RESET}")

def print_error(message):
    print(f"{RED}ERROR: {message}{RESET}")

def run_command(command, description=None):
    """Run a shell command and print its output"""
    if description:
        print(f"{BOLD}> {description}{RESET}")
    
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if process.stdout.strip():
        print(process.stdout.strip())
    
    if process.returncode != 0:
        print_error(f"Command failed with exit code {process.returncode}")
        if process.stderr.strip():
            print(process.stderr.strip())
        return False
    
    return True

def fix_backend_main_py():
    """Fix the AttributeError in main.py"""
    print_header("Fixing AttributeError in main.py")
    
    # Paths to check
    possible_paths = [
        "backend_fixed/app/main.py",
        "backend/app/main.py",
        "app/main.py"
    ]
    
    main_py_path = None
    for path in possible_paths:
        if os.path.exists(path):
            main_py_path = path
            break
    
    if not main_py_path:
        print_error("Could not find main.py in any of the expected locations")
        return False
    
    print(f"Found main.py at: {main_py_path}")
    
    # Create backup
    backup_path = f"{main_py_path}.bak"
    shutil.copy2(main_py_path, backup_path)
    print(f"Created backup at: {backup_path}")
    
    # Read the file
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Fix the shutdown_event initialization
    if "shutdown_event = threading.Event()" in content:
        print("shutdown_event is already properly initialized")
    else:
        # Find the line initializing shutdown_event
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "shutdown_event =" in line and "function" not in line:
                # Check if it's set to a function (causing the error)
                print(f"Found shutdown_event initialization at line {i+1}: {line}")
                lines[i] = "shutdown_event = threading.Event()"
                print(f"Updated to: {lines[i]}")
                break
        
        # Write the updated content
        with open(main_py_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print("Fixed shutdown_event initialization")
    
    return True

def fix_trajectory_router_prefix():
    """Fix the trajectory router registration prefix"""
    print_header("Fixing trajectory router prefix")
    
    # Paths to check
    possible_paths = [
        "backend_fixed/app/main.py",
        "backend/app/main.py",
        "app/main.py"
    ]
    
    main_py_path = None
    for path in possible_paths:
        if os.path.exists(path):
            main_py_path = path
            break
    
    if not main_py_path:
        print_error("Could not find main.py in any of the expected locations")
        return False
    
    # Read the file
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Check how trajectory router is registered
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "app.include_router(trajectory.router" in line:
            print(f"Found trajectory router registration at line {i+1}: {line}")
            
            # If it already has prefix="/api", keep it
            if 'prefix="/api"' in line or 'prefix="/api/' in line:
                print("Trajectory router is already correctly registered with /api prefix")
            else:
                # Update to use prefix="/api"
                lines[i] = 'app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])'
                print(f"Updated to: {lines[i]}")
    
    # Write the updated content
    with open(main_py_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return True

def install_dependencies():
    """Install missing dependencies in Docker container and local environment"""
    print_header("Installing missing dependencies")
    
    # Check if we're in Docker environment
    in_docker = os.path.exists("/.dockerenv")
    
    if in_docker:
        print("Running in Docker environment")
        run_command("pip install scipy psycopg2-binary", "Installing dependencies in Docker")
    else:
        # Try to install in Docker container
        result = run_command("docker exec asttroshield_v02-backend-1 pip install scipy psycopg2-binary", 
                           "Installing dependencies in backend container")
        if not result:
            print_warning("Failed to install dependencies in Docker container. Is Docker running?")
        
        # Also install locally
        run_command("pip install scipy psycopg2-binary", "Installing dependencies locally")
    
    return True

def restart_services():
    """Restart the backend services"""
    print_header("Restarting services")
    
    # Try to stop services first
    run_command("./stop_astroshield.sh", "Stopping AstroShield services")
    
    # Restart Docker containers if they exist
    if os.path.exists("docker-compose.yml"):
        run_command("docker-compose restart backend", "Restarting backend container")
    else:
        run_command("docker restart asttroshield_v02-backend-1", "Restarting backend container")
    
    # Start services
    run_command("./start_astroshield.sh", "Starting AstroShield services")
    
    return True

def main():
    """Main function that runs all fixes"""
    print_header("AstroShield Backend Fix Script")
    
    steps = [
        ("Fix AttributeError in main.py", fix_backend_main_py),
        ("Fix trajectory router prefix", fix_trajectory_router_prefix),
        ("Install dependencies", install_dependencies),
        ("Restart services", restart_services)
    ]
    
    results = []
    for step_name, step_func in steps:
        print_header(f"Step: {step_name}")
        try:
            result = step_func()
            results.append((step_name, result))
            if result:
                print(f"{GREEN}✓ {step_name} completed successfully{RESET}")
            else:
                print(f"{RED}✗ {step_name} failed{RESET}")
        except Exception as e:
            print_error(f"Exception occurred during {step_name}: {str(e)}")
            results.append((step_name, False))
    
    # Print summary
    print_header("Fix Summary")
    all_success = True
    for step_name, result in results:
        status = f"{GREEN}✓ Success" if result else f"{RED}✗ Failed"
        print(f"{status}{RESET}: {step_name}")
        all_success = all_success and result
    
    if all_success:
        print(f"\n{GREEN}All fixes were applied successfully!{RESET}")
        print("You should now be able to use the trajectory analysis feature.")
    else:
        print(f"\n{YELLOW}Some fixes failed. Check the logs above for details.{RESET}")
        print("You may need to manually fix the remaining issues.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 