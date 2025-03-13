#!/usr/bin/env python3

import json
import requests
import sys
from rich.console import Console
from rich.table import Table
from rich import box
import time

# Configuration
API_URL = "http://localhost:8000"
SAMPLE_DATA_FILE = "sample_data.json"

console = Console()

def test_api_health():
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_URL}/services/orbit-identification/openapi.json")
        if response.status_code == 200:
            console.print("[green]API is running and healthy.[/green]")
            return True
        else:
            console.print(f"[red]API returned status code {response.status_code}.[/red]")
            return False
    except requests.exceptions.ConnectionError:
        console.print("[red]Cannot connect to the API. Make sure it's running.[/red]")
        console.print("[yellow]Run: uvicorn main:app --host 0.0.0.0 --port 8000[/yellow]")
        return False

def load_sample_data():
    """Load sample data from JSON file."""
    try:
        with open(SAMPLE_DATA_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        console.print(f"[red]Sample data file {SAMPLE_DATA_FILE} not found.[/red]")
        return None
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON in sample data file {SAMPLE_DATA_FILE}.[/red]")
        return None

def test_single_identification(record):
    """Test single orbit identification endpoint."""
    console.print(f"\n[bold blue]Testing single identification for {record.get('ENTITY_ID', 'Unknown')}[/bold blue]")
    
    try:
        response = requests.post(f"{API_URL}/identify", json=record)
        if response.status_code == 200:
            data = response.json()
            tags = data.get("TAGS", [])
            
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Property", style="dim")
            table.add_column("Value")
            
            # Add orbital elements to table
            for key, value in record.items():
                table.add_row(key, str(value))
            
            console.print(table)
            
            # Display orbit tags
            tags_table = Table(show_header=True, header_style="bold green", box=box.SIMPLE)
            tags_table.add_column("Orbit Tags", style="green")
            
            for tag in tags:
                tags_table.add_row(tag)
            
            console.print(tags_table)
            return True
        else:
            console.print(f"[red]Error: {response.status_code}[/red]")
            console.print(response.json())
            return False
    except Exception as e:
        console.print(f"[red]Exception: {str(e)}[/red]")
        return False

def test_batch_identification(data):
    """Test batch orbit identification endpoint."""
    console.print("\n[bold blue]Testing batch identification[/bold blue]")
    
    try:
        response = requests.post(f"{API_URL}/identify-batch", json=data)
        if response.status_code == 200:
            result_data = response.json()
            records = result_data.get("RECORDS", [])
            
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Entity ID", style="dim")
            table.add_column("Orbit Type")
            table.add_column("Direction")
            table.add_column("Eccentricity Class")
            table.add_column("IADC Region")
            
            for i, record in enumerate(records):
                entity_id = data["RECORDS"][i].get("ENTITY_ID", f"Record {i+1}")
                tags = record.get("TAGS", [])
                
                # Categorize tags
                orbit_type = next((tag for tag in tags if tag in ["LEO", "VLEO", "MEO", "GSO", "GEO", "HIGH_EARTH_ORBIT"]), "Unknown")
                direction = next((tag for tag in tags if tag in ["PROGRADE", "RETROGRADE"]), "Unknown")
                ecc_class = next((tag for tag in tags if tag in ["CIRCULAR", "NEAR_CIRCULAR", "ELLIPTIC", "PARABOLIC", "HYPERBOLIC", "HIGHLY_ELLIPTICAL_ORBIT"]), "Unknown")
                iadc_region = next((tag for tag in tags if "IADC" in tag), "None")
                
                table.add_row(entity_id, orbit_type, direction, ecc_class, iadc_region)
            
            console.print(table)
            return True
        else:
            console.print(f"[red]Error: {response.status_code}[/red]")
            console.print(response.json())
            return False
    except Exception as e:
        console.print(f"[red]Exception: {str(e)}[/red]")
        return False

def main():
    """Main function to run tests."""
    console.print("[bold]Orbit Family Identification API Test Script[/bold]")
    
    # Check if API is running
    if not test_api_health():
        return
    
    # Load sample data
    data = load_sample_data()
    if not data:
        return
    
    # Test single identification for each record
    for record in data["RECORDS"]:
        test_single_identification(record)
        time.sleep(0.5)  # Small delay between requests
    
    # Test batch identification
    test_batch_identification(data)

if __name__ == "__main__":
    main() 