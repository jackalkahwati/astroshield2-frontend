#!/usr/bin/env python3
"""
Test script for the UDL client.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path("backend")
sys.path.append(str(backend_path.absolute()))

async def main():
    try:
        # Import the UDL client
        from app.services.udl_client import get_udl_client
        
        # Get UDL client
        print("Getting UDL client...")
        udl_client = await get_udl_client()
        
        # Test authentication
        print("Testing authentication...")
        authenticated = await udl_client.authenticate()
        print(f"Authentication result: {authenticated}")
        
        # Get state vectors
        print("Getting state vectors...")
        state_vectors = await udl_client.get_state_vectors(limit=3)
        
        print(f"Received {len(state_vectors)} state vectors:")
        for i, sv in enumerate(state_vectors):
            print(f"  {i+1}. {sv.id}: {sv.name}")
            print(f"     Position: x={sv.position['x']:.1f}, y={sv.position['y']:.1f}, z={sv.position['z']:.1f}")
            print(f"     Velocity: x={sv.velocity['x']:.1f}, y={sv.velocity['y']:.1f}, z={sv.velocity['z']:.1f}")
        
        print("\nUDL client test successful!")
        return True
    except Exception as e:
        print(f"Error testing UDL client: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
