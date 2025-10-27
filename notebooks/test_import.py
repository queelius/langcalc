#!/usr/bin/env python3
"""Test script to verify imports work correctly."""

import sys
import os

# Method 1: Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.model_algebra import AlgebraicModel
    print("✅ Successfully imported AlgebraicModel")
    print(f"   Module location: {AlgebraicModel.__module__}")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Python path: {sys.path[:3]}")