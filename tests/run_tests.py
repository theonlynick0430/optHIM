#!/usr/bin/env python
"""
Run all tests for the optimization library.
"""
import pytest
import sys

if __name__ == "__main__":
    # Run all tests
    sys.exit(pytest.main(["-v"])) 