#!/usr/bin/env python3
"""
Script to synchronize version from pyproject.toml to CITATION.cff
"""
import re
import sys
from pathlib import Path


def read_version_from_pyproject():
    """Read version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        return None
    
    content = pyproject_path.read_text()
    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if not version_match:
        print("Error: Could not find version in pyproject.toml")
        return None
    
    return version_match.group(1)


def update_citation_version(new_version):
    """Update version in CITATION.cff"""
    citation_path = Path("CITATION.cff")
    if not citation_path.exists():
        print("Error: CITATION.cff not found")
        return False
    
    content = citation_path.read_text()
    
    # Replace version line
    updated_content = re.sub(
        r'version:\s*["\']([^"\']+)["\']',
        f'version: "{new_version}"',
        content
    )
    
    if content == updated_content:
        print(f"CITATION.cff already has version {new_version}")
        return True
    
    citation_path.write_text(updated_content)
    print(f"Updated CITATION.cff version to {new_version}")
    return True


def main():
    # Read version from pyproject.toml
    version = read_version_from_pyproject()
    if not version:
        sys.exit(1)
    
    # Update CITATION.cff
    if not update_citation_version(version):
        sys.exit(1)
    
    print("Version synchronization completed successfully")


if __name__ == "__main__":
    main()
