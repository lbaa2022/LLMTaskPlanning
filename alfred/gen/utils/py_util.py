"""
Python utilities for ALFRED.
"""

def remove_spaces_and_lower(text):
    """Remove extra spaces and lowercase text."""
    if text is None:
        return ""
    return ' '.join(text.lower().split())
