"""Shared structural contracts for dictionaries crossing component boundaries.

Records with fixed fields use TypedDict. Dictionaries keyed by scene, filename,
metric name or injected component remain mappings with concrete value types.
"""
