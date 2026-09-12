"""Reproducible simulation campaign manager with provenance gates.

A small, domain-agnostic tool for running a simulation campaign, recording a
provenance manifest (code commit, input-deck hash, solver version) alongside
every result, and refusing to record a result as an accepted baseline when
its outputs moved but nothing that should explain a move (commit, input
deck, solver version) actually changed.
"""

__version__ = "0.1.0"
