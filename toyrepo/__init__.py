"""Toy-repo generator: builds tiny buggy Python packages for PatchTune-mini."""

from .generate import BUG_KINDS, BUG_BY_KIND, ToyRepo, generate_instance, random_instance

__all__ = ["BUG_KINDS", "BUG_BY_KIND", "ToyRepo", "generate_instance", "random_instance"]
