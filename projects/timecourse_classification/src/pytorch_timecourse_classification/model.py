"""Backward-compatible entry point for the time-course classifier."""

from .models.extended_cnns import ExtendedCNNClassifier


TimecourseClassifier = ExtendedCNNClassifier

__all__ = ["ExtendedCNNClassifier", "TimecourseClassifier"]
