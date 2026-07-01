"""Test data generator for GraphRAG-KG.

Generates synthetic document corpora with known ground truth entities,
relationships, and communities for pipeline validation.
"""

from graphrag_kg.data.ground_truth import GroundTruth, Entity, Relationship, Community, TestQuery
from graphrag_kg.data.generator import TestDataGenerator

__all__ = [
    "TestDataGenerator",
    "GroundTruth",
    "Entity",
    "Relationship",
    "Community",
    "TestQuery",
]
