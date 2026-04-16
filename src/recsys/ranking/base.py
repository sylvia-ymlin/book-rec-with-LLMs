from abc import ABC, abstractmethod
from typing import List, Optional, Any
import numpy as np


class BaseRanker(ABC):
    """
    Abstract base class for all ranking models.
    Provides a unified interface for model loading and score prediction.
    """

    @abstractmethod
    def load(self) -> bool:
        """
        Load model weights and required metadata.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def predict(
        self,
        user_id: str,
        candidate_items: List[str],
        features_df: Optional[Any] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate ranking scores for candidate items for a given user.

        Args:
            user_id: The target user identifier.
            candidate_items: List of ISBNs to rank.
            features_df: Pre-computed features DataFrame (optional).
            **kwargs: Implementation-specific context (e.g., override_hist).

        Returns:
            numpy array of scores.
        """
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return the list of feature names required by this ranker."""
        pass
