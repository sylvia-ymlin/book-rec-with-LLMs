"""
Unit tests for recall layer: fusion RRF, ItemCF, Swing, PopularityRecall, UserCF.
"""
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from src.recall.fusion import _merge_config, RecallFusion, DEFAULT_CHANNEL_CONFIG
from src.recall.itemcf import ItemCF
from src.recall.swing import Swing
from src.recall.popularity import PopularityRecall
from src.recall.usercf import UserCF


class TestMergeConfig:
    """Test _merge_config deep-merge logic."""

    def test_merge_empty_override(self):
        """No override returns copy of default."""
        result = _merge_config(DEFAULT_CHANNEL_CONFIG, None)
        assert result == {k: dict(v) for k, v in DEFAULT_CHANNEL_CONFIG.items()}

    def test_merge_enable_override(self):
        """Override enables a channel."""
        override = {"usercf": {"enabled": True}}
        result = _merge_config(DEFAULT_CHANNEL_CONFIG, override)
        assert result["usercf"]["enabled"] is True
        assert result["itemcf"]["enabled"] is True  # unchanged

    def test_merge_weight_override(self):
        """Override weight for a channel."""
        override = {"popularity": {"weight": 1.0}}
        result = _merge_config(DEFAULT_CHANNEL_CONFIG, override)
        assert result["popularity"]["weight"] == 1.0

    def test_merge_new_channel(self):
        """Override can add new channel."""
        override = {"new_channel": {"enabled": True, "weight": 0.5}}
        result = _merge_config(DEFAULT_CHANNEL_CONFIG, override)
        assert "new_channel" in result
        assert result["new_channel"]["enabled"] is True


class TestRecallFusionRRF:
    """Test RRF scoring in RecallFusion._add_to_candidates."""

    def test_rrf_add_single_channel(self):
        """RRF: higher rank contributes more. score = weight / (k + rank + 1)."""
        fusion = RecallFusion(rrf_k=60)
        fusion.models_loaded = True  # skip load_models
        candidates = defaultdict(float)
        recs = [("isbn1", 1.0), ("isbn2", 0.9), ("isbn3", 0.8)]
        fusion._add_to_candidates(candidates, recs, weight=1.0)
        # rank 0: 1/(60+1)=0.0164, rank 1: 1/62, rank 2: 1/63
        assert "isbn1" in candidates
        assert "isbn2" in candidates
        assert "isbn3" in candidates
        assert candidates["isbn1"] > candidates["isbn2"] > candidates["isbn3"]

    def test_rrf_empty_recs(self):
        """Empty recs leaves candidates unchanged."""
        fusion = RecallFusion()
        candidates = {}
        fusion._add_to_candidates(candidates, [], weight=1.0)
        assert len(candidates) == 0

    def test_rrf_multi_channel_aggregation(self):
        """Items in multiple channels get summed RRF scores."""
        fusion = RecallFusion(rrf_k=60)
        fusion.models_loaded = True
        candidates = defaultdict(float)
        fusion._add_to_candidates(candidates, [("A", 1.0), ("B", 0.9)], weight=1.0)
        fusion._add_to_candidates(candidates, [("B", 1.0), ("A", 0.8)], weight=0.5)
        # A: rank 0 from ch1 + rank 1 from ch2; B: rank 1 from ch1 + rank 0 from ch2
        assert candidates["A"] > 0
        assert candidates["B"] > 0
        assert candidates["A"] != candidates["B"]  # different positions


class TestItemCF:
    """Test ItemCF fit and recommend with synthetic data."""

    @pytest.fixture
    def temp_data_dir(self):
        """Temp dir for ItemCF DB."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_fit_synthetic_data(self, temp_data_dir):
        """ItemCF.fit builds matrix from synthetic DataFrame."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "isbn": ["A", "B", "C", "B", "C", "D"],
            "rating": [5, 4, 3, 4, 5, 4],
            "timestamp": [1, 2, 3, 1, 2, 3],
        })
        model_dir = Path(temp_data_dir) / "model"
        data_dir = Path(temp_data_dir) / "rec"
        data_dir.mkdir(parents=True)
        itemcf = ItemCF(data_dir=str(data_dir), save_dir=str(model_dir))
        itemcf.fit(df, top_k_sim=10)
        db_path = data_dir.parent / "recall_models.db"
        assert db_path.exists()
        assert hasattr(itemcf, "_sim_matrix")
        # u1: A->B->C; u2: B->C->D. Co-occurrences: (A,B), (A,C), (B,C), (B,D), (C,D)
        assert len(itemcf._sim_matrix) > 0

    def test_recommend_empty_history(self, temp_data_dir):
        """ItemCF.recommend returns [] for user with no history."""
        data_dir = Path(temp_data_dir) / "rec"
        data_dir.mkdir(parents=True)
        itemcf = ItemCF(data_dir=str(data_dir))
        result = itemcf.recommend("nonexistent_user", top_k=5)
        assert result == []


class TestSwing:
    """Test Swing fit and recommend with synthetic data."""

    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_fit_and_recommend(self, temp_dirs):
        """Swing.fit builds matrix; recommend returns items not in history."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "isbn": ["A", "B", "C", "B", "C", "D"],
            "rating": [5, 4, 3, 4, 5, 4],
            "timestamp": [1, 2, 3, 1, 2, 3],
        })
        data_dir = temp_dirs / "rec"
        model_dir = temp_dirs / "model"
        data_dir.mkdir(parents=True)
        swing = Swing(data_dir=str(data_dir), save_dir=str(model_dir))
        swing.fit(df, top_k_sim=10)
        assert len(swing.sim_matrix) > 0
        recs = swing.recommend("u1", top_k=5)
        assert isinstance(recs, list)
        # u1 history: A, B, C. Recs should exclude those.
        rec_items = [r[0] for r in recs]
        assert "A" not in rec_items
        assert "B" not in rec_items
        assert "C" not in rec_items


class TestPopularityRecall:
    """Test PopularityRecall fit and recommend."""

    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_fit_and_recommend(self, temp_dirs):
        """PopularityRecall returns top-K by count."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2", "u3", "u1", "u2"],
            "isbn": ["A", "A", "A", "B", "B"],
            "rating": [5, 4, 3, 4, 5],
        })
        data_dir = temp_dirs / "rec"
        model_dir = temp_dirs / "model"
        data_dir.mkdir(parents=True)
        pop = PopularityRecall(data_dir=str(data_dir), save_dir=str(model_dir))
        pop.fit(df)
        assert len(pop.hot_items) > 0
        assert pop.hot_items[0] == "A"  # A has count 3, B has 2
        recs = pop.recommend(top_k=5)
        assert len(recs) <= 5
        assert all(isinstance(r, (list, tuple)) and len(r) == 2 for r in recs)


class TestUserCF:
    """Test UserCF fit and recommend with synthetic data."""

    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_fit_and_recommend(self, temp_dirs):
        """UserCF.fit builds u2u sim; recommend returns items from similar users."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "isbn": ["A", "B", "B", "C", "A"],
            "rating": [5, 4, 4, 5, 3],
        })
        data_dir = temp_dirs / "rec"
        model_dir = temp_dirs / "model"
        data_dir.mkdir(parents=True)
        usercf = UserCF(data_dir=str(data_dir), save_dir=str(model_dir))
        usercf.fit(df)
        assert len(usercf.u2u_sim) > 0
        # u1 shares B with u2; u2 has C. So u1 might get C recommended.
        recs = usercf.recommend("u1", top_k=5)
        assert isinstance(recs, list)
        if recs:
            assert all(isinstance(r, (list, tuple)) and len(r) == 2 for r in recs)

    def test_recommend_unknown_user(self, temp_dirs):
        """UserCF.recommend returns [] for unknown user."""
        data_dir = temp_dirs / "rec"
        model_dir = temp_dirs / "model"
        data_dir.mkdir(parents=True)
        usercf = UserCF(data_dir=str(data_dir), save_dir=str(model_dir))
        usercf.fit(pd.DataFrame({"user_id": ["u1"], "isbn": ["A"], "rating": [5]}))
        assert usercf.recommend("unknown_user", top_k=5) == []
