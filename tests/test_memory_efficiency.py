import os
import psutil
import pytest
from src.core.metadata_store import MetadataStore
from src.recall.itemcf import ItemCF
from src.recommender import BookRecommender
from src.services.recommend_service import RecommendationService

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def test_memory_usage_loading():
    initial_mem = get_process_memory()
    print(f"\nInitial RAM: {initial_mem:.2f} MB")
    
    # 1. Initialize MetadataStore
    MetadataStore._instance = None
    store = MetadataStore()
    mem_after_store = get_process_memory()
    print(f"RAM after MetadataStore: {mem_after_store:.2f} MB (Delta: {mem_after_store - initial_mem:.2f} MB)")
    
    # 2. Initialize ItemCF (Old one took 1.4GB+ on disk, 7GB+ in RAM)
    itemcf = ItemCF()
    itemcf.load()
    mem_after_itemcf = get_process_memory()
    print(f"RAM after ItemCF Load: {mem_after_itemcf:.2f} MB (Delta: {mem_after_itemcf - mem_after_store:.2f} MB)")
    
    # 3. Initialize Recommender
    recommender = BookRecommender()
    mem_after_rec = get_process_memory()
    print(f"RAM after Recommender: {mem_after_rec:.2f} MB (Delta: {mem_after_rec - mem_after_itemcf:.2f} MB)")
    
    # 4. Initialize RecommendationService
    service = RecommendationService()
    service.load_resources()
    mem_after_service = get_process_memory()
    print(f"RAM after Service Load: {mem_after_service:.2f} MB (Delta: {mem_after_service - mem_after_rec:.2f} MB)")
    
    # Assertions
    # We expect each step to add very little RAM (certainly not GBs)
    # Most RAM will be taken by the Embedding model which is ~80-100MB
    assert mem_after_service - initial_mem < 1000  # Total data overhead should be < 1GB (miniLM embedding is small)

def test_itemcf_functionality():
    itemcf = ItemCF()
    # Test recommendation for a real user in migrations if possible, or just mock
    # Since we have data/recall_models.db, let's try a real query
    recs = itemcf.recommend("A1ZQ1LUQ9R6JHZ", top_k=5)
    print(f"\nItemCF Recs: {recs}")
    assert isinstance(recs, list)
    if recs:
        assert len(recs) <= 5
        assert isinstance(recs[0], (list, tuple))
        assert len(recs[0]) == 2 # (isbn, score)
