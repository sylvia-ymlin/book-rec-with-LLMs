"""
Locust load test for Book Recommender API.

Simulates concurrent HTTP requests to measure real-world throughput.
Run API server first, then:

    pip install locust
    locust -f benchmarks/locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to drive the load test.
"""

import random
from locust import HttpUser, task, between

# Mirror TEST_QUERIES from benchmark.py for consistency
TEST_QUERIES = [
    "a romantic comedy set in New York",
    "a philosophical novel about the meaning of life",
    "a fast-paced thriller with plot twists",
    "a coming-of-age story about friendship and loss",
    "a historical fiction set during World War II",
    "a science fiction story about space exploration",
    "a mystery novel with an unreliable narrator",
    "a fantasy epic with dragons and magic",
    "a memoir about overcoming adversity",
    "a literary fiction exploring family dynamics",
]


class RecommenderUser(HttpUser):
    """Simulates a user hitting the recommendation API."""

    wait_time = between(0.5, 2.0)  # 0.5–2s between requests

    @task(10)
    def recommend(self):
        """Primary: POST /recommend."""
        q = random.choice(TEST_QUERIES)
        self.client.post(
            "/recommend",
            json={"query": q, "category": "All"},
        )

    @task(1)
    def health(self):
        """Occasional health check."""
        self.client.get("/health")
