"""
SFT Data Factory: Self-Instruct Pipeline with LLM-as-a-Judge

SOTA References:
- Self-Instruct (Wang et al., 2022)
- UltraChat (Ding et al., 2023)
- Alpaca (Stanford, 2023)

This module generates high-quality instruction-following data for fine-tuning
a Literary Critic persona into the model.
"""
import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from src.rag.llm import LLMFactory
from src.infra.utils import setup_logger

logger = setup_logger(__name__)


class SFTDataGenerator:
    """
    Generates (Query, Response) pairs from raw reviews using Self-Instruct.
    """
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.llm = LLMFactory.create(provider=provider, api_key=api_key, temperature=0.7)
        
    def _sample_seed_reviews(self, path: str, n: int = 100, min_length: int = 200) -> List[Dict]:
        """Sample high-quality seed reviews."""
        seeds = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                isbn, review = parts[0], parts[1]
                if len(review) >= min_length:
                    seeds.append({"isbn": isbn, "review": review})
        
        # Random sample
        if len(seeds) > n:
            seeds = random.sample(seeds, n)
        logger.info(f"Sampled {len(seeds)} seed reviews")
        return seeds
    
    def _evolve_instruction(self, review: str) -> Optional[str]:
        """
        Self-Instruct Step 1: Generate a user question that would prompt this review.
        """
        prompt = f"""You are helping create training data for a book recommendation AI.

Given this enthusiastic book review, generate a realistic USER QUESTION that would have prompted such a recommendation. The question should be natural, like what a real person would type into a book search.

REVIEW:
\"\"\"{review[:500]}\"\"\"

Generate ONLY the user question, nothing else. Be creative and natural."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip().strip('"')
        except Exception as e:
            logger.error(f"Instruction evolution failed: {e}")
            return None
    
    def _transform_response(self, review: str, query: str) -> Optional[str]:
        """
        Self-Instruct Step 2: Transform review into AI assistant response style.
        """
        prompt = f"""You are a passionate Literary Critic AI assistant. 

A user asked: "{query}"

A human reviewer wrote this response:
\"\"\"{review[:600]}\"\"\"

Rewrite this as YOUR response to the user. Keep the emotional depth, specific evidence, and critical insight. But speak as a helpful AI book concierge, not as a random reviewer. 

Your response (be enthusiastic but professional):"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Response transformation failed: {e}")
            return None
    
    def generate_dataset(
        self, 
        review_path: str, 
        output_path: str, 
        n_samples: int = 100
    ) -> int:
        """
        Main pipeline: Generate SFT dataset.
        
        Returns: Number of successfully generated samples.
        """
        seeds = self._sample_seed_reviews(review_path, n=n_samples * 2)  # Over-sample
        
        dataset = []
        for seed in seeds:
            if len(dataset) >= n_samples:
                break
                
            # Step 1: Evolve instruction
            query = self._evolve_instruction(seed["review"])
            if not query:
                continue
            
            # Step 2: Transform response
            response = self._transform_response(seed["review"], query)
            if not response:
                continue
            
            dataset.append({
                "instruction": query,
                "input": "",  # No additional input for simple QA
                "output": response,
                "source_isbn": seed["isbn"]
            })
            
            if len(dataset) % 10 == 0:
                logger.info(f"Generated {len(dataset)} / {n_samples} samples")
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(dataset)} samples to {output_path}")
        return len(dataset)


class LLMJudge:
    """
    Quality filter using LLM-as-a-Judge pattern.
    Scores generated dialogues on multiple dimensions.
    """
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.llm = LLMFactory.create(provider=provider, api_key=api_key, temperature=0.1)
        
    def score(self, query: str, response: str) -> Dict:
        """
        Score a (query, response) pair on multiple dimensions.
        Returns: {"empathy": int, "specificity": int, "critique_depth": int, "avg": float}
        """
        prompt = f"""You are evaluating the quality of an AI book recommendation response.

USER QUESTION: "{query}"

AI RESPONSE:
\"\"\"{response}\"\"\"

Rate the response on these dimensions (1-10 each):
1. EMPATHY: Does it understand and connect with what the user is looking for?
2. SPECIFICITY: Does it mention concrete details (plot points, themes, comparisons)?
3. CRITIQUE_DEPTH: Does it offer genuine literary insight, not just generic praise?

Respond in JSON format ONLY:
{{"empathy": X, "specificity": Y, "critique_depth": Z}}"""

        try:
            result = self.llm.invoke(prompt)
            # Parse JSON from response
            import re
            match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                scores["avg"] = (scores["empathy"] + scores["specificity"] + scores["critique_depth"]) / 3
                return scores
        except Exception as e:
            logger.error(f"Judge scoring failed: {e}")
        
        return {"empathy": 0, "specificity": 0, "critique_depth": 0, "avg": 0}
    
    def filter_dataset(
        self, 
        input_path: str, 
        output_path: str, 
        threshold: float = 7.0
    ) -> Tuple[int, int]:
        """
        Filter dataset keeping only high-quality samples.
        
        Returns: (kept_count, total_count)
        """
        kept = []
        total = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                item = json.loads(line)
                scores = self.score(item["instruction"], item["output"])
                
                if scores["avg"] >= threshold:
                    item["quality_scores"] = scores
                    kept.append(item)
                    
                if total % 10 == 0:
                    logger.info(f"Judged {total} samples, kept {len(kept)}")
        
        # Save filtered
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Filtered: {len(kept)} / {total} passed (threshold={threshold})")
        return len(kept), total


# CLI Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFT Data Generator")
    parser.add_argument("--mode", choices=["generate", "judge"], required=True)
    parser.add_argument("--n", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--provider", default="mock", help="LLM provider (openai/ollama/mock)")
    parser.add_argument("--api-key", default=None, help="API key for provider")
    args = parser.parse_args()
    
    if args.mode == "generate":
        generator = SFTDataGenerator(provider=args.provider, api_key=args.api_key)
        generator.generate_dataset(
            review_path="data/review_highlights.txt",
            output_path="data/sft/raw_generated.jsonl",
            n_samples=args.n
        )
    elif args.mode == "judge":
        judge = LLMJudge(provider=args.provider, api_key=args.api_key)
        judge.filter_dataset(
            input_path="data/sft/raw_generated.jsonl",
            output_path="data/sft/filtered_high_quality.jsonl",
            threshold=7.0
        )
