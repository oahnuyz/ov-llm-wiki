import re
import string
import collections
from typing import List


class MetricsCalculator:
    @staticmethod
    def qa_token_usage(usage: dict) -> tuple[int, int, int]:
        """Read cumulative generation usage, including query embedding tokens.

        New records distinguish llm_total_tokens from the combined total_tokens.
        Legacy records used total_tokens for LLM usage alone. Judge, ingestion
        and Wiki construction are excluded.
        """
        def first_value(*keys):
            return next((int(usage[key]) for key in keys if usage.get(key) is not None), 0)

        input_tokens = first_value("prompt_tokens", "input_tokens", "total_input_tokens")
        output_tokens = first_value("completion_tokens", "output_tokens", "llm_output_tokens")
        embedding_tokens = int(usage.get("retrieval_embedding_tokens") or 0)
        llm_total = usage.get("llm_total_tokens")
        if llm_total is None:
            llm_total = usage.get("total_tokens")
            if llm_total is None:
                llm_total = input_tokens + output_tokens
        # The legacy input alias can already include embedding in normalized records.
        if any(usage.get(key) is not None for key in ("prompt_tokens", "input_tokens")):
            input_tokens += embedding_tokens
        total_tokens = int(llm_total) + embedding_tokens
        return input_tokens, output_tokens, total_tokens

    @staticmethod
    def embedding_usage_complete(result: dict) -> bool:
        usage = result.get("token_usage") or {}
        if "retrieval_embedding_usage_complete" in usage:
            return usage["retrieval_embedding_usage_complete"] is True
        # Old VikingBot records hard-coded embedding=0 even when search was used.
        return "openviking_search" not in ((result.get("vikingbot") or {}).get("tools_used_names") or [])

    @staticmethod
    def llm_usage_complete(result: dict) -> bool:
        usage = result.get("token_usage") or {}
        if "llm_usage_complete" in usage:
            return usage["llm_usage_complete"] is True
        return all(usage.get(key) is not None for key in
                   ("prompt_tokens", "completion_tokens", "total_tokens"))

    @staticmethod
    def usage_issues(result: dict) -> list[dict]:
        """Preserve request-level issues; identify missing metadata in older results."""
        issues = list((result.get("token_usage") or {}).get("usage_issues") or [])
        for kind, complete in (
            ("llm", MetricsCalculator.llm_usage_complete(result)),
            ("embedding", MetricsCalculator.embedding_usage_complete(result)),
        ):
            if not complete and not any(issue.get("kind") == kind for issue in issues):
                issues.append({"kind": kind, "reason": "usage_missing_or_incomplete"})
        return issues

    @staticmethod
    def average_qa_tokens(results: list[dict]) -> dict:
        """Match the efficiency report's successful-query denominator."""
        results = [r for r in results if r.get("generation_failed") is not True]
        usages = [
            MetricsCalculator.qa_token_usage(result.get("token_usage") or {})
            for result in results
        ]
        embedding = [int((r.get("token_usage") or {}).get("retrieval_embedding_tokens") or 0) for r in results]
        embedding_total = sum(embedding)
        missing = sum(not MetricsCalculator.embedding_usage_complete(r) for r in results)
        count = len(results)
        llm_input = sum(u[0] for u in usages) - embedding_total
        llm_output = sum(u[1] for u in usages)
        llm_total = sum(u[2] for u in usages) - embedding_total
        totals = {
            "Average Input Tokens": llm_input + embedding_total,
            "Average Output Tokens": llm_output,
            "Average LLM Tokens": llm_total,
            "Average Embedding Tokens": embedding_total,
            "Average Total Tokens": llm_total + embedding_total,
        }
        averages = {key: value / count if count else 0 for key, value in totals.items()}
        averages["Queries Missing Embedding Usage"] = missing
        averages["Queries Missing LLM Usage"] = sum(
            not MetricsCalculator.llm_usage_complete(r) for r in results
        )
        return averages

    @staticmethod
    def normalize_answer(s):
        """Normalize answer text: remove punctuation, convert to lowercase, remove articles"""
        s = str(s).replace(',', "") 
        def remove_articles(text): return re.sub(r'\b(a|an|the|and)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    @staticmethod
    def calculate_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = MetricsCalculator.normalize_answer(prediction).split()
        truth_tokens = MetricsCalculator.normalize_answer(ground_truth).split()
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def check_recall(retrieved_texts: List[str], evidence_list: List[str], soft_threshold: float = 0.8, min_soft_match_tokens: int = 4) -> float:
        """
        Calculate retrieval recall combining strict substring matching with dynamic token-based soft matching.
        
        Approach:
        - Combine and preprocess: concatenate multiple retrieved text chunks into a single string.
        - Strict matching first: check if evidence exists as a complete substring in the combined retrieved text.
        - Length blocking mechanism: calculate effective token count of evidence. If below threshold (e.g., short IDs or entities), directly determine no hit after strict match failure, prohibiting soft matching.
        - Soft matching fallback: for long text evidence, calculate token coverage in retrieved text, consider hit if threshold is met.
        - Equal weighting: each evidence has equal weight, final score is hit count / total count.
        
        Args:
            retrieved_texts: List[str], list of text chunks returned by retrieval module (required)
            evidence_list: List[str], ground truth evidence list containing IDs or long text evidence (required)
            soft_threshold: float, coverage threshold for soft matching to be considered a hit (optional, 0.0~1.0, default 0.8)
            min_soft_match_tokens: int, minimum effective token count threshold allowing fallback to soft matching (optional, default 4. Short texts below this length require strict matching)
            
        Returns:
            float, retrieval recall score, range 0.0 to 1.0
        """
        if not evidence_list: 
            return 0.0 
            
        combined_retrieved = " ".join(retrieved_texts)
        
        normalized_retrieved = MetricsCalculator.normalize_answer(combined_retrieved)
        ret_tokens = set(normalized_retrieved.split())
        
        hit_count = 0
        
        for evidence in evidence_list:
            if evidence in combined_retrieved:
                hit_count += 1
                continue
                
            normalized_ev = MetricsCalculator.normalize_answer(evidence)
            ev_tokens = set(normalized_ev.split())
            
            if not ev_tokens:
                continue
                
            if len(ev_tokens) < min_soft_match_tokens:
                continue
                
            overlap_count = len(ev_tokens & ret_tokens)
            coverage = overlap_count / len(ev_tokens)
            
            if coverage >= soft_threshold:
                hit_count += 1
                
        return hit_count / len(evidence_list)
