import json
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .evaluator import StructuredEvaluator
from .openai_client import OpenAIClient, create_openai_client
from .data_loader import load_benchmark_data


class BenchmarkRunner:
    def __init__(self, 
                 benchmark_name: str,
                 schema: Dict[str, Any],
                 openai_client: Optional[OpenAIClient] = None,
                 use_local_api: bool = True):
        self.benchmark_name = benchmark_name
        self.schema = schema
        self.evaluator = StructuredEvaluator()
        
        if openai_client is None:
            self.openai_client = create_openai_client(local=use_local_api)
        else:
            self.openai_client = openai_client
    
    def run_evaluation(self,
                      sample_size: Optional[int] = None,
                      system_prompt: Optional[str] = None,
                      model: Optional[str] = None,
                      temperature: float = 0.0,
                      max_workers: int = 5
                      ) -> Dict[str, Any]:
        print(f"Start {self.benchmark_name} benchmark")
        
        print("load dataset")
        tasks, ground_truths, raw_data, system_prompt  = load_benchmark_data(
            self.benchmark_name, 
            sample_size=sample_size,
            system_prompt=system_prompt
        )
        
        print(f"load {len(tasks)} samples")
        responses = []
        targets = []
        error_logs = []
        for task, ground_truth in tqdm(zip(tasks, ground_truths), desc="send task"):
            response, status = self.openai_client.get_structured_response(
                task=task,
                system_prompt=system_prompt,
                response_format=self.schema,
                model=model,
                temperature=temperature
            )
            if status:
                if response.choices[0].message.parsed:
                    responses.append(response.choices[0].message.parsed.model_dump())
                else:
                    responses.append(json.loads(response.choices[0].message.content))
                targets.append(ground_truth)
            else:
                error_logs.append(response)
            del response, status
        results = self._evaluate_accuracy(responses, targets)
        stats = self._calculate_statistics(results, responses)
        final_results = {
            "benchmark_name": self.benchmark_name,
            "model": model or self.openai_client.model,
            "sample_size": len(tasks),
            "success_number": len(responses),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": stats,
            "detailed_results": results
        }
        print(f"Overall accuracy: {stats['overall_accuracy']:.2%}")
        return final_results
    
    def _evaluate_accuracy(self, 
                         responses: List[Optional[Dict[str, Any]]], 
                         targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_pairs = [
            (pred, true) for pred, true in zip(responses, targets)
            if pred is not None
        ]
        if not valid_pairs:
            return {
                "overall_correctness": [],
                "field_breakdowns": [],
                "failed_responses": len(responses)
            }
        valid_predictions, valid_targets = zip(*valid_pairs)
        overall_correctness, field_breakdowns = self.evaluator.evaluate_response_accuracy_with_breakdown(
            list(valid_predictions), list(valid_targets)
        )
        return {
            "overall_correctness": overall_correctness.tolist(),
            "field_breakdowns": field_breakdowns,
            "failed_responses": len(responses) - len(valid_pairs)
        }
    
    def _calculate_statistics(self, 
                            results: Dict[str, Any],
                            responses: List[Optional[Dict[str, Any]]],
                            ) -> Dict[str, Any]:
        overall_correctness = results["overall_correctness"]
        field_breakdowns = results["field_breakdowns"]
        # failed_responses = results["failed_responses"]
        # total_samples = len(responses)
        # valid_samples = len(overall_correctness)
        overall_accuracy = sum(overall_correctness) / len(overall_correctness) if overall_correctness else 0.0
        # success_rate = valid_samples / total_samples if total_samples > 0 else 0.0
        field_stats = {}
        if field_breakdowns:
            all_fields = set()
            for breakdown in field_breakdowns:
                all_fields.update(breakdown.keys())
            
            for field in all_fields:
                field_correct = sum(1 for breakdown in field_breakdowns if breakdown.get(field, False))
                field_stats[field] = field_correct / len(field_breakdowns)
        return {
            # "total_samples": total_samples,
            # "valid_samples": valid_samples,
            # "failed_samples": failed_responses,
            # "success_rate": success_rate,
            "overall_accuracy": overall_accuracy,
            "field_accuracies": field_stats
        }

def run_benchmark(benchmark_name: str,
                 schema: Any,
                 sample_size: Optional[int] = None,
                 use_local_api: bool = False,
                 **kwargs) -> Dict[str, Any]:
    runner = BenchmarkRunner(
        benchmark_name=benchmark_name,
        schema=schema,
        use_local_api=use_local_api
    )
    
    return runner.run_evaluation(sample_size=sample_size, **kwargs)
