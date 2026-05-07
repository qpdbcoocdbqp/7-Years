import json
import re
import time
import pyarrow as pa
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .evaluator import StructuredEvaluator, judge
from .openai_client import OpenAIClient, create_openai_client
from .data_loader import load_benchmark_data, FLAT_TRANSFORMS


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

        judgement = []
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
                try:
                    if response.choices[0].message.parsed:
                        fit = response.choices[0].message.parsed.model_dump()
                    else:
                        content = response.choices[0].message.content
                        if not content:
                            raise ValueError("Empty response content from model")
                        
                        # Try direct JSON parse
                        try:
                            fit = json.loads(content)
                        except json.JSONDecodeError:
                            # Try to extract JSON from text (e.g. markdown blocks)
                            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                            if json_match:
                                fit = json.loads(json_match.group(1))
                            else:
                                # Try finding the first '{' and last '}'
                                start = content.find('{')
                                end = content.rfind('}')
                                if start != -1 and end != -1:
                                    fit = json.loads(content[start:end+1])
                                else:
                                    raise
                    
                    judgement.append(judge(
                        true_dict=ground_truth,
                        fit_dict=fit,
                        flat_transform=FLAT_TRANSFORMS.get(self.benchmark_name))
                        )
                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    print(f"Error processing response: {e}")
                    if not response.choices[0].message.parsed:
                        print(f"Raw content: {response.choices[0].message.content}")
                    error_logs.append({"error": str(e), "response": str(response)})
            else:
                error_logs.append(response)
            del response, status

        if len(judgement) == 0:
            overall_accuracy = 0.0
            field_stats = []
        else:
            overall_accuracy = sum(list(map(lambda x: all(x.column("is_correct")), judgement))) / len(judgement)
            judgement = pa.concat_tables(judgement)
            field_stats = judgement.group_by(["key"]).aggregate([("is_correct", "mean")]).rename_columns(["field", "accuracy"]).to_pylist()

        final_results = {
            "benchmark_name": self.benchmark_name,
            "model": model or self.openai_client.model,
            "sample_size": len(tasks),
            "success_number": len(tasks) - len(error_logs),
            "overall_accuracy": overall_accuracy,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": field_stats,
        }
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        return final_results

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
