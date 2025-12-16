import os
from rich.console import Console
from dotenv import load_dotenv
from common.schema import get_schema
from common.runner import run_benchmark


load_dotenv()

console = Console()
benchmarks = ["data_table_analysis", "financial_entities", "insurance_claims", "pii_extraction"]

sample_size = int(os.getenv("SAMPLE_SIZE", -1))
use_local_api = True if os.getenv("LOCAL_OPENAI_BASE_URL") else False
model = os.getenv("LOCAL_OPENAI_MODEL") if use_local_api else os.getenv("OPENAI_MODEL")
temperature = float(os.getenv("TEMPERATURE"))
max_workers = int(os.getenv("MAX_WORKERS"))

for benchmark_name in benchmarks:
    schema = get_schema(benchmark_name)
    results = run_benchmark(
        benchmark_name=benchmark_name,
        schema=schema,
        sample_size=sample_size,
        use_local_api=use_local_api,
        model=model,
        temperature=temperature,
        max_workers=max_workers
        )
    for key in ["benchmark_name", "model", "sample_size", "success_number", "timestamp", "statistics"]:
        console.print(f"[yellow]{key}[/yellow]", results.get(key))
