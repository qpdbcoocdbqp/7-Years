import os
import random
from rich.console import Console
from dotenv import load_dotenv
from common.schema import get_schema
from common.runner import BenchmarkRunner, run_benchmark
from common.data_loader import load_benchmark_data


load_dotenv()

console = Console()
benchmark_name = "data_table_analysis"
use_local_api = True if os.getenv("LOCAL_OPENAI_BASE_URL") else False
model = os.getenv("LOCAL_OPENAI_MODEL") if use_local_api else os.getenv("OPENAI_MODEL")
temperature = float(os.getenv("TEMPERATURE"))
max_workers = int(os.getenv("MAX_WORKERS"))

schema = get_schema(benchmark_name)
tasks, ground_truths, raw_data, system_prompt  = load_benchmark_data(
    benchmark_name,
    sample_size=-1,
    system_prompt=None
)

# create a skill to handle json format request
runner = BenchmarkRunner(
    benchmark_name=benchmark_name,
    schema=schema,
    use_local_api=use_local_api
)
runner.openai_client.client.models.list()
agent = runner.openai_client.client

i = min(random.randint(0, 100), 99)
message = """
create a skill, use make clean JSON. User will provide a guideline, output format, and input data.
# guideline: {system_prompt}

# output schema: {schema}

# input data: {task}

# Ground Truth: {ground_truth}

""".format(system_prompt=system_prompt, task=tasks[i], schema=schema, ground_truth=ground_truths[i])

print(message)
response = agent.chat.completions.create(
    messages=[
        {"role": "user", "content": message}
    ],
    model=model,
    temperature=0.7,
    max_tokens=2048
    )

for _ in range(5):
    i = min(random.randint(0, 100), 99)
    message = """
optimize the csv_to_structured_json skill. User will provide a guideline, output format, and input data.
# guideline: {system_prompt}

# output schema: {schema}

# input data: {task}

# Ground Truth: {ground_truth}

""".format(system_prompt=system_prompt, task=tasks[i], schema=schema, ground_truth=ground_truths[i])
    response = agent.chat.completions.create(
        messages=[
            {"role": "user", "content": message}
        ],
        model=model,
        temperature=0.7,
        max_tokens=2048
        )

print(response.choices[0].message.content)

# run the benchmark
results = run_benchmark(
    benchmark_name=benchmark_name,
    schema=schema,
    sample_size=1,
    use_local_api=use_local_api,
    model=model,
    temperature=temperature,
    max_workers=max_workers
    )

for key in ["benchmark_name", "model", "sample_size", "success_number", "overall_accuracy", "timestamp", "statistics"]:
    console.print(f"[yellow]{key}[/yellow]", results.get(key))
