import os
import random
import openai
import json
from rich.console import Console
from dotenv import load_dotenv
from common.schema import get_schema
from common.data_loader import load_benchmark_data


load_dotenv()

console = Console()
# benchmarks = ["data_table_analysis", "financial_entities", "insurance_claims", "pii_extraction"]
benchmark_name = "data_table_analysis"
use_local_api = True if os.getenv("LOCAL_OPENAI_BASE_URL") else False
model = os.getenv("LOCAL_OPENAI_MODEL") if use_local_api else os.getenv("OPENAI_MODEL")
temperature = float(os.getenv("TEMPERATURE"))
max_workers = int(os.getenv("MAX_WORKERS"))

# prepare dataset
schema = get_schema(benchmark_name)
tasks, ground_truths, raw_data, system_prompt = load_benchmark_data(
    benchmark_name,
    sample_size=-1,
    system_prompt=None
)

# agent setup
agent = openai.Client(base_url="http://localhost:8642/v1", api_key="your-secret-key")
print(agent.models.list())

# SOUL.md
soul = """
# Guideline\n\n{system_prompt}\n
# Output Schema\n\n{schema}
""".format(
    system_prompt=system_prompt,
    schema=schema if isinstance(schema, str) or isinstance(schema, dict) else schema.model_json_schema()
    )

with open(".hermes/SOUL.md", "w") as f:
    f.write(soul)

# send task to agent
i = min(random.randint(0, 100), 99)
message = {"role": "user", "content": f"""# Content\n\n{tasks[i]}\n\n---\nConvert to output JSON format"""}
response = agent.chat.completions.create(
    messages=[
        {"role": "user", "content": message}
    ],
    model=model,
    temperature=0.7,
    max_tokens=2048
    )
raw_agent_response = response.choices[0].message.content
print(raw_agent_response)

# parse agent response to sturctured output
parser_client = openai.Client(base_url="http://localhost:9006/v1", api_key="***")
final_response = parser_client.chat.completions.parse(
    model="alm",
    messages=[
        {"role": "system", "content": "Extract the data from the input and return the JSON format."},
        {"role": "user", "content": raw_agent_response}
        ],
    response_format=schema,
    max_tokens=2048,
    temperature=0.0,
    )

if final_response.choices[0].message.parsed is None:
    result = json.loads(final_response.choices[0].message.content)
else:
    result = final_response.choices[0].message.parsed

console.print("Task Case:\n{tasks[i]}\n---\nGround Truth:\n{ground_truths[i]}\n---\nOutput:\n{result}\n---")
