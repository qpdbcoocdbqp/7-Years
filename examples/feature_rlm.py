import os
import dotenv
from rlm import RLM
from rlm.logger import RLMLogger


dotenv.load_dotenv()
logger = RLMLogger()
rlm = RLM(
    backend="openai",
    backend_kwargs={  
        "base_url": os.getenv("LOCAL_OPENAI_BASE_URL"),
        "api_key": os.getenv("LOCAL_OPENAI_API_KEY"),
        "model_name": os.getenv("LOCAL_OPENAI_MODEL"),
        "temperature": 0.05,
        "repetition_penalty": 1.7,
        "max_tokens": 2048,
    },  
    environment="docker",
    logger=logger,
    verbose=True,
    )

prompt = "Compute the sum 1 + 2 + 3 + 4 + 5 and print the result using a REPL block, then return it with FINAL_VAR."
response = rlm.completion(prompt)

print("Response:", response.response)
if response.metadata:
    traj = response.metadata
    print("Trajectory: run_metadata +", len(traj.get("iterations", [])), "iterations")
    print("Trajectory:", traj)
    if traj.get("iterations"):
        first = traj["iterations"][0]
        print("  First iteration keys:", list(first.keys()))
else:
    print("Trajectory: (none — no logger or metadata not captured)")

print("Full response:", response)

rlm.close()
