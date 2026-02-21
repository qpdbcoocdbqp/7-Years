import os
import dotenv
from rlm import RLM


dotenv.load_dotenv()

rlm = RLM(
    backend="openai",
    backend_kwargs={  
        "base_url": os.getenv("LOCAL_OPENAI_BASE_URL"),
        "api_key": os.getenv("LOCAL_OPENAI_API_KEY"),
        "model_name": os.getenv("LOCAL_OPENAI_MODEL"),
    },  
    environment="local",
    persistent=True,
    verbose=True,
    )

prompt = "Print me the first 32 powers of two, each on a newline."
response = rlm.completion(prompt)

print(rlm._persistent_env.locals.keys())

rlm.close()
