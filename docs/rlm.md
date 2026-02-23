# Recursive Language Models (RLMs)

* [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## Setup

* use `uv` to run an example
    ```sh
    # install rlms
    uv pip install 'git+https://github.com/alexzhang13/rlm.git@v0.1.1a'
    ```

* **IMPORTANT**: LLM chat template should allow duplicated `user` role order in input messages.

    The function `RLM.completion` will create `current_prompt` like below format to be input of LLM:

    ```python
    [
        {'role': 'system', 'context': '<system message>'},
        {'role': 'user', 'context': '<previous message>'},
        {'role': 'user', 'context': '<current message>'},
    ]
    ```

    * For example, use [`mistralai/Ministral-3-3B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) model. Here modify chat template into [chat_template_rlm.jinja](../examples/chat_template_rlm.jinja) to satisfy the input format.

## Example

* [app/feature_rlm](../app/feature_rlm.py)

    For debug, use settings `verbose=True`

    * Default use `environment="local"`
      * settings

          ```python
          rlm = RLM(
              ...,
              environment="local",
              verbose=True,
              )
          ```

    * Use `environment="docker"`

        ```sh
        docker pull python:3.11-slim
        ```

        ```python
        rlm = RLM(
            ...,
            environment="docker",
            verbose=True,
            )
        ```

    * example output will like

        ```sh
        ...

        ╭─ ★ Final Answer
        │
        │ <response you will see>
        │
        ╰───────────────

        Iterations       <Iterations>
        Total Time       <Total Time>
        Input Tokens     <Input Tokens>
        Output Tokens    <Output Tokens>

        ```

    * RLM logger visualizer
      
      The visualization UI for log files that `RLMLogger` written.

        ```sh
        git clone https://github.com/alexzhang13/rlm.git
        cd rlm/visualizer/
        pnpm install
        pnpm run dev --hostname localhost --port 19001

        # http://localhost:19001
        ```
