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