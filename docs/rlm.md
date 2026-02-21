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

    For debug, use settings `persistent=True` and `verbose=True`
    
    * settings

        ```python
        rlm = RLM(
            ...,
            environment="local",
            persistent=True,
            verbose=True,
            )
        ```

    * example output:

        ```sh
        ...

        ╭─ ★ Final Answer
        │
        │ <response you will see>
        │
        ╰───────────────

        Iterations       4
        Total Time       264.49s
        Input Tokens     20,580
        Output Tokens    2,393

        ```