# sambanova cloud rate-limited factory

Global Rate limiter implementation over a SambaNova Cloud OpenAI like Chat interface

Below you find a code example how you can use the ratelimited calls simply.

```python
# Example code
import os
from sambanova_factory.LLMHelper import LLMHelper

sambanova_api_key = os.environ['SAMBANOVA_API_KEY']

LLMHelper(sambanova_api_key=sambanova_api_key)

call_result = LLMHelper.call_llama31_70b(
    system_prompt="""You are a helpful assistant!""",
    user_prompt="My name is {name}",
    variables={
        "name": "Mr. Smith"
    }
)

print(call_result)

```