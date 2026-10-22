import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["ZAI_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

MODEL = os.environ.get("LLM_MODEL", "glm-4.7-flash")

response = client.chat.completions.create(
    model=MODEL,
    max_tokens=256,  # 关闭思考后，256足够输出一句话
    messages=[
        {"role": "user", "content": "What is a neural network in one sentence?"}
    ],
    extra_body={"thinking": {"type": "disabled"}},  # 关闭思考
)

msg = response.choices[0].message

print("content =", repr(msg.content))
print("finish_reason =", response.choices[0].finish_reason)
print("reasoning_content =", repr(getattr(msg, "reasoning_content", None)))
print("--- full response ---")
print(response)

print("---")
print("input tokens:", response.usage.prompt_tokens)
print("output tokens:", response.usage.completion_tokens)