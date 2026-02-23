from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print(response.choices[0].message.content)