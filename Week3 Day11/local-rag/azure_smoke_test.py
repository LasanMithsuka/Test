import os
from openai import AzureOpenAI

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # you must set this
if not deployment:
    raise SystemExit("Set AZURE_OPENAI_DEPLOYMENT first (your Azure deployment name).")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version=api_version,
)

resp = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Say only: OK"}],
    max_tokens=5,
)

print(resp.choices[0].message.content)