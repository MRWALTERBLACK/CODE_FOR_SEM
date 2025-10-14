#    python -m pip install openai
#    pip3 install openai
from openai import OpenAI
client = OpenAI(api_key="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")

question = "Your Question"

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": question}
    ]
)
print(response.choices[0].message.content)
