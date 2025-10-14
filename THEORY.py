#    python -m pip install openai
#    pip3 install openai
#    _oFx7dzu3DSsvWVq7LXYwzBk0SRk2ATqa2TSzOu3XQjj4A
from openai import OpenAI

# Initialize client with your API key
client = OpenAI(api_key="sk-proj-mv26O99ElRdEt2i9Osus5bR2Plr-lTGN-ASi2izFGp3aX5GU9VgaKlez9GpsjiYCIsdJZu_W3fT3BlbkFJDNwKmCi9Q2egnvsJStfyYI0tQK3Q")

question = "what is chi test? in hypothesis testing in just 2 lines"

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are an expert statistics tutor."},
        {"role": "user", "content": question}
    ]
)

# Correct way to access the content
print(response.choices[0].message.content)


