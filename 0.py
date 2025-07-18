import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.2",  # Note: use this exact model name
    "prompt": "Generate A code for Binary Search in Cpp",
    "stream": False
}

response = requests.post(url, json=payload)
result = response.json()
print(result)  # See the full response

if "response" in result:
    print(result["response"])
else:
    print("Error or unexpected response:", result)
