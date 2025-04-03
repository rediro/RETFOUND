import requests
import json

# Flask API URL
url = 'http://localhost:5000/generate'

while True:
    user_input_instruction = input("Enter your prompt: ")

    if user_input_instruction.lower() == "exit":
        print("Exiting chatbot.")
        break

    # Create JSON payload
    data = {'instruction': user_input_instruction}
    headers = {'Content-Type': 'application/json'}

    # Send request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Print response
    if response.status_code == 200:
        print("Bot:", response.json()['generated_text'])
    else:
        print('Error:', response.json()['error'])
