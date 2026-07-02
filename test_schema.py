import requests, json
from models import ExtractedInvoiceData

schema = ExtractedInvoiceData.model_json_schema()

res = requests.post('http://localhost:11434/api/chat', json={
    'model': 'qwen2.5:3b',
    'messages': [{'role': 'user', 'content': 'test'}],
    'format': schema
})
print("STATUS CODE:", res.status_code)
if res.status_code != 200:
    print(res.text)
