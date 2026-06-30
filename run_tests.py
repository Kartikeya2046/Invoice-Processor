import time
import requests
import os
import json

TEST_FILES = [
    'boe_corrupted_sbcess.png',
    'fake.pdf',
    'garbage.png',
    'test_image.png',
    'test_mouser.pdf',
    'test_multi_real.pdf',
    'test_multi_real_data.pdf',
    'test_multi_synthetic.pdf',
    'test_pdf.pdf',
    'test_sample.png',
    'test_single.pdf'
]
url = 'http://127.0.0.1:8000/upload'

print("Starting tests for remaining files...")

for file in TEST_FILES:
    file_path = os.path.join('test_data', file)
    print(f"\n======================================")
    print(f"Testing {file}")
    print(f"======================================")
    start = time.time()
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(url, files={'file': f})
        end = time.time()
        elapsed = end - start
        
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Status Code: {response.status_code}")
        
        try:
            json_resp = response.json()
            print("Response JSON:")
            print(json.dumps(json_resp, indent=2))
        except Exception as e:
            print("Raw Response:", response.text)
            
    except Exception as e:
        print(f"Error testing {file}: {e}")
