import urllib.request
import json

try:
    req = urllib.request.Request('http://127.0.0.1:8000/api/documents/')
    res = urllib.request.urlopen(req)
    data = json.loads(res.read())
    
    docs = data.get('documents', [])
    target = next((d for d in docs if 'Batch' in d['filename']), None)
    
    if target:
        delete_url = f'http://127.0.0.1:8000/api/documents/{target["id"]}'
        print(f"Deleting doc ID: {target['id']}")
        req = urllib.request.Request(delete_url, method='DELETE')
        urllib.request.urlopen(req)
        print(f"Successfully deleted {target['filename']}")
    else:
        print("Document not found!")
except Exception as e:
    print(f"Error: {e}")
