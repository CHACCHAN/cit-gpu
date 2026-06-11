import urllib.request
import tarfile
import io

url = 'https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

print('Fetching archive stream from GitHub...')
with urllib.request.urlopen(req) as response:
    compressed_data = response.read()
    
print('Extracting package archive in memory...')
with tarfile.open(fileobj=io.BytesIO(compressed_data), mode='r:gz') as tar:
    tar.extractall(path='.')
    
print('Ollama extracted successfully into ./bin/ollama')