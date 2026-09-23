"""Live acceptance checks: model alias, chat, two images, JSON response mode."""
import base64
import json
import os
import struct
import urllib.request
import zlib

BASE = 'http://127.0.0.1:23002/v1'
MODEL = 'qwen3-vl-4b-instruct-remote'
KEY = os.environ.get('QWEN_API_KEY', 'local-placeholder')
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, payload=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'},
    )
    with HTTP.open(req, timeout=180) as response:
        return json.load(response)


def png_data_url(rgb):
    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff))
    raw = (b'\x00' + bytes(rgb) * 128) * 128
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('!2I5B', 128, 128, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def main():
    models = request('/models')
    assert MODEL in {m['id'] for m in models['data']}, models
    print('PASS /v1/models: project model alias is available', flush=True)
    result = request('/chat/completions', {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': 'Reply with the single word OK.'}],
        'max_tokens': 16, 'temperature': 0,
    })
    assert result['choices'][0]['message']['content'].strip(), result
    print('PASS text chat:', result['choices'][0]['message']['content'], flush=True)
    result = request('/chat/completions', {
        'model': MODEL, 'temperature': 0, 'max_tokens': 128,
        'response_format': {'type': 'json_object'},
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Identify the solid color of each image in order. Return a JSON object with keys first and second, each containing a lowercase English color name.'},
            {'type': 'image_url', 'image_url': {'url': png_data_url((255, 0, 0))}},
            {'type': 'image_url', 'image_url': {'url': png_data_url((0, 0, 255))}},
        ]}],
    })
    content = json.loads(result['choices'][0]['message']['content'])
    assert content.get('first', '').lower() == 'red', content
    assert content.get('second', '').lower() == 'blue', content
    print('PASS two-image vision + JSON mode:', content, flush=True)


if __name__ == '__main__':
    main()
