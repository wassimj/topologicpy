#!/usr/bin/env python3

import requests

def read_file_to_string(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()
    return file_content

def upload_files():
    file_contents = {}
    file_contents['torus.mtl'] = read_file_to_string('torus.mtl')
    file_contents['torus.obj'] = read_file_to_string('torus.obj')

    try:
        response = requests.post('http://127.0.0.1:8000/upload/123', files=file_contents)
        if response.status_code != 200:
            print(f'Failed to upload file(s): {response.status_code} {response.reason}')
        else:
            print(f'File(s) uploaded successfully. View at http://127.0.0.1:8000/#channel=123')
    except requests.exceptions.RequestException as e:
        print(f'Error uploading file(s): {e}')

if __name__ == '__main__':
    upload_files()
