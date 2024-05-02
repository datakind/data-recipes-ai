from flask import Flask, Response
import yaml

app = Flask(__name__)

@app.route('/openapi.yaml')
def serve_openapi():
    with open('openapi.yaml', 'r') as file:
        yaml_content = file.read()
    return Response(yaml_content, mimetype='application/yaml')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
