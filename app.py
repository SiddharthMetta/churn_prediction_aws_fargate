import flask
from flask import Flask, request
import pandas as pd
from app_utils import predict
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def main():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            req = request.json
            df_input = pd.json_normalize(req)
            return predict(df_input)
    except Exception as e:
        print(f"Failed to load the request {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
