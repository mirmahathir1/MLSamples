import json
from disaster_model import calculate
from flask import Flask, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def add_message():
    content = request.json
    print(content['mytext'])
    prediction=calculate(content['mytext'])
    return json.dumps({"name":str(prediction)})
