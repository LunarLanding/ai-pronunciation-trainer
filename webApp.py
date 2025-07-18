import json
import os
import webbrowser

from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS

import lambdaGetSample
import lambdaSpeechToScore
import lambdaTTS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/getAudioFromText', methods=['POST'])
def getAudioFromText():
    event = {'body': json.dumps(request.get_json(force=True))}
    return lambdaTTS.lambda_handler(event, [])


@app.route('/getSample', methods=['POST'])
def getNext():
    event = {'body':  json.dumps(request.get_json(force=True))}
    return lambdaGetSample.lambda_handler(event, [])


@app.route('/GetAccuracyFromRecordedAudio', methods=['POST'])
def GetAccuracyFromRecordedAudio():

    try:
        event = {'body': json.dumps(request.get_json(force=True))}
        lambda_correct_output = lambdaSpeechToScore.lambda_handler(event, [])
    except Exception as e:
        import traceback
        print(e)
        print(traceback.format_exc())

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }

    return lambda_correct_output


@app.route("/<filename>")
def others(filename):
    return send_from_directory('./static', filename)

if __name__ == "__main__":
    language = 'de'
    #print(os.system('pwd'))
    #webbrowser.open_new('http://127.0.0.1:3000/')
    app.config.update(
        TEMPLATES_AUTO_RELOAD=True
    )
    app.run()
