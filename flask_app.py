import sys
# Add user directory so we can run this in sudo mode
sys.path = ['', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/michaelfatemi/.local/lib/python3.8/site-packages', '/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages']

import os
import elevenlabs
from inference import audio2head

import flask

app = flask.Flask(__name__)

@app.route("/")
def index():
	return "Hello world!"

@app.route("/animate", methods=['GET'])
def animate_head():
	text = flask.request.args['text']
	# generate driving audio
	audio = elevenlabs.generate(text, api_key=os.environ['ELEVENLABS_API_KEY'], voice='ovOwiv8AZiT34IOXyVTO')
	# # return as mp3
	# return flask.Response(audio, mimetype='audio/mpeg')
	# return audio
	# save to mp3
	with open('audio.mp3', 'wb') as f:
		f.write(audio)
	path = audio2head('audio.mp3', 'pavel.png', './checkpoints/audio2head.pth.tar', './results')
	print(path)
	# write mp4 response
	return flask.send_file(path, mimetype='video/mp4')

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, port=80)
