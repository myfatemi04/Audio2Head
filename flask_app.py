import sys
# Add user directory so we can run this in sudo mode
sys.path = ['', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/michaelfatemi/.local/lib/python3.8/site-packages', '/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages']

import os
import elevenlabs
from inference import audio2head
import flask_cors
import random

with open("11.key") as f:
	os.environ['ELEVENLABS_API_KEY'] = f.read().strip()

import flask

app = flask.Flask(__name__)

flask_cors.CORS(app)

@app.route("/")
def index():
	return """
<script>
function anim() {
	var text = document.getElementById('text').value;
	document.getElementById('text').value = '';
	if (text.trim() == '') {
		return;
	}
	var video = document.getElementById('v');
	video.src = 'http://35.224.119.12/animate?text=' + encodeURIComponent(text);
	console.log("Animating...");
}
</script>

<input type="text" id="text">
<button onclick="anim()">Animate</button>

<video id=v controls></video>	
"""

@app.route("/synthesize", methods=['POST'])
def animate_head():
	text = flask.request.json['text']

	# generate driving audio
	audio = elevenlabs.generate(text, api_key=os.environ['ELEVENLABS_API_KEY'], voice='AUfmNX7a1AJ7HNn4TKft')
	
	request_id = "request_" + str(random.randint(0, 1000000000))

	# save to mp3
	with open(f'./requests/{request_id}.mp3', 'wb') as f:
		f.write(audio)

	path = audio2head(request_id, 'pavel.png', './checkpoints/audio2head.pth.tar')
	# path = audio2head('audio.mp3', 'pavel.png', './checkpoints/audio2head.pth.tar', './results')
	print(path)
	# write mp4 response
	return flask.send_file(path, mimetype='video/mp4')

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, port=80)
