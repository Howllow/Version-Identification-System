from flask import *
from gevent import pywsgi
import pygame
from processing import *
from add_song import *
from similar import *
import json
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'VIS'


@app.route('/')
def homepage():
    return redirect(url_for('index'))


@app.route('/index', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')


@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'GET':
        return render_template('query.html')

    elif request.method == 'POST':
        music = request.files['file_data']
        print(music.filename)
        print(music.name)
        path = './tmp.' + music.filename.split('.')[1]
        music.save(path)
        message = dict()
        cqt_stat, cqt = get_cqt(path)
        if cqt_stat == 1:
            message['stat'] = 1
            return json.dumps(message)
        message['music_list'] = get_similar(cqt, 0)
        message['stat'] = 0
        return json.dumps(message)


@app.route('/add', methods=['POST', 'GET'])
def addsong():
    if request.method == 'GET':
        return render_template('add.html')

    elif request.method == 'POST':
        message = dict()
        musics = request.files.getlist('file_data')
        message['error_list'] = add_songs(musics)
        return json.dumps(message)


@app.route('/judge', methods=['POST', 'GET'])
def judge_onaji():
    if request.method == 'GET':
        return render_template('judge.html')

    elif request.method == 'POST':
        musics = request.files.getlist('file_data')
        message = dict()
        message['rate'] = cal_similar(musics)
        print(message['rate'])
        return json.dumps(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', threaded=False)