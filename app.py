#!/usr/bin/env python
from threading import Lock
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
from config import loader
import tweepy
from flask_pymongo import PyMongo
import json
import datetime
import random
from bson.objectid import ObjectId


# Override tweepy.StreamListener to add logic to on_status
class StreamProcessor(tweepy.StreamListener):
    def __init__(self, tweepy, socketio):
        super().__init__()
        self._tweetCount = 0
        self._tweepy = tweepy,
        self._socketio = socketio

    def on_status(self, status):
        data = status._json
        # themes = ['fund', 'edu', 'awareness']
        data['sentiment'] = random.uniform(-1, 1)
        self._socketio.emit('hello', {'number': str(self._tweetCount), 'tweet': json.dumps(data)}, namespace='/test')
        # u = mongo.db.users.find_one({"screen_name": status._json['user']['screen_name']})
        # if u is None:
        #     mongo.db.users.insert_one(status._json['user'])
        # mongo.db.tweets.insert_one(status._json)
        self._tweetCount = self._tweetCount + 1
        print(self._tweetCount)

class JSONEncoder(json.JSONEncoder):
    ''' extend json-encoder class'''

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime.datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)


# FLASK, Config and DB
app = Flask(__name__)
app = loader.load_config(app)
app.json_encoder = JSONEncoder
mongo = PyMongo(app)

# SOCKET IO
# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = "threading"
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

# Tweepy
auth = tweepy.OAuthHandler(app.config['consumer_key'], app.config['consumer_secret'])
auth.set_access_token(app.config['access_token'], app.config['access_token_secret'])
api = tweepy.API(auth)


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/test')


@app.route('/')
def index():
    return render_template('index2.html', async_mode=socketio.async_mode)

@app.route('/hi')
def hi():
    return render_template('sample.html', async_mode=socketio.async_mode)


@app.route('/hello')
def hello():
    socketio.emit('hello', {'data': 'hello_doobs_doobs'}, namespace='/test')
    print('DONE! SENT HELLO!')
    return "DONE!"


@socketio.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})


@socketio.on('my_broadcast_event', namespace='/test')
def test_broadcast_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         broadcast=True)


@socketio.on('join', namespace='/test')
def join(message):
    join_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.on('leave', namespace='/test')
def leave(message):
    leave_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'coufrom app import *nt': session['receive_count']})


@socketio.on('close_room', namespace='/test')
def close(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response', {'data': 'Room ' + message['room'] + ' is closing.',
                         'count': session['receive_count']},
         room=message['room'])
    close_room(message['room'])


@socketio.on('my_room_event', namespace='/test')
def send_room_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         room=message['room'])


@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']})
    disconnect()


@socketio.on('my_ping', namespace='/test')
def ping_pong():
    emit('my_pong')


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected', request.sid)


stream = tweepy.Stream(auth=api.auth, listener=StreamProcessor(api, socketio))
stream.filter(track=app.config['keywords'], languages=["en"], is_async=True)
socketio.run(app, debug=False)
