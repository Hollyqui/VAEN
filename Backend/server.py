import uvicorn
import socketio
import os

# if True, compile the most recent build version of the UI
# requires yarn package manager to be installed on the machine
DEV_MODE = True
if DEV_MODE:
    if 'build' in os.listdir():
        os.system('rm -r build')

    os.system('cd ../frontend && yarn build && mv build ../Backend')


static_files = {
    "/": "./build/index.html",
    "/static": "./build/static"
}

sio = socketio.AsyncServer()
app = socketio.ASGIApp(sio, static_files=static_files)

# NAMESPACES
class GeneralNamespace(socketio.AsyncNamespace):
    async def on_connect(self, sid, environ):
        # perform user authentication
        

        print('user connected ', sid)

    async def on_disconnect(self, sid, environ):
        # 
        print('user disconnected ', sid)

    
class Home(GeneralNamespace):
    def on_event():
        pass

mockup_data = [
    {
        "name": "hey",
        "trainable": "true",
        "dtype": "real",
        "id": 0,
        "avg_weight": "G",
        "avg_abs_weight": "string"
    },
    {
        "name": "hey",
        "trainable": "true",
        "dtype": "real",
        "id": 1,
        "avg_weight": "G",
        "avg_abs_weight": "string"
    }
]

class Network(GeneralNamespace):
    def on_get_data(self, sid):
        self.emit('rec_data', data=mockup_data, to=sid, namespace='/dataset_server')

sio.register_namespace(Network('/network_data'))

class Dataset(GeneralNamespace):
    def on_event():
        pass

class TrainingMetrics(GeneralNamespace):
    def on_event():
        pass

class ResultMetrics(GeneralNamespace):
    def on_event():
        pass




# @sio.event
# async def connect(sid, environ):
#     # perform user authentication
#     # environ 'standard WSGI format' containing request information, incl HTTP headers
#     # return False to reject the connection
#     # if data is to be passed to rejected client, socketio.exceptions.ConnectionRefusedError
#     #   can be raised (raise) and all of it's arguments will be sent to the clients with the 
#     #   rejection message
#     print('connect', sid)

#     username = autenthicate_user(environ)
#     await sio.save_session(sid, {'user': username})

# @sio.event
# async def disconnect(sid, environ):
#     print('disconnect', sid)

# @sio.event
# async def message(sid, data):
#     session = await sio.get_session(sid)
#     print('message from ', session['user'])

# @sio.on('custom event')
# async def custom_event(sid, data):
#     pass

# def autenthicate_user(user):
#     pass


uvicorn.run(app, host='localhost', port=6969)