import socketio
import os
from aiohttp import web
import sys

# if True, compile the most recent build version of the UI
# requires yarn package manager to be installed on the machine


UPDATE_UI = len(sys.argv) > 1
if UPDATE_UI:
    if 'build' in os.listdir():
        os.system('rm -r build')

    os.system('cd ../frontend && yarn build && mv build ../Backend')


static_files = {
    "/": "./build/index.html",
    "/static": "./build/static"
}

sio = socketio.AsyncServer()
# app = socketio.ASGIApp(sio, static_files=static_files)
app = web.Application()
sio.attach(app)

async def index(request):
    with open('./build/index.html') as file:
        return web.Response(text=file.read(), content_type='text/html')

# NAMESPACES
class GeneralNamespace(socketio.AsyncNamespace):
    async def on_connect(self, sid, environ):
        # perform user authentication
        # self.emit('send-message', 'fuck you')
        # print('this is environ', environ)
        print('user connected ', sid)
        # print('this is self', self)

    async def on_disconnect(self, sid):
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
    async def on_get_data(self, sid):
        # print(sid)
        # print(mockup_data)
        await self.emit('rec_data', data=mockup_data)

    async def on_message(self, sid, message):
        print('message from', sid)
        print(message)

sio.register_namespace(Network('/network_data'))

class Dataset(GeneralNamespace):
    def on_event():
        pass

sio.register_namespace(Dataset('/dataset_data'))

class TrainingMetrics(GeneralNamespace):
    def on_event():
        pass

sio.register_namespace(TrainingMetrics('/training-metrics_data'))

class ResultMetrics(GeneralNamespace):
    def on_event():
        pass

sio.register_namespace(ResultMetrics('/result-metrics_data'))



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


app.router.add_get('/', index)
app.router.add_static('/static', './build/static')

if __name__ == '__main__':
    web.run_app(app, host='localhost', port=6969)