from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from webapp import SocialDistance

 
camera_id = {}
app = Flask(__name__)
@app.route("/" ) 
def main():
    
    return render_template("home.html")

def gen(camera): 
    while True:  
        frame = camera.main()
        if frame != "":
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    id  = 0
    return Response(gen(SocialDistance(id)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_response',methods = ['POST']) 
def get_response():
    json = request.get_json()
    print(json)
    if json is not None:
        h = json['status']
        print(h)  
    return print(camera_id)
    
if __name__ == '__main__':
    # Serve the app with gevent
    app.run(host='0.0.0.0', threaded=True, debug = True)