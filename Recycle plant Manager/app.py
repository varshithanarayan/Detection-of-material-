from flask import *
import time
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from time import sleep
import random
import os
from functools import wraps
import webbrowser
import ctypes
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
from time import sleep
from flask_mysqldb import MySQL
from tqdm import tqdm
import socket
import controller as ct
import cv2
def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN
app=Flask(__name__, template_folder='templates', static_folder='static')
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='root'
app.config['MYSQL_DB']='garbage'
app.config['MYSQL_CURSORCLASS']='DictCursor'
mysql=MySQL(app)

@app.route('/login',methods=['POST','GET'])
def login():
    status=True
    if request.method=='POST':
        uname=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("select * from admin where email=%s and password=%s",(uname,ct.md5(pwd)))
        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["username"]
            flash('Login Successfully','success')
            return redirect('home')
        else:
            flash('Invalid Login credentials. Try Again','danger')
    return render_template("login.html",url = url)


@app.route('/')
def index():
    return render_template('login.html')

def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap



@app.route('/get_dataset', methods=['GET', 'POST'])
@is_logged_in
def get_dataset():
    if (os.listdir('../Dataset')):
        count = 0
        for root_dir, cur_dir, files in os.walk(r'../Dataset'):
            count += len(files)
        time.sleep(3)
        return str(count) + " images found"
    else:
        return "No dataset Found in the path specified. Copy the files to path and refresh and try again"
@app.route('/start_training', methods=['GET', 'POST'])
@is_logged_in
def start_training():
    ct.train()
    return "Training Completed"

@app.route('/save_model', methods=['GET', 'POST'])
@is_logged_in
def save_model():
    if(ct.save_model()):
        return "Model Saved Successfully"
    else:
        return "Failed to save model"
@app.route('/save_memo', methods=['GET', 'POST'])
@is_logged_in
def save_memo():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[1, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route('/show_accuracy', methods=['GET', 'POST'])
@is_logged_in
def show_accuracy():
    time.sleep(2)
    return send_file('../Plots/accuracy.png', mimetype='image/jpg')

@app.route('/show_loss', methods=['GET', 'POST'])
@is_logged_in
def show_loss():
    time.sleep(2)
    return send_file('../Plots/loss.png', mimetype='image/png')
@app.route('/predict', methods=['GET', 'POST'])
@is_logged_in
def predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static','input_images',secure_filename(f.filename))
        f.save(file_path)
        output_file = f.filename
    return render_template('demo.html',url=url,filename = file_path)
global output_file


#Home page
@app.route("/home",methods=['POST','GET'])
@is_logged_in
def home():
    global url
    return render_template('train.html',data = session['username'],url = url)

@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == '__main__':
    global url
    app.secret_key='secret123'
    myIP = ct.get_ip_address_of_host()
    url = 'http://' + myIP + ':5002'
    # if ct.key_validate():
    #     print("Key Validation Successful. Press Any key to continue")
    #     input()
    if True:
        #ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        app.run(debug=False, host='0.0.0.0',port = 5002)
    else:
        print("Key invalid Contact your Software Provider")
        input()
