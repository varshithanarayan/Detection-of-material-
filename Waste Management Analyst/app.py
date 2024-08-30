from flask import *
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
import controller as ct

from time import sleep
from flask_mysqldb import MySQL
from tqdm import tqdm
import socket
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
#index route
@app.route('/')
def index():
    return render_template('login.html',error_message = '',url = url,)
#login to the application
@app.route('/login',methods=['POST','GET'])
def login():
    status=True
    if request.method=='POST':
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("select * from user where email=%s and password=%s",(email,pwd))
        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["username"]
            flash('Login Successfully','success')
            return redirect('home')
        else:
            flash('Invalid Login Credentials. Try Again','danger')
    return render_template("login.html",url = url,)



#login validation
def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap
  
#Registration routing
@app.route('/reg',methods=['POST','GET'])
def reg():
    status=False
    if request.method=='POST':
        name=request.form["uname"]
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("select * from user where username=%s",[name])
        data=cur.fetchone()
        if not data:
            cur.execute("insert into user(username,password,email) values(%s,%s,%s)",(name,pwd,email))
            mysql.connection.commit()
            cur.close()
            flash('Registration Successfully. Login Now...','success')
        else:
            flash('Username Exists...Kindly use different username','danger')
        return redirect('login')
    return render_template("login.html",status=status,url = url,data = session['username'])



@app.route("/save_memo_alexnet", methods=['GET', 'POST'])
@is_logged_in
def save_memo_alexnet():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[6, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route("/save_memo_vgg16", methods=['GET', 'POST'])
@is_logged_in
def save_memo_vgg16():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[7, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route("/save_memo_resnet", methods=['GET', 'POST'])
@is_logged_in
def save_memo_resnet():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[8, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route("/save_memo_densenet", methods=['GET', 'POST'])
@is_logged_in
def save_memo_densenet():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[9, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"

@app.route("/alexnet",methods=['POST','GET'])
@is_logged_in
def alexnet():
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[2, 'memo']
    return render_template('alexnet.html',url = url,data = session['username'],test_memo=memo)

@app.route("/resnet",methods=['POST','GET'])
@is_logged_in
def resnet():
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[4, 'memo']
    return render_template('resnet.html',url = url,data = session['username'],test_memo=memo)

@app.route("/VGG16",methods=['POST','GET'])
@is_logged_in
def VGG16():
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[3, 'memo']
    return render_template('VGG16.html',url = url,data = session['username'],test_memo=memo)

@app.route("/densenet",methods=['POST','GET'])
@is_logged_in
def densenet():
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[5, 'memo']
    return render_template('densenet.html',url = url,data = session['username'],test_memo=memo)

global output_file
@app.route('/alexnet_predict', methods=['GET', 'POST'])
@is_logged_in
def alexnet_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static',secure_filename(f.filename))
        f.save(file_path)
        output_file = file_path
    return render_template('alexnet_result.html',url=url,filename = file_path,data = session['username'])

@app.route('/get_result_alexnet', methods=['GET'])
def get_result_alexnet():
    print('hello')
    global output_file
    output = ct.predict_alexnet(output_file)
    # Replace 'path_to_image' with the actual path to your image file
    return jsonify(output)


@app.route('/get_result_VGG16', methods=['GET'])
def get_result_VGG16():
    print('hello')
    global output_file
    output = ct.predict_VGG16(output_file)
    # Replace 'path_to_image' with the actual path to your image file
    return jsonify(output)

@app.route('/VGG16_predict', methods=['GET', 'POST'])
@is_logged_in
def VGG16_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static',secure_filename(f.filename))
        f.save(file_path)
        output_file = file_path
    return render_template('VGG16_result.html',url=url,filename = file_path,data = session['username'])

@app.route('/get_result_resnet', methods=['GET'])
def get_result_resnet():
    print('hello')
    global output_file
    output = ct.predict_resnet(output_file)
    # Replace 'path_to_image' with the actual path to your image file
    return jsonify(output)

@app.route('/resnet_predict', methods=['GET', 'POST'])
@is_logged_in
def resnet_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static',secure_filename(f.filename))
        f.save(file_path)
        output_file = file_path
    return render_template('resnet_result.html',url=url,filename = file_path,data = session['username'])

@app.route('/get_result_densenet', methods=['GET'])
def get_result_densenet():
    print('hello')
    global output_file
    output = ct.predict_densenet(output_file)
    # Replace 'path_to_image' with the actual path to your image file
    return jsonify(output)
@app.route('/densenet_predict', methods=['GET', 'POST'])
@is_logged_in
def densenet_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static',secure_filename(f.filename))
        f.save(file_path)
        output_file = file_path
    return render_template('densenet_result.html',url=url,filename = file_path,data = session['username'])


@app.route("/home",methods=['POST','GET'])
@is_logged_in
def home():
    global url
    if request.method=='POST':
        if request.form.get("submit") == "AlexNet":
            return redirect('alexnet')
        if request.form.get("submit") == "VGG-16":
            return redirect('VGG16')
        if request.form.get("submit") == "ResNet-50":     
            return redirect('resnet')
        if request.form.get("submit") == "DenseNet-169":
            return redirect('densenet')
    return render_template('index.html',data = session['username'],url = url)

#logout
@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == '__main__':
    global url
    app.secret_key='secret123'
    myIP = ct.get_ip_address_of_host()
    url = 'http://' + myIP + ':5000'
    #key = input("Enter 64 length Key To Start Server\n")
    #if ct.key_validate(key):
        #print("Key Validation Successful. Press Any key to continue")
        #input()
        #ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    app.run(debug=False, host='0.0.0.0',port = 5000)
    #else:
        #print("Key invalid Contact your Software Provider")
        #input()
