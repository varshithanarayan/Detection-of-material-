from flask import *
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
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("select * from tester where email=%s and password=%s",(email,ct.md5(pwd)))
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

@app.route('/tests', methods=['GET', 'POST'])
@is_logged_in
def tests():
    cur=mysql.connection.cursor()
    cur.execute("select input_image,output_image,status from tests where username = %s",([session['username']]))
    tests = cur.fetchall()
    cur.close()
    return render_template('test.html',data = session['username'],url = url, tests = tests)

global output_file
@app.route('/alexnet_predict', methods=['GET', 'POST'])
@is_logged_in
def alexnet_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static','input_images',secure_filename(f.filename))
        f.save(file_path)
        output_file = f.filename
        path = file_path = os.path.join('static','input_images',secure_filename(output_file))
        output_image = ct.predict_alexnet(path)
        # Replace 'path_to_image' with the actual path to your image file
        cur=mysql.connection.cursor()
        cur.execute("insert into alexnet_tests(username,input_image,output_image,status) values(%s,%s,%s,%s)",(session['username'],path,output_image["Name"],"success"))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('alexnet'))
#     return render_template('test.html',url=url,data = session['username'],tests = tests)

@app.route("/alexnet",methods=['POST','GET'])
@is_logged_in
def alexnet():
    cur=mysql.connection.cursor()
    cur.execute("select input_image,output_image,status from alexnet_tests where username = %s",([session['username']]))
    tests = cur.fetchall()
    cur.close()
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[1, 'memo']
    return render_template('alexnet.html',url = url,data = session['username'],tests = tests,train_memo=memo)
@app.route("/save_memo_alexnet", methods=['GET', 'POST'])
@is_logged_in
def save_memo_alexnet():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[2, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route("/save_memo_vgg16", methods=['GET', 'POST'])
@is_logged_in
def save_memo_vgg16():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[3, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route("/save_memo_resnet", methods=['GET', 'POST'])
@is_logged_in
def save_memo_resnet():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[4, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"
@app.route("/save_memo_densenet", methods=['GET', 'POST'])
@is_logged_in
def save_memo_densenet():
    if request.method == "POST":
         memo = request.form["memo"]
    print(memo)
    df = pd.read_csv('../Memo/memo.csv')
    df.loc[5, 'memo'] = memo
    df.to_csv('../Memo/memo.csv', index=False)
    return "Memo Saved Successfully"



@app.route("/resnet",methods=['POST','GET'])
@is_logged_in
def resnet():
    cur=mysql.connection.cursor()
    cur.execute("select input_image,output_image,status from resnet_tests where username = %s",([session['username']]))
    tests = cur.fetchall()
    cur.close()
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[1, 'memo']
    return render_template('resnet.html',url = url,data = session['username'],tests = tests,train_memo=memo)


@app.route("/vgg16",methods=['POST','GET'])
@is_logged_in
def vgg16():
    cur=mysql.connection.cursor()
    cur.execute("select input_image,output_image,status from VGG16_tests where username = %s",([session['username']]))
    tests = cur.fetchall()
    cur.close()
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[1, 'memo']
    return render_template('VGG16.html',url = url,data = session['username'],tests = tests,train_memo=memo)


@app.route("/densenet",methods=['POST','GET'])
@is_logged_in
def densenet():
    cur=mysql.connection.cursor()
    cur.execute("select input_image,output_image,status from densenet_tests where username = %s",([session['username']]))
    tests = cur.fetchall()
    cur.close()
    df = pd.read_csv('../Memo/memo.csv')
    memo = df.loc[1, 'memo']
    return render_template('densenet.html',url = url,data = session['username'],tests = tests,train_memo=memo)







@app.route('/VGG16_predict', methods=['GET', 'POST'])
@is_logged_in
def VGG16_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static','input_images',secure_filename(f.filename))
        f.save(file_path)
        output_file = f.filename
        path = file_path = os.path.join('static','input_images',secure_filename(output_file))
        output_image = ct.predict_VGG16(path)
        # Replace 'path_to_image' with the actual path to your image file
        cur=mysql.connection.cursor()
        cur.execute("insert into VGG16_tests(username,input_image,output_image,status) values(%s,%s,%s,%s)",(session['username'],path,output_image["Name"],"success"))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('vgg16'))


@app.route('/resnet_predict', methods=['GET', 'POST'])
@is_logged_in
def resnet_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static','input_images',secure_filename(f.filename))
        f.save(file_path)
        output_file = f.filename
        path = file_path = os.path.join('static','input_images',secure_filename(output_file))
        output_image = ct.predict_resnet(path)
        # Replace 'path_to_image' with the actual path to your image file
        cur=mysql.connection.cursor()
        cur.execute("insert into resnet_tests(username,input_image,output_image,status) values(%s,%s,%s,%s)",(session['username'],path,output_image["Name"],"success"))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('resnet'))


@app.route('/densenet_predict', methods=['GET', 'POST'])
@is_logged_in
def densenet_predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static','input_images',secure_filename(f.filename))
        f.save(file_path)
        output_file = f.filename
        path = file_path = os.path.join('static','input_images',secure_filename(output_file))
        output_image = ct.predict_densenet(path)
        # Replace 'path_to_image' with the actual path to your image file
        cur=mysql.connection.cursor()
        cur.execute("insert into densenet_tests(username,input_image,output_image,status) values(%s,%s,%s,%s)",(session['username'],path,output_image["Name"],"success"))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('densenet'))

#Home page
@app.route("/home",methods=['POST','GET'])
@is_logged_in
def home():
    global url
    if request.method=='POST':
        if request.form.get("submit") == "AlexNet":
            return redirect('alexnet')
        if request.form.get("submit") == "VGG-16":
            return redirect('vgg16')
        if request.form.get("submit") == "ResNet-50":     
            return redirect('resnet')
        if request.form.get("submit") == "DenseNet-169":
            return redirect('densenet')
    return render_template('index.html',data = session['username'],url = url)

@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == '__main__':
    global url
    app.secret_key='secret123'
    myIP = ct.get_ip_address_of_host()
    url = 'http://' + myIP + ':5001'
    #key = input('Enter 64 length Key')
    #if ct.key_validate(key):
       #print("Key Validation Successful. Press Any key to continue")
       #input()
        #ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    app.run(debug=False, host='0.0.0.0',port = 5001)
    #else:
       #print("Key invalid Contact your Software Provider")
        #input()
