import numpy as np
from flask import Flask, jsonify, request, redirect, session
import pickle
from flask import Flask, render_template
import mysql.connector

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfilt,sosfreqz
import pywt
import scipy
from scipy import signal
import os
import cv2
import hashlib

from tensorflow import keras


def butter_bandpass(lowcut, highcut, fs, order):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  sos = butter(order, [low, high], btype='band',output='sos')
  return sos

def butter_bandpass_filt(data, lowcut, highcut,fs, order):
  sos = butter_bandpass(lowcut, highcut, fs,order)
  y = sosfilt(sos, data)
  return y

def get_spectrogram(waveform):
  frame_length = 255
  frame_step = 128

  zero_padding = tf.zeros([20000] - tf.shape(waveform), dtype=tf.float64)

  # Concatenate audio with padding so that all audio clips will be of the same length
  waveform = tf.cast(waveform, tf.float64)
  equal_length_waveform = tf.concat([waveform, zero_padding], 0)



  spectrogram = tf.signal.stft(equal_length_waveform, frame_length=frame_length, frame_step=frame_step)
  spectrogram = tf.abs(spectrogram)

  return spectrogram

def plot_spectrogram(location , spectrogram, ax, title):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec ,cmap='inferno')
    plt.axis('off')
    plt.savefig(location, bbox_inches='tight', pad_inches=0)
    ax.set_xlim([0, 55000])
    ax.set_title(title)
    plt.close()


# load the model from the file
# with open('adaboost_classifier_with_spike_removed.pkl', 'rb') as f:
#     model = pickle.load(f)


# def preprocessing(heart_signal):
#     pass


app = Flask(__name__)
app.secret_key = 'vortex123'

@app.route('/')
def home():
    return render_template('auth-signin.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Invalid JSON payload'})

    payload = request.get_json()
    if 'data' not in payload:
        return jsonify({'error': 'No array data in the JSON payload'}), 400

    data_list = payload['data']
    patient_id = payload['patient_id']

    # Convert the received list back to a NumPy array
    audio_ = np.array(data_list)
    # Load the model from the h5 file
    model = keras.models.load_model('outcome.h5')
    
    
    audio_slice = audio_[500:20500]
    
    filtered_audio = butter_bandpass_filt(audio_slice, 20, 600, 4000,order=12)

    array = filtered_audio / np.max(filtered_audio)

    coeffs = pywt.wavedec(array,wavelet='db4',level=4)

    coeffs_arr , coeffs_slices = pywt.coeffs_to_array(coeffs)

    MAD = scipy.stats.median_abs_deviation(coeffs_arr)
    sigma = MAD/0.6745
    N = len(audio_slice)
    Threshold_ = sigma * ((2*np.log(N))**0.5)

    X = pywt.threshold(coeffs_arr, Threshold_, 'garrote')
    coeffs_filt = pywt.array_to_coeffs(X,coeffs_slices,output_format='wavedec')
    audio_sample = pywt.waverec(coeffs_filt,wavelet='db4')

    standarized_audio = (audio_sample - np.mean(audio_sample))/np.std(audio_sample)

    tensor1 = tf.convert_to_tensor(standarized_audio)

    spectrogram = get_spectrogram(tensor1)
    fig, ax = plt.subplots()
    plot_spectrogram('test.png',spectrogram.numpy(), ax, 'Spectrogram')  
    
    # Opens a image in RGB mode
    img = cv2.imread(r'test.png')
    resized_image = cv2.resize(img, (432,288))
    preprocessed_image = np.expand_dims(resized_image, axis=0) 
    result = model.predict(preprocessed_image)
    print(result)
    output = result.tolist()

    if result[0][0] == 1:
        outcome = "abnormal"
    else:
        outcome = "normal"

    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    mycursor = mydb.cursor()
    
    query1 = "INSERT INTO patients (id, murmur) VALUES (%s, %s)"
    data = (patient_id, outcome)
    mycursor.execute(query1, data)
    # commit the transaction
    mydb.commit()
    mycursor.close()
    mydb.close()
    return jsonify({'message': output})

@app.route('/search', methods=['POST'])
def murmur_show():
    data = request.get_json()
    pid = data['patient_id']
    print("patient id :", pid)

    # Process the data as required
    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    mycursor = mydb.cursor()
    query2 = "SELECT murmur FROM patients WHERE id=%s"
    data2 = (pid,)
    mycursor.execute(query2, data2)
    # fetch the result
    murmur = mycursor.fetchone()
    mycursor.close()
    mydb.close()
    
    result = {'message': 'Data processed successfully',
              'pid':pid,
              'murmur':murmur}
    return jsonify(result)

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['pwd']

        sha256_hash = hashlib.sha256()
        # Convert the input string to bytes and update the hash object
        sha256_hash.update(password.encode('utf-8'))
        # Get the hexadecimal representation of the hash value
        password = sha256_hash.hexdigest()
        
        print(email, password)

        mydb = mysql.connector.connect(
            host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
            user="admin",
            password="admin123",
            database="demodb"
            )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = mycursor.fetchone()
        mycursor.close()
        print(user)

        if user and password == user[3]:
            session['email'] = email
            # return redirect('/dashboard')
            return render_template('index.html')
        else:
            error = 'Invalid username or password'
            return render_template('auth-signin.html', error=error)

    return render_template('auth-signin.html')

@app.route('/writedb', methods=['POST'])
def update():
    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    mycursor = mydb.cursor()
    
    data = request.get_json()
    id = data.get("id")
    murmur = data.get("murmur")
    print(id,murmur)
    mycursor = mydb.cursor()

    
    query1 = "INSERT INTO patients (id, murmur) VALUES (%s, %s)"
    data = (id, murmur)
    mycursor.execute(query1, data)
    # commit the transaction
    mydb.commit()

    query2 = "SELECT * FROM patients WHERE id=%s"
    data2 = (id,)
    mycursor.execute(query2, data2)
    # fetch the result
    result = mycursor.fetchone()
    mycursor.close()
    mydb.close()

    resp = {"id":result[0],"mumur":result[1]}
    return jsonify(resp)

# @app.route('/matlab')
# def run():
#     result = engine.eval('disp("Hello, world!")')
#     return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')