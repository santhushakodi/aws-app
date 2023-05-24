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
import base64
from scipy.signal import resample

from tensorflow import keras
from transformers import TFWav2Vec2Model, Wav2Vec2Processor


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

def truncate_resample_and_pad_wav(audio_data, sample_rate, max_length, target_sample_rate):

    # Resample the audio data
    audio_data_resampled = resample(audio_data, int(len(audio_data) * target_sample_rate / sample_rate))

    # Truncate or pad the resampled audio data
    if len(audio_data_resampled) > max_length:
        audio_data_resampled = audio_data_resampled[:max_length]
    elif len(audio_data_resampled) < max_length:
        padding = np.zeros(max_length - len(audio_data_resampled), dtype=np.int16)
        audio_data_resampled = np.concatenate((audio_data_resampled, padding))

    return target_sample_rate, audio_data_resampled


################# Murmur pitch transformer model ###########
model_name = "facebook/wav2vec2-base-960h"  # Replace with the desired pre-trained Wav2Vec2 model
base_model = TFWav2Vec2Model.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Add additional layers for classification
input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.float32)
features = base_model(input_layer).last_hidden_state
pooled_features = tf.keras.layers.GlobalMaxPooling1D()(features)
dense_layer1 = tf.keras.layers.Dense(10, activation=keras.layers.LeakyReLU(alpha=0.01))(pooled_features)
output_layer = tf.keras.layers.Dense(3, activation='softmax')(dense_layer1)

# Create the model
pitch_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
pitch_model.load_weights('pitch_new.h5')
pitch_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss="categorical_crossentropy", metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'],run_eagerly=True
    )

############################################Murmur shape###########################

shape_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
shape_model.load_weights('shape3.h5')
shape_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss="categorical_crossentropy", metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'],run_eagerly=True
    )

################################## Murmur timing ######################################################
timing_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
timing_model.load_weights('timing.h5')
timing_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss="categorical_crossentropy", metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'],run_eagerly=True
    )
######################################################################################################

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
    
    
    if 'data' in request.json and 'sampling_rate' in request.json:
        data_list = request.json['data']
        sampling_rate = request.json['sampling_rate']
        patient_id = request.json['patient_id']
    # print(request.json['patient_id'])
    # if 'wav_file' in request.files:
    #     file = request.files['wav_file']
    #     wav_data = file.read()
    #     patient_id = request.json['patient_id']    
        
    else:
        # Return an error response if required fields are missing
        print("incorrect")
        return jsonify({'error': 'Invalid request data.'}), 400
    
    # sampling_rate, audio_ = wavfile.read(file)

    # Convert the received list back to a NumPy array
    audio_ = np.array(data_list)
    # Load the model from the h5 file
    model = keras.models.load_model('outcome.h5')
    wavfile.write('heart.wav', sampling_rate, audio_)

    with open('heart.wav', 'rb') as file:
        binary_data = file.read()
    
    audio_slice = audio_[500:20500]
    
    filtered_audio = butter_bandpass_filt(audio_slice, 20, 600, sampling_rate ,order=12)

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

    if result[0][0] >= 0.5:
        outcome = "abnormal"
    else:
        outcome = "normal"


    ########################################################################
    max_length = 50000
    target_sample_rate = 4000

    resampled_sample_rate, truncated_resampled_data = truncate_resample_and_pad_wav(audio_,sampling_rate, max_length, target_sample_rate)
    input_data = truncated_resampled_data.astype("float32")
    input_data = input_data.reshape(1,50000,1)

    ####################################  Murmur pitch ###########################
    output = pitch_model.predict(input_data)
    pitch_dictionary = {0: "High", 1: "Low", 2: "Medium"}
    murmur_pitch = pitch_dictionary[np.argmax(output)]
    print("murmur_pitch : ",murmur_pitch)

    ######################################murmur shape############################3
    
    m_shape = shape_model.predict(input_data)
    shape_dict = {0:"Decrescendo", 1:"Diamond", 2:"Plateau"}

    murmur_shape = shape_dict[np.argmax(m_shape)]
    print(murmur_shape)

    ############################  Murmur timing ################################################
    
    m_timing = shape_model.predict(input_data)
    timing_dict = {0:"early systolic", 1:"holo systolic", 2:"mid systolic"}

    murmur_timing = timing_dict[np.argmax(m_timing)]
    print(murmur_timing)

    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    mycursor = mydb.cursor()
    # patient_id= int(patient_id)
    # query1 = "INSERT INTO newpatients (id, murmur) VALUES (%s, %s)"
    # data = (patient_id, outcome)
    query1 = "INSERT INTO patients (patient_id, murmur_case, clinical_outcome, murmur_timing, murmur_pitch, murmur_shape, pcg_signal) VALUES (%s, %s,%s, %s, %s, %s,%s)"
    data = (patient_id, "present",outcome, murmur_timing, murmur_pitch, murmur_shape, binary_data)
    print(data)
    mycursor.execute(query1, data)
    # commit the transaction
    mydb.commit()
    mycursor.close()
    mydb.close()
    return jsonify({'message': 'WAV file received and processed successfully.'}), 200


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
    # query2 = "SELECT murmur FROM newpatients WHERE id=%s"
    # data2 = (pid,)
    query2 = "SELECT * FROM patients WHERE patient_id=%s"
    data2 = (pid,)
    mycursor.execute(query2, data2)
    # fetch the result
    output = mycursor.fetchone()

    query3 = "SELECT COUNT(*) FROM patients WHERE clinical_outcome = %s"
    mycursor.execute(query3, ("normal",))
        
        # Fetch the result
    normal_count = mycursor.fetchone()[0]

    query4 = "SELECT COUNT(*) FROM patients WHERE clinical_outcome = %s"
    mycursor.execute(query4, ("abnormal",))
        
        # Fetch the result
    abnormal_count = mycursor.fetchone()[0]

    query5 = "SELECT COUNT(*) FROM patients"
    mycursor.execute(query5)
    total_patients = mycursor.fetchone()[0]


    mycursor.close()
    mydb.close()
    
    # result = {'message': 'Data processed successfully',
    #           'pid':pid,
    #           'murmur':murmur}
    # wave = base64.b64encode(output[6]).decode('utf-8')
    result = {'message':'data processed successfully',
              'pid':pid,
              'murmur_case':output[1],
              'murmur':output[2],
              'normal_count': normal_count,
              'abnormal_count': abnormal_count,
              'murmur_timing': output[3],
              'murmur_pitch': output[4],
              'murmur_shape':output[5],
              'total_patients':total_patients
              }
    # print(result)
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
        data = {"name":user[1]}

        if user and password == user[3]:
            session['email'] = email
            # return redirect('/dashboard')
            return render_template('index.html', data=data)
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

    
    query1 = "INSERT INTO newpatients (id, murmur) VALUES (%s, %s)"
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