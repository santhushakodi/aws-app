import numpy as np
from flask import Flask, jsonify, request, redirect, session, send_file
import pickle
from flask import Flask, render_template
import mysql.connector
import io
import json

import tensorflow as tf
import tensorflow_io as tfio
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

def predict_disease(best_heard, timing, pitch, shape):
    print("predict disease : ", best_heard,timing,pitch,shape)
    timing_disease, pitch_disease, shape_disease, disease = "","","",""
    if timing == "Holo Systolic" and best_heard == "":
        timing_disease = "Mitral Regurgitation and Ventricular Septal Defect are possible based on murmur timing."
    elif timing == "Mid Systolic" and best_heard == "":
        timing_disease = "Aortic Stenosis and Atrial Septal Defect are possible based on murmur timing."
    elif timing == "Holo Systolic" and best_heard == "MV":
        timing_disease = "Mitral Regurgitation is possible based on murmur timing."
    elif timing == "Holo Systolic" and best_heard == "TV":
        timing_disease = "Ventricular Septal Defect is possible based on murmur timing."
    elif timing == "Mid Systolic" and best_heard == "AV":
        timing_disease = "Aortic Stenosis is possible based on murmur timing."
    elif timing == "Mid Systolic" and best_heard == "PV":
        timing_disease = "Atrial Septal Defect are possible based on murmur timing."
    
    if shape == "Plateau" and best_heard == "":
        shape_disease = "Mitral Regurgitation and Ventricular Septal Defect are possible based on murmur shape."
    elif shape == "Diamond" and best_heard == "":
        shape_disease = "Aortic Stenosis and Atrial Septal Defect are possible based on murmur shape."
    elif shape == "Plateau" and best_heard == "MV":
        shape_disease = "Mitral Regurgitation is possible based on murmur shape."
    elif shape == "Plateau" and best_heard == "TV":
        shape_disease = "Ventricular Septal Defect is possible based on murmur shape."
    elif shape == "Diamond" and best_heard == "AV":
        shape_disease = "Aortic Stenosis is possible based on murmur shape."
    elif shape == "Diamond" and best_heard == "PV":
        shape_disease = "Atrial Septal Defect is possible based on murmur shape."

    
    if pitch == "High" and best_heard == "":
        pitch_disease = "Mitral Regurgitation, Atrial Septal Defect and Ventricular Septal Defect are possible based on murmur pitch."
    elif pitch == "High" and best_heard == "MV":
        pitch_disease = "Mitral Regurgitation is possible based on murmur pitch."
    if pitch == "High" and best_heard == "PV":
        pitch_disease = "Atrial Septal Defect is possible based on murmur pitch."
    if pitch == "High" and best_heard == "TV":
        pitch_disease = "Ventricular Septal Defect is possible based on murmur pitch."
    
    disease = timing_disease + shape_disease + pitch_disease
    # if best_heard=="MV" and timing == "holo systolic" and pitch == "High" and shape == "Plateau":
    #     disease = "MR"
    # elif best_heard=="AV" and timing == "mid systolic" and shape == "Diamond":
    #     disease = "AS"
    # elif best_heard=="PV" and timing == "mid systolic" and pitch == "High" and shape == "Diamond":
    #     disease = "ASD"
    # elif best_heard=="TV" and timing == "holo systolic" and pitch == "High" and shape == "Plateau":
    #     disease = "VSD"   
    return disease

###########Normal abnormal############################################################
def truncate_resample_and_pad_wav1(audio_data, sample_rate, max_length, target_sample_rate):

    # Resample the audio data
    audio_data_resampled = resample(audio_data, int(len(audio_data) * target_sample_rate / sample_rate))

    # Truncate or pad the resampled audio data
    if len(audio_data_resampled) > max_length:
        audio_data_resampled = audio_data_resampled[:max_length]
    elif len(audio_data_resampled) < max_length:
        padding = np.zeros(max_length - len(audio_data_resampled), dtype=np.int16)
        audio_data_resampled = np.concatenate((audio_data_resampled, padding))

    return target_sample_rate, audio_data_resampled

def preprocess_wav(audio_data,samplerate):
    standarized_audio = (audio_data - np.mean(audio_data))/np.std(audio_data)
    _,truncated = truncate_resample_and_pad_wav1(standarized_audio, sample_rate=samplerate, max_length=50000, target_sample_rate=samplerate)
    return truncated

def flatten(l):
    return [item for sublist in l for item in sublist]
    
    
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
########################################Normal Abnormal##############################################
# Add additional layers for classification
input_layer1 = tf.keras.layers.Input(shape=(None,), dtype=tf.float32)
features1 = base_model(input_layer1).last_hidden_state
pooled_features1 = tf.keras.layers.GlobalMaxPooling1D()(features1)
dense_layer2 = tf.keras.layers.Dense(10, activation=keras.layers.LeakyReLU(alpha=0.01))(pooled_features1)
output_layer1 = tf.keras.layers.Dense(2, activation='softmax')(dense_layer2)

# Create the model
normal_abnormal_model = tf.keras.Model(inputs=input_layer1, outputs=output_layer1)
normal_abnormal_model.load_weights('p_7.h5')
normal_abnormal_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss="categorical_crossentropy", metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'],run_eagerly=True
    )

##################################################################################################

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
    data = request.get_json()
    if data:
        data_list = data['data']
        sampling_rate = data['sampling_rate']
        patient_id = data['patient_id']
        murmur_case = data['murmur_case']
        av_signal = data['AV']
        pv_signal = data['PV']
        tv_signal = data['TV']
        mv_signal = data['MV']
        print("success")
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
    # model = keras.models.load_model('outcome.h5')
    # audio_slice = audio_[500:20500]
    # filtered_audio = butter_bandpass_filt(audio_slice, 20, 600, sampling_rate ,order=12)

    # array = filtered_audio / np.max(filtered_audio)

    # coeffs = pywt.wavedec(array,wavelet='db4',level=4)

    # coeffs_arr , coeffs_slices = pywt.coeffs_to_array(coeffs)

    # MAD = scipy.stats.median_abs_deviation(coeffs_arr)
    # sigma = MAD/0.6745
    # N = len(audio_slice)
    # Threshold_ = sigma * ((2*np.log(N))**0.5)

    # X = pywt.threshold(coeffs_arr, Threshold_, 'garrote')
    # coeffs_filt = pywt.array_to_coeffs(X,coeffs_slices,output_format='wavedec')
    # audio_sample = pywt.waverec(coeffs_filt,wavelet='db4')

    # standarized_audio = (audio_sample - np.mean(audio_sample))/np.std(audio_sample)

    # tensor1 = tf.convert_to_tensor(standarized_audio)

    # spectrogram = get_spectrogram(tensor1)
    # fig, ax = plt.subplots()
    # plot_spectrogram('test.png',spectrogram.numpy(), ax, 'Spectrogram')  
    
    # # Opens a image in RGB mode
    # img = cv2.imread(r'test.png')
    # resized_image = cv2.resize(img, (432,288))
    # preprocessed_image = np.expand_dims(resized_image, axis=0) 
    # result = model.predict(preprocessed_image)
    # print(result)
    # output = result.tolist()

    # if result[0][0] >= 0.5:
    #     outcome = "abnormal"
    # else:
    #     outcome = "normal"

    ##############normal abnormal##############################
    normal_abnormal_input = []
    normal_abnormal_input.append(preprocess_wav(av_signal,sampling_rate))
    normal_abnormal_input.append(preprocess_wav(pv_signal,sampling_rate))
    normal_abnormal_input.append(preprocess_wav(tv_signal,sampling_rate))
    normal_abnormal_input.append(preprocess_wav(mv_signal,sampling_rate))

    normal_abnormal_input = flatten(normal_abnormal_input)
    normal_abnormal_input = np.array(normal_abnormal_input)
    normal_abnormal_input = normal_abnormal_input.reshape(1,200000)
    print(np.shape(np.array(normal_abnormal_input)))
    outcome = normal_abnormal_model.predict(normal_abnormal_input)
    normal_abnormal_dictionary = {0: "Abnormal", 1: "Normal"}
    clinical_outcome = normal_abnormal_dictionary[np.argmax(outcome)]
    print("clinical outcome : ",clinical_outcome)
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
    timing_dict = {0:"Early Systolic", 1:"Holo Systolic", 2:"Mid Systolic"}

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
    query1 = "INSERT INTO patients (patient_id, murmur_case, clinical_outcome, murmur_timing, murmur_pitch, murmur_shape) VALUES (%s, %s,%s, %s, %s, %s)"
    data = (patient_id, murmur_case ,clinical_outcome, murmur_timing, murmur_pitch, murmur_shape)
    # print(data)
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
    doctor_name = session.get('username')
    print("session name : ", session.get('username'))
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

    query5 = "SELECT COUNT(*) FROM patient_details WHERE doctor_name = %s"
    mycursor.execute(query5,(doctor_name,))
    total_patients = mycursor.fetchone()[0]

    query6 = "SELECT COUNT(*) FROM patients WHERE murmur_timing = %s"
    mycursor.execute(query6,("early systolic",)) 
    early_systolic_count = mycursor.fetchone()[0]

    query7 = "SELECT COUNT(*) FROM patients WHERE murmur_timing = %s"
    mycursor.execute(query7,("holo systolic",)) 
    holo_systolic_count = mycursor.fetchone()[0]

    query8 = "SELECT COUNT(*) FROM patients WHERE murmur_timing = %s"
    mycursor.execute(query8,("mid systolic",)) 
    mid_systolic_count = mycursor.fetchone()[0]

    query9 = "SELECT COUNT(*) FROM patients WHERE murmur_shape= %s"
    mycursor.execute(query9,("Decrescendo",))
    decrescendo_count = mycursor.fetchone()[0]

    query10 = "SELECT COUNT(*) FROM patients WHERE murmur_shape= %s"
    mycursor.execute(query10,("Diamond",))
    diamond_count = mycursor.fetchone()[0]

    query11 = "SELECT COUNT(*) FROM patients WHERE murmur_shape= %s"
    mycursor.execute(query11,("Plateau",))
    plateau_count = mycursor.fetchone()[0]
    
    query12 = "SELECT * FROM patient_details WHERE patient_id=%s"
    data12 = (pid,)
    mycursor.execute(query12, data12)
    # fetch the result
    detail = mycursor.fetchone()
    print("detail :" ,detail)
    if detail:
        print(detail[3],output[3],output[4],output[5])
        disease = predict_disease(detail[3],output[3],output[4],output[5])
    else:
        print("detail None")
        disease = predict_disease("",output[3],output[4],output[5])
    print("disease : ", disease)

    query13 = "select patient_id from patient_details where doctor_name=%s"
    data13 = (doctor_name,)
    mycursor.execute(query13, data13)
    rows = mycursor.fetchall()
    patient_ids = [row[0] for row in rows]

     # Replace with your actual list of IDs

    # Convert the list of IDs to a comma-separated string
    id_string = ','.join(patient_ids)

    # Execute the query to retrieve the desired column and count
    query14 = f"SELECT clinical_outcome, COUNT(*) FROM patients WHERE patient_id IN ({id_string}) GROUP BY clinical_outcome"
    mycursor.execute(query14)
    clinical_rows = mycursor.fetchall()

    query15 = f"SELECT murmur_timing, COUNT(*) FROM patients WHERE patient_id IN ({id_string}) GROUP BY murmur_timing"
    mycursor.execute(query15)
    timing_rows = mycursor.fetchall()

    query16 = f"SELECT murmur_shape, COUNT(*) FROM patients WHERE patient_id IN ({id_string}) GROUP BY murmur_shape"
    mycursor.execute(query16)
    shape_rows = mycursor.fetchall()

    mycursor.close()
    mydb.close()

    username = session.get('username')
    
    clinical_result = {}
    for row in clinical_rows:
        column_value = row[0]
        count = row[1]
        clinical_result[column_value] = count
    print(clinical_result)
    normal_count = clinical_result.get("Normal",0)
    abnormal_count = clinical_result.get("Abnormal",0)

    timing_result = {}
    for row in timing_rows:
        timing_result[row[0]] = row[1]
    print(timing_result)
    early_systolic_count = timing_result.get("Early Systolic",0)
    holo_systolic_count = timing_result.get("Holo Systolic",0)
    mid_systolic_count = timing_result.get("Mid Systolic",0)

    shape_result = {}
    for row in shape_rows:
        shape_result[row[0]] = row[1]
    print(shape_result)
    decrescendo_count = shape_result.get('Decrescendo',0)
    diamond_count = shape_result.get('Diamond',0)
    plateau_count = shape_result.get('Plateau',0)

    result = {'message':'data processed successfully',
              'pid':pid,
              'disease':disease,
              'murmur':output[2],
              'normal_count': normal_count,
              'abnormal_count': abnormal_count,
              'murmur_timing': output[3],
              'murmur_pitch': output[4],
              'murmur_shape':output[5],
              'total_patients':total_patients,
              'early_systolic_count':early_systolic_count,
              'holo_systolic_count':holo_systolic_count,
              'mid_systolic_count':mid_systolic_count,
              'decrescendo_count': decrescendo_count,
              'diamond_count':diamond_count,
              'plateau_count':plateau_count,
              'name': username
              }
    # print(result)
    if output[1] == "Absent":
        result['murmur_pitch'] = ""
        result['murmur_shape'] = ""
        result['murmur_timing'] = ""
    
    
    # return render_template("index.html", wav_data = encoded_wav)
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
            session['username'] = data["name"]
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

@app.route('/add', methods=['POST'])
def add():
    if request.get_json():
        data = request.get_json()
        patient_id = data['inserted_patient_id']
        patient_name = data['inserted_patient_name']
        doctor_name = data['inserted_doctor_name']
        best_heard = data['inserted_best_heard']

        mydb = mysql.connector.connect(
            host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
            user="admin",
            password="admin123",
            database="demodb"
            )
        mycursor = mydb.cursor()
        
        query1 = "INSERT INTO patient_details (patient_id, patient_name, doctor_name, best_heard) VALUES (%s, %s, %s, %s)"
        data = (patient_id,patient_name,doctor_name,best_heard)
        print("add patient")
        mycursor.execute(query1, data)
        # commit the transaction
        mydb.commit()
        mycursor.close()
        mydb.close()
        return jsonify({'success': 'Successfully added patient'}), 200

    else:   
        return jsonify({'error': 'Invalid request data.'}), 400
    
@app.route('/upload', methods=['POST'])
def upload():
    # print("headers:",request.headers)
    if request.files:
        pid = request.headers.get('patient_id')
        print(pid)
        av = request.files['AV'].read()
        pv = request.files['PV'].read()
        tv = request.files['TV'].read()
        mv = request.files['MV'].read()
        
        mydb = mysql.connector.connect(
            host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
            user="admin",
            password="admin123",
            database="demodb"
            )
        mycursor = mydb.cursor()
        
        query13 = "INSERT INTO pcg_table (patient_id, AV, PV, TV, MV) VALUES (%s, %s, %s, %s, %s)"
        data = (pid,av,pv,tv,mv)
        mycursor.execute(query13, data)
        # commit the transaction
        mydb.commit()
        mycursor.close()
        mydb.close()
        return jsonify({'success': 'Successfully added patient'}), 200
    else:
        return jsonify({'failed': 'failed'}), 400

@app.route('/render_audio', methods=['GET','POST'])
def render_audio():
    print(request.method)
    id = request.args.get('query')
    print(id)
    if id:
        mydb = mysql.connector.connect(
                    host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
                    user="admin",
                    password="admin123",
                    database="demodb"
                    )
        mycursor = mydb.cursor()
        
        query13 = "select * from pcg_table where patient_id=%s"
        data = (id,)
        mycursor.execute(query13, data)
        # commit the transaction
        detail = mycursor.fetchone()
        mycursor.close()
        mydb.close()
        encoded_wav1 = base64.b64encode(detail[1]).decode('utf-8')
        encoded_wav2 = base64.b64encode(detail[2]).decode('utf-8')
        encoded_wav3 = base64.b64encode(detail[3]).decode('utf-8')
        encoded_wav4 = base64.b64encode(detail[4]).decode('utf-8')
        print(type(encoded_wav1))
        return render_template("temp.html", 
                                wav_data1=encoded_wav1,
                                wav_data2=encoded_wav2,
                                wav_data3=encoded_wav3,
                                wav_data4=encoded_wav4)
    
    else:
        print("No data")
        return render_template("temp.html", 
                                wav_data1="",
                                wav_data2="",
                                wav_data3="",
                                wav_data4="")

@app.route('/home', methods=['GET', "POST"])
def render_home():
    username = session.get('username')
    return render_template("index.html", data = {"name":username})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')