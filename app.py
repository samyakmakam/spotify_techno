import os
from flask import Flask, render_template, request
import pandas as pd
import jsonify
import pickle
import numpy as np
import sklearn
import os
import numpy as np
app = Flask(__name__)
model = pickle.load(open('rf.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
        hist_user_behavior_is_shuffle = float(request.form['hist_user_behavior_is_shuffle'])
        hist_user_seek_behavior = float(request.form['hist_user_seek_behavior'])        
        time_of_day = str(request.form['time_of_day'])  
        release_condition = str(request.form['release_condition'])
        session_position = float(request.form['session_position'])
        session_length = float(request.form['session_length'])
        context_switch = float(request.form['session_length'])        
        no_pause_before_play = float(request.form['no_pause_before_play'])                
        hist_user_behavior_n_seekfwd = float(request.form['hist_user_behavior_n_seekfwd'])                
        hist_user_behavior_n_seekback = float(request.form['hist_user_behavior_n_seekback'])                
        premium = float(request.form['premium'])  
        context_type = str(request.form['context_type']) 
        hist_user_behavior_reason_start = str(request.form['hist_user_behavior_reason_start'])     
        hist_user_behavior_reason_end = str(request.form['hist_user_behavior_reason_end'])
        duration = float(request.form['duration'])       
        us_popularity_estimate = float(request.form['us_popularity_estimate'])       
        acousticness = float(request.form['acousticness'])       
        beat_strength = float(request.form['beat_strength'])       
        bounciness = float(request.form['bounciness'])         
        danceability = float(request.form['danceability'])         
        dyn_range_mean = float(request.form['dyn_range_mean'])         
        energy = float(request.form['energy'])         
        instrumentalness = float(request.form['instrumentalness'])         
        flatness = float(request.form['flatness'])         
        liveness = float(request.form['liveness'])         
        loudness = float(request.form['loudness'])         
        mechanism = float(request.form['mechanism'])         
        organism = float(request.form['organism'])         
        speechiness = float(request.form['speechiness'])         
        mode = str(request.form['mode'])
        tempo = float(request.form['tempo']) 
        valence = float(request.form['valence']) 
        acoustic_vector_0 = float(request.form['acoustic_vector_0']) 
        acoustic_vector_1 = float(request.form['acoustic_vector_1']) 
        acoustic_vector_2 = float(request.form['acoustic_vector_2']) 
        acoustic_vector_3 = float(request.form['acoustic_vector_3']) 
        acoustic_vector_4 = float(request.form['acoustic_vector_4'])    
        acoustic_vector_5 = float(request.form['acoustic_vector_5']) 
        acoustic_vector_6 = float(request.form['acoustic_vector_6']) 
        acoustic_vector_7 = float(request.form['acoustic_vector_7']) 
        arr=np.array([[ hist_user_behavior_is_shuffle,hist_user_seek_behavior,time_of_day,release_condition,session_position,session_length,context_switch,no_pause_before_play,hist_user_behavior_n_seekfwd,hist_user_behavior_n_seekback,premium,context_type,hist_user_behavior_reason_start,hist_user_behavior_reason_end,duration,us_popularity_estimate,acousticness,beat_strength,bounciness,danceability,dyn_range_mean,energy,instrumentalness,flatness,liveness,loudness,mechanism,organism,speechiness,mode,tempo,valence,acoustic_vector_0,acoustic_vector_1,acoustic_vector_2,acoustic_vector_3,acoustic_vector_4,acoustic_vector_5, acoustic_vector_6,acoustic_vector_7, ]])
        pred=model.predict(arr)
        if pred==0:
            return render_template('index.html',prediction_texts="Based on the given inputs, the song should not be skipped")
        elif pred==1:
            return render_template('index.html',prediction_texts="Based on the given inputs, the song should be skipped")

if __name__=="__main__":
    app.run(debug=True)
