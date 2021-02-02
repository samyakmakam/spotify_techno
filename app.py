import os
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import gunicorn
scaler = StandardScaler()
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import category_encoders as ce
import zipfile
import os
app = Flask(__name__)
model = pickle.load(open('rf.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        

        hist_user_behavior_is_shuffle = float(request.form['hist_user_behavior_is_shuffle'])       
        session_position = float(request.form['session_position'])
        session_length = float(request.form['session_length'])
        context_switch = float(request.form['session_length'])        
        no_pause_before_play = float(request.form['no_pause_before_play'])                
        hist_user_behavior_n_seekfwd = float(request.form['hist_user_behavior_n_seekfwd'])                
        hist_user_behavior_n_seekback = float(request.form['hist_user_behavior_n_seekback'])                
        premium = float(request.form['premium'])  
        context_type_catalog=float(request.form['context_type_catalog'])
        context_type_charts=float(request.form['context_type_charts'])
        context_type_editorial_playlist=float(request.form['context_type_editorial_playlist'])
        context_type_personalized_playlist=float(request.form['context_type_personalized_playlist'])
        context_type_radio=float(request.form['context_type_radio'])
        context_type_user_collection=float(request.form['context_type_user_collection'])
        hist_user_behavior_reason_start_appload = float(request.form['hist_user_behavior_reason_start_appload'])
        hist_user_behavior_reason_start_backbtn = float(request.form['hist_user_behavior_reason_start_backbtn'])
        hist_user_behavior_reason_start_clickrow = float(request.form['hist_user_behavior_reason_start_clickrow'])
        hist_user_behavior_reason_start_endplay = float(request.form['hist_user_behavior_reason_start_endplay'])
        hist_user_behavior_reason_start_fwdbtn = float(request.form['hist_user_behavior_reason_start_fwdbtn'])
        hist_user_behavior_reason_start_playbtn = float(request.form['hist_user_behavior_reason_start_playbtn'])
        hist_user_behavior_reason_start_remote = float(request.form['hist_user_behavior_reason_start_remote'])
        hist_user_behavior_reason_start_trackdone = float(request.form['hist_user_behavior_reason_start_trackdone'])
        hist_user_behavior_reason_start_trackerror = float(request.form['hist_user_behavior_reason_start_trackerror'])
        hist_user_behavior_reason_end_backbtn = float(request.form['hist_user_behavior_reason_end_backbtn'])
        hist_user_behavior_reason_end_clickrow = float(request.form['hist_user_behavior_reason_end_clickrow'])
        hist_user_behavior_reason_end_endplay = float(request.form['hist_user_behavior_reason_end_endplay'])
        hist_user_behavior_reason_end_fwdbtn = float(request.form['hist_user_behavior_reason_end_fwdbtn'])
        hist_user_behavior_reason_end_logout = float(request.form['hist_user_behavior_reason_end_logout'])
        hist_user_behavior_reason_end_remote = float(request.form['hist_user_behavior_reason_end_remote'])
        hist_user_behavior_reason_end_trackdone = float(request.form['hist_user_behavior_reason_end_trackdone'])
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
        is_major = float(request.form['is_major'])
        organism = float(request.form['organism'])         
        speechiness = float(request.form['speechiness'])         
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

        prediction=model.predict([[session_position,session_length,context_switch,no_pause_before_play,hist_user_behavior_n_seekfwd,hist_user_behavior_n_seekback,hist_user_behavior_is_shufffle,premium,context_type_catalog,context_type_charts,context_type_editorial_playlist,context_type_personalized_playlist,context_type_radio,context_type_user_collection,
                                   hist_user_behavior_reason_start_appload,hist_user_behavior_reason_start_backbtn,hist_user_behavior_reason_start_clickrow,hist_user_behavior_reason_start_endplay,
                                   hist_user_behavior_reason_start_fwdbtn,hist_user_behavior_reason_start_playbtn,hist_user_behavior_reason_start_remote,hist_user_behavior_reason_start_trackdone,
                                   hist_user_behavior_reason_start_trackerror,hist_user_behavior_reason_end_backbtn,hist_user_behavior_reason_end_clickrow,hist_user_behavior_reason_end_endplay,
                                   hist_user_behavior_reason_end_fedbtn,hist_user_behavior_reason_end_logout,hist_user_behavior_reason_end_remote,hist_user_behavior_reason_end_trackdone,
                                   duration,us_popularity_estimate,acousticness,beat_strength,bounciness,danceability,dyn_range_mean,energy,instrumentalness,flatness,liveness,loudness,mechanism,is_major,organism,speechiness,tempo,valence,acoustic_vector_0,acoustic_vector_1,acoustic_vector_2,acoustic_vector_3,acoustic_vector_4,acoustic_vector_5, acoustic_vector_6,acoustic_vector_7]]) 
        p=round(prediction[0],0)
         
        if p==0:
            return render_template('index.html',prediction_texts="Based on the given inputs, the song should not be skipped")
        else:
            return render_template('index.html',prediction_texts="Based on the given inputs, the song should be skipped")
    else:
        return render_template('index.html',prediction_texts="Input error")

if __name__=="__main__":
    app.run(debug=True)
