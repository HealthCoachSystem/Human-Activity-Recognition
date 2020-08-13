import pandas as pd
import numpy as np
from scipy import stats
from tensorflow.keras.models import load_model
answer = ['pushup','situp','squat','sidebend','sidecruch']

def windows(data,size):
    start = 0
    while start< data.count():
        yield int(start), int(start + size)
        start+= (size/2)
# segmenting the time series
def segment_signal(data, window_size = 30):
    segments = np.empty((0,window_size,10))
    labels= np.empty((0))
    for (start, end) in windows(data['timestamp'],window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        a = data['x-rotate'][start:end]
        b = data['y-rotate'][start:end]
        c = data['z-rotate'][start:end]
        d = data['arms'][start:end]
        e = data['rrms'][start:end]
        f = data['roll'][start:end]
        g = data['pitch'][start:end]
        if(len(data['timestamp'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x,y,z,a,b,c,d,e,f,g])])
            labels = np.append(labels,stats.mode(data['activity'][start:end])[0][0])
    return segments, labels
from flask import Flask, jsonify, request
from tensorflow import keras

cnt =0

model = load_model('../model/CNN/model.h5')

app = Flask(__name__)
@app.route("/", methods=["POST"])
def index():
    global cnt
    
    cnt+=1
    
    data = request.json
    #print(data["xAcc"].split(','))
    #print()
    tmp_data = {
           'activity'  : ['pushup']*30,
           'timestamp' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
           'x-axis'    : list(map(float,data["xAcc"].split(','))),
           'y-axis'    : list(map(float,data["yAcc"].split(','))),
           'z-axis'    : list(map(float,data["zAcc"].split(','))),
           'x-rotate'  : list(map(float,data["xRot"].split(','))),
           'y-rotate'  : list(map(float,data["yRot"].split(','))),
           'z-rotate'  : list(map(float,data["zRot"].split(','))),
           'arms'      : list(map(float,data["AccRms"].split(','))),
           'rrms'      : list(map(float,data["RotRms"].split(','))),
           'roll'      : list(map(float,data["roll"].split(','))),
           'pitch'     : list(map(float,data["pitch"].split(',')))
           }

    dataset = pd.DataFrame(tmp_data)
    #print('dataset',len(dataset))
    # window size에 다가 50% 중첩이므로 totaldata/(windowsize/2)의 개수를 가진다 
    segments, labels = segment_signal(dataset) 
    #print('segments',len(segments))
    numOfRows = segments.shape[1]
    numOfColumns = segments.shape[2]
    reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
    testX = reshapedSegments
    #print('testX',len(testX))
    testX = np.nan_to_num(testX)
    #predictions = model.predict(test_x,verbose=2)
    predictions = model.predict_classes(testX)
    
    
    predictions=predictions.tolist()
    #print('predictions',len(predictions))
    
    print('cnt-------------------------->',cnt)
    #print(dataset)

    #return jsonify({"price":result_dic[result]})
    return jsonify({"price":answer[predictions[0]]})
 
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False)
