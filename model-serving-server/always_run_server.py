import pandas as pd
import numpy as np
from scipy import stats
from tensorflow.keras.models import load_model
answer = ['break','pushup','sidebend','sidecruch','situp','squat']
#answer= ['squat','break','sidecruch','situp','sidebend','pushup']

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

# CNN 서버 실행

from flask import Flask, jsonify, request, render_template
from tensorflow import keras

cnt=0
model = load_model('../model/CNN/model.h5')
isFirst=True
tmp_data = {
            'activity'  : ['pushup']*30,
            'timestamp' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            'x-axis'    : None,
            'y-axis'    : None,
            'z-axis'    : None,
            'x-rotate'  : None,
            'y-rotate'  : None,
            'z-rotate'  : None,
            'arms'      : None,
            'rrms'      : None,
            'roll'      : None,
            'pitch'     : None
}

def data_slicing(data):
    return data[15:]

app = Flask(__name__)
@app.route("/", methods=["POST","GET"])
def index():
    
    if request.method=='GET':
        return render_template('test.html', name='2')
    
    global cnt
    global isFirst
    cnt+=1
    
    data = request.json
    if isFirst:
        tmp_data['x-axis']=list(map(float,data["xAcc"].split(',')))
        tmp_data['y-axis']=list(map(float,data["yAcc"].split(',')))
        tmp_data['z-axis']=list(map(float,data["zAcc"].split(',')))
        tmp_data['x-rotate']=list(map(float,data["xRot"].split(',')))
        tmp_data['y-rotate']=list(map(float,data["yRot"].split(',')))
        tmp_data['z-rotate']=list(map(float,data["zRot"].split(',')))
        tmp_data['arms']=list(map(float,data["AccRms"].split(',')))
        tmp_data['rrms' ]=list(map(float,data["RotRms"].split(',')))
        tmp_data['roll' ]=list(map(float,data["roll"].split(',')))
        tmp_data['pitch']=list(map(float,data["pitch"].split(',')))
        isFirst = False
        return jsonify({"price":'test'})
    else:
        tmp_data['x-axis']+=list(map(float,data["xAcc"].split(',')))
        tmp_data['y-axis']+=list(map(float,data["yAcc"].split(',')))
        tmp_data['z-axis']+=list(map(float,data["zAcc"].split(',')))
        tmp_data['x-rotate']+=list(map(float,data["xRot"].split(',')))
        tmp_data['y-rotate']+=list(map(float,data["yRot"].split(',')))
        tmp_data['z-rotate']+=list(map(float,data["zRot"].split(',')))
        tmp_data['arms']+=list(map(float,data["AccRms"].split(',')))
        tmp_data['rrms' ]+=list(map(float,data["RotRms"].split(',')))
        tmp_data['roll' ]+=list(map(float,data["roll"].split(',')))
        tmp_data['pitch']+=list(map(float,data["pitch"].split(',')))
        
        dataset = pd.DataFrame(tmp_data)
        
        tmp_data['x-axis']=data_slicing(tmp_data['x-axis'])
        tmp_data['y-axis']=data_slicing(tmp_data['y-axis'])
        tmp_data['z-axis']=data_slicing(tmp_data['z-axis'])
        tmp_data['x-rotate']=data_slicing(tmp_data['x-rotate'])
        tmp_data['y-rotate']=data_slicing(tmp_data['y-rotate'])
        tmp_data['z-rotate']=data_slicing(tmp_data['z-rotate'])
        tmp_data['arms']=data_slicing(tmp_data['arms'])
        tmp_data['rrms' ]=data_slicing(tmp_data['rrms'])
        tmp_data['roll' ]=data_slicing(tmp_data['roll'])
        tmp_data['pitch']=data_slicing(tmp_data['pitch'])
        
        
        # window size에 다가 50% 중첩이므로 totaldata/(windowsize/2)의 개수를 가진다 
        segments, labels = segment_signal(dataset) 
        numOfRows = segments.shape[1]
        numOfColumns = segments.shape[2]
        reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
        testX = reshapedSegments
        testX = np.nan_to_num(testX)
        predictions = model.predict_classes(testX)
        predictions=predictions.tolist()
        
        print('cnt-------------------------->',cnt)
        return jsonify({"exercise":answer[predictions[0]]})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False)
#application = app
