import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
import base64
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import tensorflow as tf 
import numpy as np
#import cv2
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
from flask import Flask, render_template, request, session, redirect
import pandas as pd
from bokeh.embed import components
import m8r
import segyio
import lasio
from bokeh.models import ColumnDataSource, LogColorMapper, LogTicker, ColorBar, LinearColorMapper, BasicTicker, FuncTickFormatter
from bokeh.transform import linear_cmap

# Obtain the flask app object
app = flask.Flask(__name__)
user = str(os.urandom(24))
#os.mkdir ('./static/%s'%(user))

UPLOAD_FOLDER='./static'
def load_graph(trained_model):   
    with tf.io.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph

def create_figure2(eval_dat,depth):
    p = figure(title="Gamma-ray log after preprocessing", plot_width=800, plot_height=250, x_axis_label='Sample position', y_axis_label='Gamma-ray (API)')
    p.line(np.arange(1,depth.shape[0]+1,1),eval_dat[:,0], color='navy', alpha=0.5)
    q = figure(title="Density log after preprocessing", plot_width=800, plot_height=250, x_axis_label='Sample position', y_axis_label='Density (g/cc)')
    q.line(np.arange(1,depth.shape[0]+1,1),eval_dat[:,1],color='navy',alpha=0.5)
    r = figure(title="Gamma-ray log after preprocessing", plot_width=800, plot_height=250, x_axis_label='Depth', y_axis_label='Gamma-ray (API)')
    r.line(depth[:,0],eval_dat[:,0],color='navy',alpha=0.5)
    s = figure(title="Density log after preprocessing", plot_width=800, plot_height=250,x_axis_label='Depth',y_axis_label='Density (g/cc)')
    s.line(depth[:,0], eval_dat[:,1],color='navy',alpha=0.5)
    t = column(p,q,r,s)
    return t

def create_figure11(eval_dat,depth,eval_output,a1,a2,DTC):
    p = figure(title="Gamma-ray log after preprocessing", plot_height=800, plot_width=250, y_axis_label='Depth',x_axis_label='Gamma-ray (API)')
    p.line(eval_dat[:,0], depth[:,0], color='navy', alpha=0.5)
    p.y_range.flipped = True
    q = figure(title="Density log after preprocessing", plot_height=800, plot_width=250, y_axis_label='Depth',x_axis_label='Density (g/cc)')
    q.line(eval_dat[:,1], depth[:,0], color='navy', alpha=0.5)
    q.y_range.flipped = True
    r = figure(title="Predicted sonic log", plot_height=800, plot_width=250, y_axis_label='Depth',x_axis_label='Sonic (us/ft)')
    r.line(eval_output[:,0], depth[:,0], legend='Predicted sonic log', color='navy', alpha=0.5)
    r.line(a1[:,0], depth[:,0], color='red', legend='Lower bound', alpha=0.5)
    r.line(a2[:,0], depth[:,0], color='green', legend='Upper bound', alpha=0.5)
    r.line(DTC[:,0], depth[:,0], color='black',legend='True sonic log',alpha=0.5)
    r.y_range.flipped = True
    s = figure(title="Density-Sonic crossplot", x_axis_label='Predicted sonic log',y_axis_label='Density log',x_range=(min(eval_output[:,0]),max(eval_output[:,0])), y_range=(min(eval_dat[:,1]),max(eval_dat[:,1])), toolbar_location=None)
    #Use the field name of the column source
    z = eval_output[:,0]
    mapper = linear_cmap(field_name='z', palette="Viridis256" ,low=min(z) ,high=max(z))
    source1 = ColumnDataSource(dict(x=eval_output[:,0],y=eval_dat[:,1],z=eval_output[:,0]))
    source2 = ColumnDataSource(dict(x=eval_output[:,0],y=eval_dat[:,0],z=eval_output[:,0]))
    source3 = ColumnDataSource(dict(x=eval_output[:,0],y=DTC[:,0],z=eval_output[:,0]))
    s.scatter(x='x', y='y', marker="circle", color=mapper, source=source1)
    t = figure(title="GR-Sonic crossplot", x_axis_label='Predicted sonic log', y_axis_label='GR log',x_range=(min(eval_output[:,0]),max(eval_output[:,0])), y_range=(min(eval_dat[:,0]),max(eval_dat[:,0])), toolbar_location=None)
    t.scatter(x='x', y='y', marker="circle", color=mapper, source=source2)
    u = figure(title="TrueSonic-PredictedSonic crossplot", x_axis_label='Predicted sonic log',y_axis_label='True sonic log',x_range=(min(eval_output[:,0]),max(eval_output[:,0])), y_range=(min(DTC[:,0]),max(DTC[:,0])), toolbar_location=None)
    u.scatter(x='x',y='y',marker="circle",color=mapper,source=source3)
    color_bar = ColorBar(color_mapper=mapper['transform'], ticker=LogTicker(), label_standoff=12, border_line_color=None, location=(0,0))
    s.add_layout(color_bar, 'right')
    t.add_layout(color_bar, 'right')
    u.add_layout(color_bar, 'right')
    v = column(row(p,q,r),s,t,u)
    return v


def create_figure1(eval_dat,depth,eval_output,a1,a2):
    p = figure(title="Gamma-ray log after preprocessing", plot_height=800, plot_width=250, y_axis_label='Depth',x_axis_label='Gamma-ray (API)')
    p.line(eval_dat[:,0], depth[:,0], color='navy', alpha=0.5)
    p.y_range.flipped = True
    q = figure(title="Density log after preprocessing", plot_height=800, plot_width=250, y_axis_label='Depth',x_axis_label='Density (g/cc)')
    q.line(eval_dat[:,1], depth[:,0], color='navy', alpha=0.5)
    q.y_range.flipped = True
    r = figure(title="Predicted sonic log", plot_height=800, plot_width=250, y_axis_label='Depth',x_axis_label='Sonic (us/ft)')
    r.line(eval_output[:,0], depth[:,0], legend='Predicted sonic log', color='navy', alpha=0.5)
    r.line(a1[:,0], depth[:,0], color='red', legend='Lower bound', alpha=0.5)
    r.line(a2[:,0], depth[:,0], color='green', legend='Upper bound', alpha=0.5)
    r.y_range.flipped = True
    s = figure(title="Density-Sonic crossplot", x_range=(min(eval_output[:,0]),max(eval_output[:,0])), x_axis_label='Predicted sonic log',y_axis_label='Density log',y_range=(min(eval_dat[:,1]),max(eval_dat[:,1])), toolbar_location=None)
    #Use the field name of the column source
    z = eval_output[:,0]
    mapper = linear_cmap(field_name='z', palette="Viridis256" ,low=min(z) ,high=max(z))
    source1 = ColumnDataSource(dict(x=eval_output[:,0],y=eval_dat[:,1],z=eval_output[:,0]))
    source2 = ColumnDataSource(dict(x=eval_output[:,0],y=eval_dat[:,0],z=eval_output[:,0]))
    source3 = ColumnDataSource(dict(x=eval_dat[:,0],y=eval_dat[:,1],z=eval_output[:,0]))
    s.scatter(x='x', y='y', marker="circle", color=mapper, source=source1)
    t = figure(title="GR-Sonic crossplot", x_range=(min(eval_output[:,0]),max(eval_output[:,0])), y_range=(min(eval_dat[:,0]),max(eval_dat[:,0])), toolbar_location=None,x_axis_label='Predicted sonic log',y_axis_label='GR log')
    t.scatter(x='x', y='y', marker="circle", color=mapper, source=source2)
    u = figure(title="Density-GR crossplot", x_range=(min(eval_dat[:,0]),max(eval_dat[:,0])), y_range=(min(eval_dat[:,1]),max(eval_dat[:,1])), toolbar_location=None,x_axis_label='GR log',y_axis_label='Density log')
    u.scatter(x='x',y='y',marker="circle",color=mapper,source=source3)
    color_bar = ColorBar(color_mapper=mapper['transform'], ticker=LogTicker(), label_standoff=12, border_line_color=None, location=(0,0))
    s.add_layout(color_bar, 'right')
    t.add_layout(color_bar, 'right')
    u.add_layout(color_bar, 'right')
    v = column(row(p,q,r),s,t,u)
    return v

def ticker():
    return "{:.2f}".format(tick)
def ticker1():
    return "{:.0f}".format(tick)

# Create the main plot
def create_figure(result, var, test, position, number):
    color_mapper1 = LinearColorMapper(palette="Viridis256",low=np.amin(result),high=np.amax(result))
    color_mapper2 = LinearColorMapper(palette="Greys256",low=np.amin(var),high=np.amax(var))
    color_mapper3 = LinearColorMapper(palette="Greys256",low=np.amin(test),high=np.amax(test))
    p = figure(title="Channel Probability", x_range=(0, 5), y_range=(0, 5), toolbar_location=None)
    q = figure(title="Seismic Amplitude", x_range=(0, 5), y_range=(0, 5), toolbar_location=None)
    r = figure(title="Model Uncertainty", x_range=(0, 5), y_range=(0, 5), toolbar_location=None)
    #s = figure(title="Model Uncertainty Max Vote", x_range=(0,10), y_range=(0,10), toolbar_location=None)
    i = number
    color_bar1 = ColorBar(color_mapper=color_mapper1, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0,0),formatter=FuncTickFormatter.from_py_func(ticker))
    p.add_layout(color_bar1, 'right')
    color_bar2 = ColorBar(color_mapper=color_mapper2, ticker=BasicTicker(),label_standoff=12, border_line_color=None, location=(0,0),formatter=FuncTickFormatter.from_py_func(ticker))
    color_bar3 = ColorBar(color_mapper=color_mapper3, ticker=BasicTicker(), label_standoff=20, border_line_color=None, location=(0,0),formatter=FuncTickFormatter.from_py_func(ticker1))
    q.add_layout(color_bar3, 'right')
    r.add_layout(color_bar2, 'right')
    #s.add_layout(color_bar2, 'right')

    if position==1:
        # must give a vector of image data for image parameter
        p.image(image=[np.transpose(result[i,:,:])], x=0, y=0, dw=5, dh=5, color_mapper=color_mapper1)
        # must give a vector of image data for image parameter
        q.image(image=[np.transpose(test[i,:,:])], x=0, y=0, dw=5, dh=5, color_mapper=color_mapper3)
        r.image(image=[np.transpose(var[i,:,:])], x=0, y=0, dw=5, dh=5, color_mapper=color_mapper2)
        #s.image(image=[np.transpose(var2[i,:,:])], x=0, y=0, dw=10, dh=10, color_mapper=color_mapper2)
    elif position==2:
        # must give a vector of image data for image parameter
        p.image(image=[np.transpose(result[:,i,:])], x=0, y=0, dw=5, dh=5, color_mapper=color_mapper1)
        # must give a vector of image data for image parameter
        q.image(image=[np.transpose(test[:,i,:])], x=0, y=0, dw=5, dh=5,color_mapper=color_mapper3)
        r.image(image=[np.transpose(var[:,i,:])], x=0, y=0, dw=5, dh=5,color_mapper=color_mapper2)
        #s.image(image=[np.transpose(var2[:,i,:])], x=0, y=0, dw=10, dh=10,color_mapper=color_mapper2)
    else:
        # must give a vector of image data for image parameter
        p.image(image=[np.transpose(result[:,:,i])], x=0, y=0, dw=5, dh=5,color_mapper=color_mapper1)
        # must give a vector of image data for image parameter
        q.image(image=[np.transpose(test[:,:,i])], x=0, y=0, dw=5, dh=5,color_mapper=color_mapper3)
        r.image(image=[np.transpose(var[:,:,i])], x=0, y=0, dw=5, dh=5,color_mapper=color_mapper2)
        #s.image(image=[np.transpose(var2[:,:,i])], x=0, y=0, dw=10, dh=10,color_mapper=color_mapper2)
        
    t = column(row(p, r),q)
    return t

def MAX_VOTE(pred,prob,NUM_CLASS):
    """
    logit: the shape should be [NUM_SAMPLES,Batch_size, image_h,image_w,NUM_CLASS]
    pred: the shape should be[NUM_SAMPLES,NUM_PIXELS]
    label: the real label information for each image
    prob: the probability, the shape should be [NUM_SAMPLES,image_h,image_w,NUM_CLASS]
    Output:
    logit: which will feed into the Normal loss function to calculate loss and also accuracy!
    """

    image_h = 156
    image_w = 156
    image_d = 100
    NUM_SAMPLES = np.shape(pred)[0]
    #transpose the prediction to be [NUM_PIXELS,NUM_SAMPLES]
    pred_tot = np.transpose(pred)
    prob_re = np.reshape(prob,[NUM_SAMPLES,image_h*image_w*image_d,NUM_CLASS])
    prediction = []
    variance_final = []
    step = 0
    for i in pred_tot:
        
        value = np.bincount(i,minlength = NUM_CLASS)
        value_max = np.argmax(value)
        #indices = [k for k,j in enumerate(i) if j == value_max]
        indices = np.where(i == value_max)[0]
        prediction.append(value_max)
        variance_final.append(np.var(prob_re[indices,step,:],axis = 0))
        step = step+1
        
     
    return variance_final,prediction

def var_calculate(pred,prob_variance):
    """
    Inputs: 
    pred: predicted label, shape is [NUM_PIXEL,1]
    prob_variance: the total variance for 12 classes wrt each pixel, prob_variance shape [image_h,image_w,12]
    Output:
    var_one: corresponding variance in terms of the "optimal" label
    """
        
    image_h = 156
    image_w = 156
    image_d = 100
    NUM_CLASS = np.shape(prob_variance)[-1]
    var_sep = [] #var_sep is the corresponding variance if this pixel choose label k
    length_cur = 0 #length_cur represent how many pixels has been read for one images
    for row in np.reshape(prob_variance,[image_h*image_w*image_d,NUM_CLASS]):
        temp = row[pred[length_cur]]
        length_cur += 1
        var_sep.append(temp)
    var_one = np.reshape(var_sep,[image_h,image_w,image_d]) #var_one is the corresponding variance in terms of the "optimal" label
    
    return var_one

from wtforms import Form, StringField, IntegerField, validators
class InputForm(Form):
    Position = IntegerField(default=3, validators=[validators.InputRequired(),validators.NumberRange(min=1, max=3)], description=' (Input slice position you want: 1-Horizontal, 2-Vertical, 3-Depth.)')
    Number = IntegerField(default=1, validators=[validators.InputRequired(),validators.NumberRange(min=1)], description=' (Input slice number you want: The value should be from 1 to the maximum size of the slice position.)')

class InputForm1(Form):
    InlineKeyNumber = IntegerField(default=189, validators=[validators.InputRequired(), validators.NumberRange(min=1)], description=' (Input inline key number of the input seismic.)')
    XlineKeyNumber = IntegerField(default=193, validators=[validators.InputRequired(), validators.NumberRange(min=1)], description=' (Input crossline key number of the input seismic.)')
    NumSamp = IntegerField(default=10, validators=[validators.InputRequired(), validators.NumberRange(min=1)], description=' (Input number of samples to generate DL model uncertainty.)')

class InputForm2(Form):
    TopSamplePosition = IntegerField(default=1, validators=[validators.InputRequired(), validators.NumberRange(min=1)], description=' (Input the position of starting sample of log interval to predict. Log samples position start from 1.)')
    LastSamplePosition = IntegerField(default=61, validators=[validators.InputRequired()], description=' (Input the position of last sample of log interval to predict. Make sure that the length of interval is a multiple of 61. Interval length = LastSamplePosition - TopSamplePosition + 1).')
    NumSamp = IntegerField(default=10, validators=[validators.InputRequired(), validators.NumberRange(min=1)], description=' (Input number of samples to generate DL model uncertainty.)')

class InputForm3(Form):
    GRLogName = StringField(default=str('GR'), validators=[validators.InputRequired()], description=' (Input name of Gamma-ray log in the LAS file.)')
    DENSLogName = StringField(default=str('DENS'), validators=[validators.InputRequired()], description=' (Input name of Density log in the LAS file.)')
    DTCLogName = StringField(default=str('No'), validators=[validators.InputRequired()], description=' (Input name of Sonic log in the LAS file if it is available for comparison.)')

@app.route('/')
def home():
    #os.system('mkdir static/%s'%(user))
    return render_template("home.html", my_image='static/nam.jpg')

slice_names = ['Horizontal','Vertical','Depth']

@app.route('/about')
def about():
    return render_template("about.html", group_image='static/Group.jpg')

@app.route('/seis',methods=['POST','GET'])
def seis():
    form1 = InputForm1(request.form)
    if request.method == 'POST' and form1.validate():
        np.savez_compressed('static/QC.npz',form1.InlineKeyNumber.data,form1.XlineKeyNumber.data, form1.NumSamp.data)
        #np.savez_compressed('static/%s/numsamp.npz'%(user),form1.NumSamp.data)
        return render_template("channel_index3.html")
    return render_template("channel_indexQC.html", form=form1)

@app.route('/log',methods=['POST','GET'])
def log():
    form3 = InputForm3(request.form)
    if request.method == 'POST' and form3.validate():
        file1 = open("static/logGR.txt","w")    
        file1.write(str(form3.GRLogName.data))  
        file1.close() #to change file access modes 
        file2 = open("static/logDENS.txt","w")
        file2.write(str(form3.DENSLogName.data))
        file2.close()
        file3 = open("static/logDTC.txt","w")
        file3.write(str(form3.DTCLogName.data))
        file3.close()
        return render_template("log_index3.html")
    return render_template("log_indexQC.html", form=form3)

@app.route('/log/result',methods=['POST','GET'])
def logresult():
    form2 = InputForm2(request.form)
    script2 = None
    div2 = None
    if request.method == 'POST' and form2.validate()==False:
        myfileGR = open("static/logGR.txt","r")
        GRLogName = ""
        linesGR = myfileGR.readlines()
        for lineGR in linesGR:
            GRLogName = GRLogName + lineGR.strip()
        myfileDENS = open("static/logDENS.txt","r")
        DENSLogName = ""
        linesDENS = myfileDENS.readlines()
        for lineDENS in linesDENS:
            DENSLogName = DENSLogName + lineDENS.strip()
        myfileDTC = open("static/logDTC.txt","r")
        DTCLogName = ""
        linesDTC = myfileDTC.readlines()
        for lineDTC in linesDTC:
            DTCLogName = DTCLogName + lineDTC.strip()
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        las = lasio.read(os.path.join(UPLOAD_FOLDER, filename))
        data = las.df()
        GR_mask = data[str(GRLogName)].notnull().values
        GR_mask_pos = data[str(GRLogName)]>=0
        DENS_mask = data[str(DENSLogName)].notnull().values
        DENS_mask_pos = data[str(DENSLogName)]>=0
        mask = [all(tup) for tup in zip(GR_mask, DENS_mask, GR_mask_pos, DENS_mask_pos)]
        GR = data[str(GRLogName)][mask].values
        if str(DTCLogName) != str('No'):
            DTC = data[str(DTCLogName)][mask].values
            DTC = np.reshape(DTC,(DTC.shape[0],1))
            np.savez_compressed('static/eval_DTC.npz',DTC)
        DEPTH = data.index[mask]    
        GR = np.reshape(GR,(GR.shape[0],1))
        DENS = data[str(DENSLogName)][mask].values # Eliminate where GR, DENS, DTC, and NEUT is NaN
        DENS = np.reshape(DENS,(DENS.shape[0],1))
        DEPTH = np.reshape(DEPTH,(DENS.shape[0],1))
        eval_dat = np.concatenate((GR,DENS),axis=1) # Every example will have shape (GR.shape[0],3)
        #mediandat = array([62.125 ,  2.3658])
        #quantdat = array([29.6208,  0.2847])
        #eval_dat = (eval_dat-mediandat)/quantdat
        np.savez_compressed('static/eval_dat.npz',eval_dat)
        np.savez_compressed('static/eval_depth.npz', DEPTH)
        plot2 = create_figure2(eval_dat, DEPTH)
        script2, div2 = components(plot2)
    if request.method=="POST" and form2.validate():
        start = form2.TopSamplePosition.data - 1
        end = form2.LastSamplePosition.data - 1
        eval_dat2 = np.load('static/eval_dat.npz')['arr_0']
        DEPTH2 = np.load('static/eval_depth.npz')['arr_0']
        mediandat = np.array([62.125 ,  2.3658])
        quantdat = np.array([29.6208,  0.2847])
        eval_dat1 = (eval_dat2 - mediandat)/quantdat
        eval_dat1 = eval_dat1[start:end+1,:]
        eval_dat1 = np.reshape(eval_dat1,(-1,61,2))
        evaldat = np.pad(eval_dat1,((0,0),(10,10),(0,0)), 'constant',constant_values=(0,0))
        eval_logs = np.zeros((evaldat.shape[0],evaldat.shape[1]-20,21,2))
        for i in range (10,evaldat.shape[1]-10,1):
            eval_logs[:,i-10,:,:] = np.concatenate((evaldat[:,i-10:i,:],evaldat[:,i:i+11,:]),axis=1)
        eval_logs = np.transpose(eval_logs,(0,2,1,3))
        graph2 = app.graph2
        y_pred = graph2.get_tensor_by_name("output:0")
        x_tensor = graph2.get_tensor_by_name("X:0")
        keep_prob = graph2.get_tensor_by_name("Placeholder:0")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        sess = tf.compat.v1.Session(graph=graph2, config=config)
        feed_dict_testing={x_tensor:eval_logs, keep_prob:0.7}
        NumSamp = form2.NumSamp.data
        predict_iter_tot = [] #Store each MCMC 
        variance_iter_tot = [] #Store each MCMC 
        for step in range(NumSamp):
            eval_output = sess.run(y_pred, feed_dict=feed_dict_testing)
            eval_output = np.reshape(eval_output,(-1,1))
            predict_iter_tot.append(eval_output) #MCMC Distribution
        pred_mean = np.nanmean(predict_iter_tot,axis = 0)
        pred_variance = np.var(predict_iter_tot, axis = 0)
        pred_mean = np.array(pred_mean)
        pred_variance = np.array(pred_variance)
        medianresult = np.array([98.6806])
        quantresult = np.array([34.4058])
        eval_output = (pred_mean*quantresult)+medianresult
        a1=(pred_mean-1.645*np.sqrt(pred_variance))*quantresult+medianresult
        a2=(pred_mean+1.645*np.sqrt(pred_variance))*quantresult+medianresult
        myfileDTC = open("static/logDTC.txt","r")
        DTCLogName = ""
        linesDTC = myfileDTC.readlines()
        for lineDTC in linesDTC:
            DTCLogName = DTCLogName + lineDTC.strip()
        if str(DTCLogName) != str('No'):
            DTC2 = np.load('static/eval_DTC.npz')['arr_0']
            DTC2 = np.reshape(DTC2,(DTC2.shape[0],1))
            DTC2 = np.nan_to_num(DTC2)
            plot1 = create_figure11(eval_dat2[start:end+1,:],DEPTH2[start:end+1,:],eval_output,a1,a2,DTC2[start:end+1,:])
            script2,div2 = components(plot1)
        else:
            plot1 = create_figure1(eval_dat2[start:end+1,:],DEPTH2[start:end+1,:],eval_output,a1,a2)
            script2, div2 = components(plot1)
    return render_template("log_index6.html", form=form2, script=script2, div=div2)

@app.route('/seis/result',methods=['POST','GET'])
def seisresult():
    form = InputForm(request.form)
    script = None
    div = None
    if request.method == 'POST' and form.validate()==False:
        NumSamp = np.load('static/QC.npz')['arr_2']
        InlineKeyNumber = np.load('static/QC.npz')['arr_0']
        XlineKeyNumber = np.load('static/QC.npz')['arr_1']
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image_height=156
        image_width=156
        image_depth=100
        num_channels=1
        images = []
        # Reading the seismic using Pickle
        #with open(os.path.join(UPLOAD_FOLDER, filename), 'rb') as f:
            #test = pickle.load(f, encoding='bytes')
        #test=np.array(test)
        try:
            seis = segyio.open(os.path.join(UPLOAD_FOLDER, filename), iline=int(InlineKeyNumber), xline=int(XlineKeyNumber))
            seismic = segyio.cube(seis)
        except:
            seismic = np.load(os.path.join(UPLOAD_FOLDER, filename))
        #np.savez_compressed('static/seishape.npz',seismic.shape[0],seismic.shape[1],seismic.shape[2])
        seismic.tofile('static/seismic.asc',sep=" ")
        os.system('echo in=static/seismic.asc n1=%d n2=%d n3=%d data_format=ascii_float | sfdd form=native | sfpatch w=100,156,156>static/seismicaf.rsf' % (seismic.shape[2], seismic.shape[1], seismic.shape[0]))
        e = m8r.File('static/seismicaf.rsf')
        c = e[:]

        test = c.reshape(-1,156, 156, 100)
        #print('Size of seismic volume is: %s' % str(test.shape))
            

        m=-6.40475426972431e-05
        s=0.006666915856214509
        test = (test-m)/s

        graph =app.graph
        ## NOW the complete graph with values has been restored
        y_pred = tf.nn.softmax(graph.get_tensor_by_name("Classifier/logits:0"))
        ## Let's feed the images to the input placeholders
        #using the model for prediction
        x_tensor = graph.get_tensor_by_name("Input/Placeholder:0")
        #keep_prob is not always necessary it depends on your model
        keep_prob = graph.get_tensor_by_name("Input/Placeholder_2:0")
        is_training = graph.get_tensor_by_name("Input/Placeholder_3:0")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess= tf.compat.v1.Session(graph=graph,config=config)
        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x_tensor: test,  keep_prob: 0.7, is_training:True}
            
        prob_iter_tot = []
        pred_iter_tot = []
        prob_variance_tot = []
        for i in range (NumSamp):
            result=sess.run(y_pred, feed_dict=feed_dict_testing)
            prob_iter_tot.append(result)
            #pred_iter_tot.append(np.reshape(np.argmax(result,axis = -1),[result.shape[0],-1]))
            
        #pred_iter_tot = np.array(pred_iter_tot)
        #pred_iter_tot = np.reshape(pred_iter_tot,(NumSamp,result.shape[0],-1))
        #pred_iter_tot = np.transpose(pred_iter_tot,(1,0,2))
        prob_iter_tot = np.array(prob_iter_tot)
        prob_iter_tot = np.reshape(prob_iter_tot,(NumSamp,test.shape[0],156,156,100,2))
        #prob_iter_tot_transp = np.transpose(prob_iter_tot,(1,0,2,3,4,5))

        #for k in range(test.shape[0]):
            #prob_variance,pred = MAX_VOTE(pred_iter_tot[k,:,:], prob_iter_tot_transp[k,:,:,:,:,:],2)
            #prob_variance_tot.append(prob_variance)

        prob = np.nanmean(prob_iter_tot, axis=0)
        var = np.nanvar(prob_iter_tot, axis=0)
        #prob_variance_tot = np.array(prob_variance_tot)
        #prob_variance_tot = np.reshape(prob_variance_tot,(test.shape[0],156,156,100,2))

        test = np.reshape(test,(c.shape[0],c.shape[1],c.shape[2],156,156,100))
        result1 = np.reshape(prob[:,:,:,:,1],(c.shape[0],c.shape[1],c.shape[2],156,156,100))
        var1 = np.reshape(var[:,:,:,:,1],(c.shape[0],c.shape[1],c.shape[2],156,156,100))
        #var2 = np.reshape(prob_variance_tot[:,:,:,:,1],(c.shape[0],c.shape[2],156,156,100))
        var1.tofile('static/var1.asc', sep=" ")
        #var2.tofile('static/var2.asc', sep=" ")
        result1.tofile('static/result1.asc',sep=" ")
        test.tofile('static/test.asc',sep=" ")
            

        os.system('echo in=static/test.asc n1=100 n2=156 n3=156 n4=%d n5=%d n6=%d data_format=ascii_float | sfdd form=native | sfpatch inv=y weight=y n0=%d,%d,%d>static/test.rsf' % (c.shape[2],c.shape[1],c.shape[0],seismic.shape[2],seismic.shape[1],seismic.shape[0]))
        f = m8r.File('static/test.rsf')
        a = f[:]
        a = np.reshape(a,(seismic.shape[0],seismic.shape[1],seismic.shape[2]))
        os.system('echo in=static/result1.asc n1=100 n2=156 n3=156 n4=%d n5=%d n6=%d data_format=ascii_float | sfdd form=native | sfpatch inv=y weight=y n0=%d,%d,%d>static/result1.rsf' % (c.shape[2],c.shape[1],c.shape[0],seismic.shape[2],seismic.shape[1],seismic.shape[0]))
        g = m8r.File('static/result1.rsf')
        b = g[:]
        b = np.reshape(b,(seismic.shape[0],seismic.shape[1],seismic.shape[2]))
        os.system('echo in=static/var1.asc n1=100 n2=156 n3=156 n4=%d n5=%d n6=%d data_format=ascii_float | sfdd form=native | sfpatch inv=y weight=y n0=%d,%d,%d>static/var1.rsf' % (c.shape[2],c.shape[1],c.shape[0],seismic.shape[2],seismic.shape[1],seismic.shape[0]))
        h = m8r.File('static/var1.rsf')
        k = h[:]
        k = np.reshape(k,(seismic.shape[0],seismic.shape[1],seismic.shape[2]))
        #os.system('echo in=static/var2.asc n1=100 n2=156 n3=156 n4=%d n5=%d n6=%d data_format=ascii_float | sfdd form=native | sfpatch inv=y weight=y n0=%d,%d,%d>static/var2.rsf' % (c.shape[2],c.shape[1],c.shape[0],seismic.shape[2],seismic.shape[1],seismic.shape[0]))
        #l = m8r.File('static/var2.rsf')
        #l1 = l[:]
        #l1 = np.reshape(l1,(seismic.shape[0],seismic.shape[1],seismic.shape[2]))

        np.savez_compressed('static/result.npz',b)
        np.savez_compressed('static/test.npz',seismic)
        np.savez_compressed('static/var.npz',k)
        #np.savez_compressed('static/var2.npz',l1)
        #form = InputForm(request.form)
        return render_template("channel_index6.html", form=form, script=script, div=div)
    if request.method=="POST" and form.validate():    
        # Determine the selected slice
        current_slice = form.Position.data 
        # Determine the selected slice number
        current_slice_number = form.Number.data
                
        result = np.load('static/result.npz')['arr_0']
        test = np.load('static/test.npz')['arr_0']
        var = np.load('static/var.npz')['arr_0']
        #var2 = np.load('static/var2.npz')['arr_0']

        #rsfarray_1 = m8r.File(test)
        #rsfarray_2 = m8r.put(d1=0.004, o1=0).patch(inv=True, weight=True, n0=[100,312,312])[rsfarray_1]
        #test = rsfarray_2[:]

        #rsfarray_1 = m8r.File(result1)
        #rsfarray_2 = m8r.put(d1=0.004, o1=0).patch(inv=True, weight=True, n0=[100,312,312])[rsfarray_1]
        #result = rsfarray_2[:]
                


        #print(result.shape)
        #print(test.shape)
        # Create the plot
        plot = create_figure(result, var, test, current_slice, current_slice_number)
        # Embed plot into HTML via Flask Render
        script, div = components(plot)
    return render_template("channel_index6.html", form=form, script=script, div=div)
app.graph=load_graph('static/optimized_model.pb')
app.graph2=load_graph('static/optimized_modellog.pb')
if __name__ == '__main__':
    app.run("0.0.0.0", os.environ.get('PORT',8080),debug=True)
    
