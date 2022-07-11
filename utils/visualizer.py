import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

def channel_value_to_bar_img(att_y,time_interval=None):
    x_axis_length  = att_y.size(0)
    time_unit = int(x_axis_length/9)
    att_x = np.arange(x_axis_length )
    figure = plt.figure()
    plot=figure.add_subplot(111)
    plot.bar(att_x,att_y.cpu().detach().numpy())
    if time_interval is not None:
        exp_time, ro_time=time_interval.split('-')
        exp_time = int(exp_time) 
        ro_time = int(ro_time) 
        unit_time = exp_time + ro_time 
        max_idx = int(time_unit+ (x_axis_length-time_unit)  * exp_time/unit_time)  
        plot.axvline(x=max_idx,color="red",linestyle='--')
        plot.axvline(x=time_unit,color="orange",linestyle='--')
    figure.canvas.draw()
    bar_img = np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img

def channel_average_time_value_to_bar_img(att_y,time_interval=None):
    x_axis_length  = 9
    time_unit = int(att_y.size(0)/9)
    att_x = np.arange(x_axis_length)
    fig = plt.figure()
    plot = fig.add_subplot(111)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)
    if time_interval is not None:
        x_time, y_time = time_interval.split('-')
        x_time = int(x_time) 
        y_time = int(y_time)
        unit_time = x_time+y_time
        att_y = att_y
        avg_y_list = []
        for i in range(x_axis_length):
            cur_time = att_y[time_unit*i:time_unit*(i+1)].mean()
            avg_y_list.append(cur_time)
        att_y = np.array(avg_y_list)
        plot.bar(att_x, att_y)
        max_idx = int(1 + (x_time/unit_time)*8)
        plot.axvline(x=max_idx,color="red",linestyle='--')
        plot.axvline(x=1,color="orange",linestyle='--')
    fig.canvas.draw()
    bar_img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
    return bar_img#.transpose([2,0,1])


'''
multi plot
'''

def channel_average_time_value_to_line_img_2x2(att_y_list,time_interval_list):
    x_axis_length  = 9
    att_x = np.arange(x_axis_length)
    plt.rcParams["figure.dpi"] =  300.0 #(default: 100.0) #Resolution
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plot_list =[]
    plot_list.append(ax1)
    plot_list.append(ax2)
    plot_list.append(ax3)
    plot_list.append(ax4)
    for i, axes in enumerate(plot_list):
        att_y = att_y_list[i]
        time_interval = time_interval_list[i]
        time_unit = int(att_y.size(0)/9)
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y =att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time=att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y=np.array(avg_y_list)
            axes.bar(att_x,att_y)
            max_idx = int(1 + (x_time/unit_time)*8)   # 1, 7/12 * 8 
            axes.axvline(x=max_idx,color="red",linestyle='--').set_label('exposure time')
            axes.axvline(x=1,color="orange",linestyle='--').set_label('t1')
            axes.xaxis.set_ticks([])
            axes.yaxis.set_ticks([])
    plt.tight_layout()
    figure.canvas.draw()
    bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img

def channel_average_time_value_to_line_img_2x1(att_y_list,time_interval_list):
    x_axis_length  = 9
    att_x = np.arange(x_axis_length)
    plt.rcParams["figure.figsize"] = [6.4,2.4]# (default: [6.4, 4.8])
    plt.rcParams["figure.dpi"] =  300.0 #(default: 100.0) #Resolution
    figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    plot_list =[]
    plot_list.append(ax1)
    plot_list.append(ax2)
    a=0
    for i, axes in enumerate(plot_list):
        att_y = att_y_list[i]
        time_interval = time_interval_list[i]
        time_unit = int(att_y.size(0)/9)
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y =att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time=att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y=np.array(avg_y_list)
            axes.bar(att_x,att_y)
            max_idx = round(1 + (x_time/unit_time)*8)   # 1, 7/12 * 8 
            axes.axvline(x=max_idx,color="red",linestyle='--').set_label('exposure time')
            axes.axvline(x=1,color="orange",linestyle='--').set_label('t1')
            axes.xaxis.set_ticks([])
            axes.yaxis.set_ticks([])
    plt.tight_layout()
    figure.canvas.draw()
    bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img#.transpose([2,0,1]) 

def detail_channel_average_time_value_to_line_img_1x1(att_y_list,time_interval_list):
    x_axis_length  = att_y_list[0].size(0) # 8 + 64 
    att_x = np.arange(x_axis_length)
    plt.rcParams["figure.dpi"] =  300.0 #(default: 100.0) #Resolution
    figure, (ax1) = plt.subplots(nrows=1, ncols=1)
    plot_list =[]
    plot_list.append(ax1)
    a=0
    for i, axes in enumerate(plot_list):
        att_y = att_y_list[i]
        time_interval = time_interval_list[i]
        time_unit = int(att_y.size(0)) # 8
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y =att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            att_y=np.array(att_y)
            axes.bar(att_x,att_y)
            max_idx = round(math.ceil(8 + (x_time/unit_time)*64)/9)*9   # 1, 7/12 * 8 
            axes.axvline(x=max_idx,color="red",linestyle='--').set_label('exposure time')
            axes.axvline(x=1,color="orange",linestyle='--').set_label('t1')
            axes.xaxis.set_ticks([])
            axes.yaxis.set_ticks([])
    plt.tight_layout()
    figure.canvas.draw()
    bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img

def channel_average_time_value_to_line_img_1x1(att_y_list,time_interval_list):
    x_axis_length  = 9
    att_x = np.arange(x_axis_length)
    plt.rcParams["figure.dpi"] =  300.0 #(default: 100.0) #Resolution
    figure, (ax1) = plt.subplots(nrows=1, ncols=1)
    plot_list =[]
    plot_list.append(ax1)
    a=0
    for i, axes in enumerate(plot_list):
        att_y = att_y_list[i]
        time_interval = time_interval_list[i]
        time_unit = int(att_y.size(0)/9) # 8
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y =att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time=att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y=np.array(avg_y_list)
            axes.bar(att_x,att_y)
            max_idx = int(1 + (x_time/unit_time)*8 )   # 1, 7/12 * 8 
            axes.axvline(x=max_idx,color="red",linestyle='--').set_label('exposure time')
            axes.axvline(x=1,color="orange",linestyle='--').set_label('t1')
            axes.xaxis.set_ticks([])
            axes.yaxis.set_ticks([])
    plt.tight_layout()
    figure.canvas.draw()
    bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img#.transpose([2,0,1]) 

def channel_average_time_value_to_line_img_4x1(att_y_list,time_interval_list,smooth=False):
    x_axis_length  = 9
    att_x = np.arange(x_axis_length)
    plt.rcParams["figure.figsize"] = [12.8,3.6]# (default: [6.4, 4.8])
    figure, plot_list = plt.subplots(nrows=1, ncols=4)
    for i, axes in enumerate(plot_list):
        axes.grid(True)
        att_y = att_y_list[i]
        time_interval = time_interval_list[i]
        time_unit = int(att_y.size(0)/9)
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y = att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time = att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y = np.array(avg_y_list)
            if smooth:
                xnew = np.linspace(att_x.min(), att_x.max(), 300) 
                spl = make_interp_spline(att_x, att_y, k=2)  # type: BSpline
                power_smooth = spl(xnew)
                axes.plot(xnew,power_smooth)
            else:
                axes.plot(att_x,att_y)
            max_idx = int(1 + (x_time/unit_time)*8)   # 1, 7/12 * 8 
            axes.axvline(x=max_idx,color="red",linestyle='--').set_label('exposure time')
            axes.axvline(x=1,color="orange",linestyle='--').set_label('t1')
            axes.set_title(time_interval)
            axes.set_xlabel("time unit (channel index)")
            axes.set_ylabel("activation")
    plt.tight_layout()
    figure.canvas.draw()
    bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img

def channel_average_time_value_to_line_img_overlay(att_y_list,time_interval_list,smooth=False):
    x_axis_length  = 9
    att_x = np.arange(x_axis_length)
    figure, axes = plt.subplots(nrows=1, ncols=1)
    for i, time_interval in enumerate(time_interval_list):
        att_y = att_y_list[i]
        time_unit = int(att_y.size(0)/9)
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y =att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time=att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y=np.array(avg_y_list)
            if smooth:
                xnew = np.linspace(att_x.min(), att_x.max(), 300) 
                spl = make_interp_spline(att_x, att_y, k=3)  # type: BSpline
                power_smooth = spl(xnew)
                axes.plot(xnew,power_smooth,label=time_interval)
            else:
                axes.plot(att_x,att_y,label=time_interval)
            max_idx = int(1 + (x_time/unit_time)*8)   # 1, 7/12 * 8 
        axes.set_xlabel("time unit (channel index)")
        axes.set_ylabel("activation")
        axes.legend()
    plt.tight_layout()
    figure.canvas.draw()
    bar_img = np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img



def channel_average_time_value_to_bar_img_2x2(att_y_list,time_interval_list,smooth=False):
    x_axis_length  = 9
    att_x = np.arange(x_axis_length)
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plot_list =[]
    plot_list.append(ax1)
    plot_list.append(ax2)
    plot_list.append(ax3)
    plot_list.append(ax4)
    for i, axes in enumerate(plot_list):
        att_y = att_y_list[i]
        time_interval = time_interval_list[i]
        time_unit = int(att_y.size(0)/9)
        if time_interval is not None:
            x_time, y_time = time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time + y_time # 12
            att_y = att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time=att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y = np.array(avg_y_list)
            if smooth:
                xnew = np.linspace(att_x.min(), att_x.max(), 300) 
                spl = make_interp_spline(att_x, att_y, k=3)  # type: BSpline
                power_smooth = spl(xnew)
                axes.plot(xnew,power_smooth)
            else:
                axes.plot(att_x,att_y)
            max_idx = int(1 + (x_time/unit_time)*8)   # 1, 7/12 * 8 
            axes.axvline(x=max_idx,color="red",linestyle='--').set_label('exposure time')
            axes.axvline(x=1,color="orange",linestyle='--').set_label('t1')
            axes.set_title(time_interval)
            axes.set_xlabel("time unit (channel index)")
            axes.set_ylabel("activation")
    plt.tight_layout()
    figure.canvas.draw()
    bar_img = np.array(figure.canvas.renderer._renderer)[:,:,:3]
    plt.close()
    return bar_img