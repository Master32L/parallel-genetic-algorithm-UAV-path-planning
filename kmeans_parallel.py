#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:40:32 2019

@author: haofang
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import cuda



def setup_points(filename,PL=100000,LNG=-140,plot=True):
    df = pd.read_csv(filename)
    select = df.loc[(df.population>PL)&(df.lng>LNG),['city', 'lng','lat']]
    select.lng = abs(select.lng)
    if plot is True: 
        plt.figure(1)
        plt.plot(select.lng,select.lat,"o")
        plt.xlabel("Longitude")
        plt.ylabel("Latituede")
        plt.title("US cities")
    plt.show
    return select

@cuda.jit(device = True)
def point_dist(x,y,c_x,c_y):
    return ((x-c_x)**2+(y-c_y)**2)**0.5




@cuda.jit()
def point_dist_kernel(d_out,d_x,d_y,d_cx,d_cy):
    i,j = cuda.grid(2)
    m = d_x.size # get size of input arrays
    n = d_cx.size
    if i<m and j<n:
        d_out[i,j] = point_dist(d_x[i],d_y[i],d_cx[j],d_cy[j])
        

def find_nearest_centroid(x,y,old_c_x,old_c_y,TPBX,TPBY):
    m, n= x.size, len(old_c_x)
    BPGX, BPGY = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY
    d_cx  = cuda.to_device(old_c_x)
    d_cy = cuda.to_device(old_c_y)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array([m,n])
    blocks = (BPGX,BPGY)
    threads = (TPBX,TPBY)
    point_dist_kernel[threads, blocks](d_out,d_x,d_y,d_cx,d_cy)
    return d_out.copy_to_host()

#@cuda.jit()
#def point_dist_kernel(d_out,d_x,d_y,d_cx,d_cy):
#    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
#    if i < d_cx.size:
#        d_out[i]=point_dist(d_x,d_y,d_cx[i],d_cy[i])
    



#def find_nearest_centroid(x,y,old_c_x,old_c_y):
#    size = len(old_c_x)
#    BPG = (size + TPB - 1)//TPB
#    d_cx  = cuda.to_device(old_c_x)
#    d_cy = cuda.to_device(old_c_y)
#    d_out = cuda.device_array(size)
#    point_dist_kernel[BPG, TPB](d_out,x,y,d_cx,d_cy)
#    return d_out.copy_to_host()
    

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
    
    
def init_centroid(x,y,n_UVA):
    length = len(x)
    array = np.arange(length)
    np.random.shuffle(array)
    index = array[:n_UVA]
    old_c_x = x[index]
    old_c_y = y[index]
    return old_c_x,old_c_y
    

def k_means(data,n_UVA,TPBX,TPBY,iteration=100):
    x = data.lng.values
    y = data.lat.values
    old_c_x,old_c_y= init_centroid(x,y,n_UVA)   
    #old_c_x = np.random.uniform(low=np.min(x), high=np.max(x), size=(n_UVA,))
    init_c_x=old_c_x 
    #old_c_y = np.random.uniform(low=np.min(y), high=np.max(y), size=(n_UVA,))
    init_c_y=old_c_y
    all_dist=[]
    for trial in range(iteration):
        d_out=find_nearest_centroid(x,y,old_c_x,old_c_y,TPBX,TPBY)
        dists=d_out.min(axis=1)
        labels = np.argmin(d_out, axis=1)
        if trial==iteration-1:
            all_dist=dists
        old_c_x=[]
        old_c_y=[]
        for i in range(n_UVA):
            these_x = [x[p] for p, value in enumerate(labels) if value == i]
            these_y =[y[p] for p, value in enumerate(labels) if value == i]
            c_x=np.mean(these_x)
            old_c_x.append(c_x)
            c_y = np.mean(these_y)
            old_c_y.append(c_y)
            
    return old_c_x,old_c_y,labels,init_c_x,init_c_y,np.sum(all_dist)
            

def optimal_cluster(data,n_UVA,TPBX,TPBY,trials=100,kmean_itr=100):
    opt_c_x,opt_c_y,opt_labels,opt_init_cx,opt_init_cy,opt_sum=k_means(data,n_UVA,TPBX,TPBY,iteration=kmean_itr)
    for trial in range(trials):
        if (trial+1)%10==0:
            print("In processing. Trial: ",trial+1)
        new_c_x,new_c_y,new_labels,new_init_cx,new_init_cy,new_sum=k_means(data,n_UVA,TPBX,TPBY,iteration=kmean_itr)
        if new_sum<opt_sum:
            opt_c_x = new_c_x
            opt_c_y = new_c_y
            opt_labels = new_labels
            opt_init_cx = new_init_cx
            opt_init_cy = new_init_cy
            opt_sum = new_sum
    return opt_c_x,opt_c_y,opt_labels,opt_init_cx,opt_init_cy,opt_sum
        
    
       
def main(file_dir,n_UVA,kmean_trial=100,kmean_itr=100,TPBX=64,TPBY=64,PL=100000,plot_clustered=True,plot_unclustered=True,plot_init_centroid=False):
    data = setup_points(file_dir,PL=PL,plot=plot_unclustered)
    start_time=time.time()
    centroid_x,centroid_y,labels,init_c_x,init_c_y,sum_dist = optimal_cluster(data,n_UVA,TPBX,TPBY,trials=kmean_trial,kmean_itr=kmean_itr) # trials=100
    end_time=time.time()
    x = data.lng.values
    y = data.lat.values
    if plot_clustered is True:
        plt.figure()
        for i in range(n_UVA):
            these_x = [x[p] for p, value in enumerate(labels) if value == i]
            these_y =[y[p] for p, value in enumerate(labels) if value == i]
            plt.plot(these_x,these_y,"o")
        plt.plot(centroid_x,centroid_y,"k+",label="Centroids")
        if plot_init_centroid is True:
            plt.plot(init_c_x,init_c_y,"x")
        plt.xlabel("Longitude")
        plt.ylabel("Latituede")
        plt.title("US cities")
        plt.legend()
        plt.show
    runtime = end_time-start_time
    return centroid_x,centroid_y,labels,x,y,runtime
            
if __name__== "__main__":
    centroid_x,centroid_y,labels,x,y,runtime=main("uscitiesv1.4.csv",10,kmean_trial=10,plot_init_centroid=False)
    print("Runtime: ", runtime)