#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:48:33 2019

@author: haofang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



def setup_points(filename,PL=10000,LNG=-140,plot=True):
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


def point_dist(x,y,c_x,c_y):
    return np.sqrt((x-c_x)**2+(y-c_y)**2)


def find_nearest_centroid(x,y,old_c_x,old_c_y):
    dists=[]
    for v in range(len(old_c_x)):
        dist = point_dist(x,y,old_c_x[v],old_c_y[v])
        dists.append(dist)
    return np.argmin(dists),min(dists)
    

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
    
    
#def init_centroid(x,y,n_UVA):
#    x_split = split_list(x,n_UVA)
#    y_split = split_list(y,n_UVA)
#    old_c_x=[]
#    old_c_y=[]
#    for v in range(len(x_split)):
#        old_c_x.append(np.random.uniform(low=np.min(x_split[v]), high=np.max(x_split[v])))
#        old_c_y.append(np.random.uniform(low=np.min(y_split[v]), high=np.max(y_split[v])))
#    return old_c_x,old_c_y
def init_centroid(x,y,n_UVA):
    length = len(x)
    array = np.arange(length)
    np.random.shuffle(array)
    index = array[:n_UVA]
    old_c_x = x[index]
    old_c_y = y[index]
    return old_c_x,old_c_y
    

def k_means(data,n_UVA,iteration=100):
    x = data.lng.values
    y = data.lat.values
    old_c_x,old_c_y= init_centroid(x,y,n_UVA)   
    #old_c_x = np.random.uniform(low=np.min(x), high=np.max(x), size=(n_UVA,))
    init_c_x=old_c_x 
    #old_c_y = np.random.uniform(low=np.min(y), high=np.max(y), size=(n_UVA,))
    init_c_y=old_c_y
    all_dist=[]
    for trial in range(iteration):
        labels=[]
        for v in range(len(x)):
            n,dist = find_nearest_centroid(x[v],y[v],old_c_x,old_c_y)
            labels.append(n)
            if trial==iteration-1:
                all_dist.append(dist)
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
            

def optimal_cluster(data,n_UVA,trials=100,kmean_itr=100):
    opt_c_x,opt_c_y,opt_labels,opt_init_cx,opt_init_cy,opt_sum=k_means(data,n_UVA)
    for trial in range(trials):
        if trial%10==0:
            print("In processing. Trial: ",trial)
        new_c_x,new_c_y,new_labels,new_init_cx,new_init_cy,new_sum=k_means(data,n_UVA,iteration=kmean_itr)
        if new_sum<opt_sum:
            opt_c_x = new_c_x
            opt_c_y = new_c_y
            opt_labels = new_labels
            opt_init_cx = new_init_cx
            opt_init_cy = new_init_cy
            opt_sum = new_sum
    return opt_c_x,opt_c_y,opt_labels,opt_init_cx,opt_init_cy,opt_sum
        
    
       
def main(file_dir,n_UVA,kmean_trial=100,kmean_itr=100,plot_clustered=True,plot_unclustered=True,plot_init_centroid=False):
    data = setup_points(file_dir,plot=plot_unclustered)
    start_time=time.time()
    centroid_x,centroid_y,labels,init_c_x,init_c_y,sum_dist = optimal_cluster(data,n_UVA,trials=kmean_trial,kmean_itr=kmean_itr) # trials=100
    end_time=time.time()
    if plot_clustered is True:
        x = data.lng.values
        y = data.lat.values
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
    return centroid_x,centroid_y,labels,init_c_x,init_c_y,sum_dist,runtime
            
if __name__== "__main__":
    file_dir =  "/Users/haofang/Desktop/me574/project/uscitiesv1.4.csv"
    centroid_x,centroid_y,labels,_,_,_,runtime=main(file_dir,2,kmean_trial=10,plot_init_centroid=False)
    print("Runtime: ", runtime)
        


    
            









