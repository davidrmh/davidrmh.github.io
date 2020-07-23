#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:20:15 2020

@author: david

Bishop Chapter 8
Image de-noising using
iterated conditional models (ICM)
"""
import imageio
import matplotlib.pyplot as plt
import numpy as np

def get_energy(img_noisy, img_den,
               h, beta, eta,
               row, col):
    '''
    Get's the energy for point x[row, col]
    in both scenarios x[row, col] = 1
    and x[row, col] = -1

    Parameters
    ----------
    img_noisy : 2d numpy array
        noisy image
        
    img_den : 2d numpy array
        Current state of the de-noised image
        
    h : float
    
    beta : positive float
        
    eta : positive float
        
    row : integer
        row of the pixel
        
    col : integer
        column of the pixel

    Returns
    -------
    list
        list[0] energy when x[row, col] = 1
        list[1] energy when x[row, col] = -1
    '''

    #Changes 1 pixel from the de-noised image
    img_den_1 = img_den.copy()
    img_den_1[row, col] = 1.
    
    img_den_m1 = img_den.copy()
    img_den_m1[row, col] = -1.
    
    #Calculates energy for a point
    #energy_1 = h * np.sum(img_den_1) - eta * img_noisy[row, col] * img_den_1[row,col]
    #energy_m1 = h * np.sum(img_den_m1) - eta * img_noisy[row, col] * img_den_m1[row,col]
    
    energy_1 = h * np.sum(img_den_1)\
        - eta * np.sum(img_noisy * img_den_1)
    energy_m1 = h * np.sum(img_den_m1)\
        - eta * np.sum(img_noisy * img_den_m1)
    
    #Calculates the double sum
    #aux stores the neighbors
    #of node (row, col)
    aux = [img_den[row, col - 1], 
           img_den[row, col + 1],
           img_den[row - 1, col],
           img_den[row + 1, col]]
    
    doub_sum_1 = np.sum(aux)
    doub_sum_m1 = -1 * doub_sum_1
    
    energy_1 = energy_1 - beta * doub_sum_1
    energy_m1 = energy_m1 - beta * doub_sum_m1
    
    return [energy_1, energy_m1]     

def ICM(img_noisy, h, beta, eta, n_iter = int(1e3),
        n_points = 178):
    '''
    De-noises a noisy image using
    Iterated conditional modes

    Parameters
    ----------
    img_noisy : 2d numpy array
        noisy image
        
    h : float
        
    beta : positive float
        
    eta : positive float
        
    n_iter : positive integer, optional
        Number of iterations. The default is int(1e3).
        
    n_points : positive integer, optional
        Number of pixels to be updated
        per iteration. The default is 178.

    Returns
    -------
    img_den : 2d numpy array
        De-noised image
    '''
    
    np.random.seed(54321)
    #image shape
    shape = img_noisy.shape
    
    #initializes the de-noised image
    img_den = img_noisy.copy()
    
    #applies iterated conditional modes
    #(coordinate-wise gradient ascent)
    #for a number of iterations
    for n in range(n_iter):
        
        #indexes of the pixels
        #to be updated
        idx_row = np.random.choice(range(1, shape[0] - 1),
                                   size = n_points,
                                   replace = False)
        
        idx_col = np.random.choice(range(1, shape[1] - 1),
                                   size = n_points,
                                   replace = False)
        
        idx_update = zip(idx_row, idx_col)
        
        for row, col in idx_update:
            
            #Calculates the energy
            #for pixel (row, col)
            energy = get_energy(img_noisy,
                                img_den,
                                h, beta, eta, row, col)
            e_1 = energy[0]
            e_m1 = energy[1]
            #Update image
            if e_1 <= e_m1:
                img_den[row, col] = 1.0
            else:
                img_den[row, col] = -1.0
                
    return img_den

def plot_images(img_noisy, img_orig,
                h=0, beta=1, eta=1,
                n_iter = 180*180,
                n_points = 2): 
    
    img_den = ICM(img_noisy, h = h,
                  beta = beta,
                  eta = eta,
                  n_iter = n_iter,
                  n_points = n_points)
    
    #Accuracy
    met_den = 100 * np.mean(img_den == img_orig)
    met_noi = 100 * np.mean(img_noisy == img_orig)
    
    #plot
    plt.clf()
    plt.subplot(131)
    plt.imshow(img_orig, vmin = -1, vmax = 1)
    plt.title(f'Original image')           
    
    plt.subplot(132)
    plt.imshow(img_noisy, vmin = -1, vmax = 1)
    plt.title(f'Noisy image accuracy = {round(met_noi,2)}')
    
    plt.subplot(133)
    plt.imshow(img_den, vmin = -1, vmax = 1)
    plt.title(f'De-noised image accuracy = {round(met_den,2)}')
    
    return img_den   
 
#Original Image
img_orig = imageio.imread('bayes_180_180.png')

#Rescales the values to -1 or 1
#-1 is background
# 1 is drawing
img_orig_mod = -1*np.ones(shape = img_orig[:,:,0].shape)
img_orig_mod[img_orig[:,:,0] < img_orig[:,:,0].max() / 2] = 1

#Adds noise by flipping
#the pixels
np.random.seed(54321)
noise = 0.1
n_nodes = np.product(img_orig_mod.shape)
flips = np.random.choice([1,-1],
                         size = img_orig_mod.shape,
                         p =[1-noise, noise])
img_noisy = flips * img_orig_mod

h = 0.0
beta = 1.
eta = 1.
n_iter = 1000
n_points = 178
img_den = plot_images(img_noisy, img_orig_mod,\
                      h, beta, eta, n_iter, n_points)
