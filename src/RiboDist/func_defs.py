# Copyright 2021 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import sys
import re
from os.path import exists

import numpy as np
import pandas as pd
import starfile as sf

from sklearn.cluster import AgglomerativeClustering
import scipy.interpolate as spin
from scipy.spatial.transform import Rotation as R


def get_ribo_from_star(star_file):
    """
    Function to retrieve information from a given star file

    Args:
    star_file (str) :: Path to star file

    Returns:
    DataFrame, list, float
    """

    pixel_size_nm = sf.read(star_file)['optics'].rlnImagePixelSize.values[0] * 0.1
    ribo_star = sf.read(star_file)
    ribo_star['particles']['rlnTS'] = [int(i.split('/')[1].split('_')[-1]) for i in list(ribo_star['particles'].rlnImageName.values)]
    TS_list = pd.unique(ribo_star['particles'].rlnTS)

    return ribo_star, TS_list, pixel_size_nm


def get_coords(star_df_in, TS, model_bin, star_bin):
    """
    Function to retrieve positions of a subset of particles from given star-obtained DataFrame

    Args:
    star_df_in (DataFrame) :: DataFrame obtained from reading specified star file
    TS (int)               :: Tilt series number
    model_bin (int)        :: Binning factor for model
    star_bin (int)         :: Binning factor for star file

    Returns:
    DataFrame
    """

    ribo = star_df_in[star_df_in.rlnTS==TS][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy() * star_bin / model_bin

    return ribo


def get_model(model_file):
    """
    Function to retrieve model from a text file

    Args:
    model_file (str) :: Path to txt file containing model

    Returns:
    ndarray
    """

    model = np.loadtxt(model_file)
    model = model[model[:,2].argsort()]

    return model


def segment_surfaces(model_in):
    ac = AgglomerativeClustering(n_clusters=2, linkage="single")
    ac.fit(model_in)

    labels = ac.labels_
    model_0 = np.squeeze(model_in[np.argwhere(labels==0)], axis=1)
    model_1 = np.squeeze(model_in[np.argwhere(labels==1)], axis=1)

    plane_centroids_diff = np.mean(model_0[:,1:], axis=0) - np.mean(model_1[:,1:], axis=0)
    centroid_of_planes = 0.5 * (np.mean(model_0[:,1:], axis=0) + np.mean(model_1[:,1:], axis=0))

    rot_ang = np.arctan2(plane_centroids_diff[2], plane_centroids_diff[0])
    rot_vec = R.from_rotvec([0, rot_ang, 0])

    model_intermediate = rot_vec.apply(model_in[:, 1:] - centroid_of_planes)
    model_lower = np.squeeze(model_in[np.argwhere(model_intermediate[:,2]>0)], axis=1)
    model_upper = np.squeeze(model_in[np.argwhere(model_intermediate[:,2]<0)], axis=1)

    return labels, model_lower, model_upper


def interpolator(coords_in, upper_in, lower_in, N=100):
    """
    Function to interpolate surfaces and calculate shortest distance of particles to surfaces

    Args:
    coords_in (ndarray) :: Array containing coordinates of particles
    upper_in (ndarray)  :: Original modelling points in upper surface
    lower_in (ndarray)  :: Original modelling points in lower surface
    N (int)             :: Number of interpolating points

    Returns:
    ndarray, ndarray, ndarray
    """

    x_top = np.linspace(np.min(upper_in[:,0]), np.max(upper_in[:,0]), N)
    y_top = np.linspace(np.min(upper_in[:,1]), np.max(upper_in[:,1]), N)
    XX, YY = np.meshgrid(x_top, y_top)

    x_bot = np.linspace(np.min(lower_in[:,0]), np.max(lower_in[:,0]), N)
    y_bot = np.linspace(np.min(lower_in[:,1]), np.max(lower_in[:,1]), N)
    xx, yy = np.meshgrid(x_bot, y_bot)


    itp_top = spin.LinearNDInterpolator(list(zip(upper_in[:,0], upper_in[:,1])), upper_in[:,2])
    itp_bot = spin.LinearNDInterpolator(list(zip(lower_in[:,0], lower_in[:,1])), lower_in[:,2])
    ZZ = itp_top(XX, YY)
    zz = itp_bot(xx, yy)

    interped_top = np.dstack((XX, YY, ZZ))
    interped_bot = np.dstack((xx, yy, zz))

    top_centre = np.array([XX[N//2, N//2], YY[N//2, N//2], itp_top(XX[N//2], YY[N//2])[N//2]])
    bot_centre = np.array([xx[N//2, N//2], yy[N//2, N//2], itp_bot(xx[N//2], yy[N//2])[N//2]])
    thickness = np.linalg.norm(top_centre-bot_centre, axis=0)

    to_edge = np.empty((len(coords_in), 2))
    for idx, point in enumerate(coords_in):
        to_edge[idx] = [np.nanmin(np.linalg.norm(interped_top - point, axis=2)), np.nanmin(np.linalg.norm(interped_bot - point, axis=2))]

    return interped_top, interped_bot, to_edge, thickness
