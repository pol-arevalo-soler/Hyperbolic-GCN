from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np

import torch
from config import parser

from utils.data_utils_v2 import load_data


def train(args):
    # Load data
    data = load_data(args, 'data/')
    print(data)

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)