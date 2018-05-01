#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:17:26 2018

@author: sathu
"""

import numpy as np
import pandas as pd
import re


df = pd.DataFrame()
# Loading the dataset

text = pd.read_csv("/Users/sathu/Desktop/Text Classification/mbti_normalized.csv", index_col='type')
#print(text.shape)
#print(text[0:5])
#print(text.iloc[2])


# Function to clean data ... will be useful later
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    print "executing"
    post = post.lower()

    # Remove |||
    post = post.replace('|||', "")

    # Remove URLs, links etc
    post = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        '', post, flags=re.MULTILINE)
    # This would have removed most of the links but probably not all

    # Remove puntuations
    puncs1 = ['@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', '"', "'",
              ';', ':', '<', '>', '/']
    for punc in puncs1:
        post = post.replace(punc, '')

    puncs2 = [',', '.', '?', '!', '\n']
    for punc in puncs2:
        post = post.replace(punc, ' ')
        # Remove extra white spaces
    post = re.sub('\s+', ' ', post).strip()
    return post




# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing.
posts = text.posts.tolist()
print "list made"
posts = [post_cleaner(post) for post in posts]
print "cleaning done"
df = pd.DataFrame({'col': posts})
print df
df.to_csv("finaloutput.csv", sep=',')
