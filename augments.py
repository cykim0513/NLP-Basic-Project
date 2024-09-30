import pandas as pd
import urllib.request
import json


## Under Sampling 통해 0점 데이터 개수 줄이기 (750개 제거)
def undersampling():
    df = pd.read_csv('train.csv')
    df_0 = df[df['label']==0][0:1500].copy()

    df_new = df[df['label']!=0].copy()
    df_new = pd.concat([df_new, df_0])
    df_new


## swap sentence
def swap_sentence():
    df_switched = pd.read_csv('train.csv')
    df_switched["sentence_1"] = df["sentence_2"]
    df_switched["sentence_2"] = df["sentence_1"]
    df_switched = df_switched[df_switched['label'] != 0]
    df_switched


## copied sentence
def copied_sentence():
    copied_df = df[df['label']==0][1500:].copy()
    copied_df['sentence_1'] = copied_df['sentence_2']
    copied_df['label'] = 5.0
    copied_df