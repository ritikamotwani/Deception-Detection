import os, string
import pandas as pd
import itertools

def read_fileNames(src, datapath, subfolder=None):
    for home, dirs, files in os.walk(datapath+subfolder):
        for filename in files:
            src.append(home+'/'+filename)

def sort_Lists(sources, length):
    for i in range(0,length):
        sources[i]=sorted(sources[i])

def read_sort_CSV(df, datapath, filename, sort_column):
    df.append(pd.read_csv(datapath+filename))
    index=len(df)-1
    df[index]=df[index].sort_values(by=sort_column)

def readFilesFromSources(text, sources):
    for source in list(itertools.chain.from_iterable(sources)):
        with open(source) as f_input:
            text.append(f_input.read())