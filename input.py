from utilities import *
from sklearn import preprocessing

def readTxt_Spam():
    #create source
    sources = [[] for i in range(4)]

    datapath='./Spam_Detection_Data/'

    #read filenames from directories
    read_fileNames(sources[0], datapath,'deceptive_neg')
    read_fileNames(sources[1], datapath,'truthful_neg')
    read_fileNames(sources[2], datapath,'deceptive_pos')
    read_fileNames(sources[3], datapath,'truthful_pos')

    #sort filenames in each directory
    sort_Lists(sources,len(sources))

    #set class labels
    labels=np.concatenate((np.ones((400),dtype=int),np.zeros((400),dtype=int),np.ones((400),dtype=int),np.zeros((400),dtype=int)))
    #encode Deceptive/Truthful Class labels
    encoder = preprocessing.LabelEncoder()
    labels=encoder.fit_transform(labels)

    #read text files from source list
    text=[]
    readFilesFromSources(text,sources)
    
    return text,labels

def readLIWC_Spam():
    datapath='./Spam_Detection_Data/'
    
    #initialize dataframe for different sets of LIWC features
    df=[]

    read_sort_CSV(df, datapath, 'LIWC_Negative_Deceptive.csv','Filename')
    read_sort_CSV(df, datapath, 'LIWC_Negative_Truthful.csv','Filename')
    read_sort_CSV(df, datapath, 'LIWC_Positive_Deceptive.csv','Filename')
    read_sort_CSV(df, datapath, 'LIWC_Positive_Truthful.csv','Filename')

    #concatenate individual frames into one dataframe
    dfLIWC=pd.concat((df[0],df[1],df[2],df[3])).iloc[:,2:]

    return dfLIWC

def readTxt_RealLife():
    #create source
    sources = [[] for i in range(2)]
    datapath = './Real_Life_Trial_Data/'

    #read file names from datapath
    read_fileNames(sources[0], datapath,'Deceptive')
    read_fileNames(sources[1], datapath,'Truthful')

    #sort filenames
    sort_Lists(sources,len(sources))

    #create label array corresponding to text files
    labels=np.concatenate((np.ones(len(sources[0]), dtype=int),np.zeros(len(sources[1]), dtype=int)))
    #encode Deceptive/Truthful Class labels
    encoder = preprocessing.LabelEncoder()
    labels=encoder.fit_transform(labels)

    #read text files from source list
    text=[]
    readFilesFromSources(text,sources)
    return text,labels


def readLIWC_RealLife():

    datapath = './Real_Life_Trial_Data/'

    #initialize dataframe for different sets of LIWC features
    df=[]
    
    read_sort_CSV(df, datapath, 'LIWC2015_Deceptive.csv','Filename')
    read_sort_CSV(df, datapath, 'LIWC2015_Truthful.csv','Filename')

    #concatenate 2 different feature sets together
    dfLIWC=pd.concat((df[0],df[1])).iloc[:,2:]
    
    return dfLIWC