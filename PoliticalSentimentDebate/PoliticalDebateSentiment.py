# Chad Wyngaard
# 216122740
# Projects 3: NLP

# Imports
#---------
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# Dataset insert
#----------------
data = pd.read_csv('2020 Debate Transcripts.csv',encoding = 'iso-8859-1')

# Method used for getting the correct formatting
#------------------------------------------------
def joinMethod(row):
    dataList = row['Discourse w/o names']
    joined = ( " ".join(dataList))
    return joined

# Algorithm  used for cleaning data
#-----------------------------------
data['Discourse w/o names'] = data['Discourse w/o names'].str.lower().str.split()
language = set(stopwords.words("english"))
data['Discourse w/o names'] = data['Discourse w/o names'].apply(lambda x : [line for line in x if line not in language])
data['Discourse w/o names'] = data.apply(joinMethod, axis=1)

# Creating the dataFrame
#------------------------
data1 = pd.DataFrame(data, columns = ['Speaker', 'Topic','Discourse w/o names'])

# Extracting the unnecessary speakers and topics from their respective columns
#------------------------------------------------------------------------------
data1 = data1[data1['Speaker']!='ANNOUNCER']
data1 = data1[data1['Speaker']!='(UNKNOWN)']
data1 = data1[data1['Topic']!='Undocumented Immigrants']

# Algorithm to attain plolarity from dataset
#--------------------------------------------
data1['Polarity'] = data1.apply(lambda x: TextBlob(x['Discourse w/o names']).sentiment.polarity, axis=1)

# Calculating the polarity and storing the values
#-------------------------------------------------
agreementPolarity =  data1.Polarity[data1.Polarity > 0].sum()
conflictPolarity = data1.Polarity[data1.Polarity < 0].sum() * -1

# Checking to see if the polarities for positive and negative comments are correct
#----------------------------------------------------------------------------------
print ("Agreement: {}".format(agreementPolarity))
print ("Conflict: {}".format(conflictPolarity))

# Converting the cleaned code to a csv file
#-------------------------------------------
data1.to_csv('CleanedData.csv')

# Creating the model
#--------------------
df = pd.DataFrame({'Agreement compared to Conflict' : [agreementPolarity,conflictPolarity],
                  'radius': [2439.7, 6051.8]},
               index=['Agreemnt', 'Conflict'])

plot = df.plot.pie(y='Agreement compared to Conflict', figsize=(5, 5),autopct='%.1f%%')
plt.show()

