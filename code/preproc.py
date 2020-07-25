"""
Utility functions to preprocess the datasets that we are implying (categorical features, unused features, etc...). These functions were provided directly from the sources we gathered those datasets from
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

# Note: Titanic dataset was discarded
def titanic_preproc(data):
    data['relatives'] = data['SibSp'] + data['Parch']

    data.loc[data['relatives'] > 0, 'not_alone'] = 0
    data.loc[data['relatives'] == 0, 'not_alone'] = 1

    data['not_alone'] = data['not_alone'].astype(int)
    data['_name'] = LE.fit_transform(data['Name'])
    data['_sex'] = LE.fit_transform(data['Sex'])
    data['_ticket'] = LE.fit_transform(data['Ticket'])
    data['Embarked'] = data['Embarked'].fillna(value=data['Embarked'].mode()[0])
    data['_embarked'] = LE.fit_transform(data['Embarked'])

    # Drop leftover columns 
    data = data.drop(['PassengerId', 'Sex', 'relatives', 'Parch', 'SibSp',
                      'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    return data

def crimes_preproc(data):
    data['Category'] = LE.fit_transform(data.Category)

    feature_cols =['DayOfWeek', 'PdDistrict']
    data = pd.get_dummies(data, columns=feature_cols)
    
    data['years'] = data['Dates'].dt.year
    data['months'] = data['Dates'].dt.month
    data['days'] = data['Dates'].dt.day
    data['hours'] = data['Dates'].dt.hour
    data['minutes'] = data['Dates'].dt.minute
    data['seconds'] = data['Dates'].dt.second

    data = data.drop(['Dates', 'Address','Resolution'], axis = 1)
    data = data.drop(['Descript'], axis = 1)

    return data    
