import pandas as pd
import numpy as np
import detect_duplicate as dd

data = {'patient_id' : [77777],
        'given_name': ['prenom'], 
        'surname': ['nom'],
        'street_number' : [999],
        'address_1' : ['add1'],
        'suburb': ['suburb'], 
        'postcode': [1000],
        'state': ['state'],
        'date_of_birth' : [19680717.0],
        'age' : [540],
        'phone_number' : ['01 23456789'],
        'address_2':['address2']}
df_test = pd.DataFrame(data=data)
df_test = pd.concat([df_test]*df_test.shape[1]).reset_index().drop(columns='index')


def test_one_na():
    ''' One NaN at different locations '''
    copy = df_test.copy()
    
    # fill with 1 NaN
    for i in range(copy.shape[1]):
        copy.loc[i, copy.columns[i]] = np.nan

    assert len(dd.detect_duplicates(copy)) == 1

def test_two_na():
    ''' Two NaN at different locations '''
    copy = df_test.copy()
    
    # fill with 2 NaN
    for i in range(copy.shape[1]):
        copy.loc[i, copy.columns[i]] = np.nan

        if i < copy.shape[1]-1:
            copy.loc[i, copy.columns[i+1]] = np.nan

    assert len(dd.detect_duplicates(copy)) == 1

def test_typos():
    ''' Increasing number of typos '''
    copy = df_test.copy()
    
    copy.loc[1, 'given_name'] = 'prrnom'
    copy.loc[2, 'given_name'] = 'prrbom'
    copy.loc[3, 'given_name'] = 'prrbim'
    copy.loc[4, 'given_name'] = 'orrbim'
    copy.loc[5, 'given_name'] = 'orrbin'
    
    assert len(dd.detect_duplicates(copy)) == 1

def test_inverted_entries():
    ''' Exchange two entries '''
    copy = df_test.copy()
    
    copy.loc[1, 'postcode'] = copy.loc[1, 'suburb']
    copy.loc[1, 'suburb'] = copy.loc[0, 'postcode'] # all rows are the same (same postcode)
    
    assert len(dd.detect_duplicates(copy.iloc[:2])) == 1
