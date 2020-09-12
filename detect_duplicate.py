import numpy as np
import re
from enchant.utils import levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

REF_YEAR = 2020

def ngrams(string, n=3):
    ''' Clean string and produce ngrams'''
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) #remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    zip_ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in zip_ngrams]

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)
    return csr_matrix((data,indices,indptr),shape=(M,N))

def correct_state(state, list_of_states):
    ''' Clean "state" typos '''
    # Not 100% correct because of states 'sa' and 'wa'
    
    if state is None or isinstance(state, float) or state in list_of_states:
        return state
    return list_of_states[np.argmin(list(map(lambda x: levenshtein(state, x), list_of_states)))]

def detect_duplicates(df_patient):
    ''' Deduplicate df_patient '''
    df = df_patient.copy()

    # cleanable typos (state) and missing values (age) : help the deduplication for better similarity
    birth_not_age = (df.age.isna()) & (~df.date_of_birth.isna())
    df.loc[birth_not_age, 'age'] = REF_YEAR - df[birth_not_age].date_of_birth // 10000
    states = list(df.state.value_counts().nlargest(8).index)
    df.state = list(map(lambda x: correct_state(x, states), df.state))
    
    # Prepare ONE string column merged from all columns (except the patient ID, likely not typed)
    df.fillna(' ', inplace=True)
    df = df.astype(str)
    merge_all = list(map(lambda x: ' '.join(x), df.values))
    
    if 'patient_id' in df_patient:
        df.drop(columns='patient_id', inplace=True)
    
    # Search duplicates with TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(merge_all)
    matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.60).toarray()
    
    #  get index of duplicates
    np.fill_diagonal(matches, 0)
    lines, cols = np.where(matches > 0)
    index_to_delete = [j for i,j in zip(lines, cols) if j > i]
    return df_patient.loc[~df_patient.index.isin(index_to_delete)]