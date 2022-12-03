import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sps
import csv

def load_URM(file_path):
    data = pd.read_csv(file_path)

    user_list = data['UserID'].tolist()
    item_list = data['ItemID'].tolist()
    rating_list = data['Data'].tolist()

    return sps.coo_matrix((rating_list, (user_list, item_list))).tocsr()

def load_ICM(file_path):
    metadata = pd.read_csv(file_path)

    item_icm_list = metadata['item_id'].tolist()
    feature_list = metadata['feature_id'].tolist()
    weight_list = metadata['data'].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))

def combine(ICM: sps.csr_matrix, URM : sps.csr_matrix):
    return sps.hstack((URM.T, ICM), format='csr')

def binarize(x):
    if x != 0:
        return 1
    return x

def binarize_ICM(ICM: sps.csr_matrix):
    vbinarize = np.vectorize(binarize)

    ICM.data = vbinarize(ICM.data)

def write_submission(recommender, target_users_path,out_path):
    targetUsers = pd.read_csv(target_users_path)['user_id']

    # topNRecommendations = recommender.recommend(targetUsers.to_numpy(), cutoff=10)

    targetUsers = targetUsers.tolist()

    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in targetUsers:
            writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])