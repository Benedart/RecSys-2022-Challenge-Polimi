
import os
import traceback

from Data_manager.loaders import *
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_import_list import *


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object

if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample

    print("Loading data...")
    URM_all = load_URM("input\interactions_and_impressions.csv")
    ICM_length = load_ICM("input\data_ICM_length.csv")
    ICM_type = load_ICM("input\data_ICM_type.csv")
    UCM_all = None

    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)
    ICM_all = ICM_length
    UCM_all = ICM_length

    print(URM_train.shape[0], URM_validation.shape[0], URM_test.shape[0])

    print("Loading complete")

    recommender_class_list = [
        Random,
        TopPop,
        GlobalEffects,
        SLIMElasticNetRecommender,
        UserKNNCFRecommender,
        IALSRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        SLIM_BPR_Cython,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
    #    UserKNNCBFRecommender,
        ItemKNNCBFRecommender,
    #    UserKNN_CFCBF_Hybrid_Recommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        LightFMCFRecommender,
    #    LightFMUserHybridRecommender,
        LightFMItemHybridRecommender,
    ]


    evaluator = EvaluatorHoldout(URM_test, [5, 20], exclude_seen=True)

    # from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": EvaluatorHoldout(URM_validation, [20], exclude_seen=True),
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }


    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")


    for recommender_class in recommender_class_list:

        try:

            print("Algorithm: {}".format(recommender_class))

            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15, **earlystopping_keywargs}
            else:
                fit_params = {}

            recommender_object.fit(**fit_params)

            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)

            recommender_object.save_model(output_root_path, file_name = "temp_model.zip")

            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            recommender_object.load_model(output_root_path, file_name = "temp_model.zip")

            os.remove(output_root_path + "temp_model.zip")

            results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

            if recommender_class not in [Random]:
                assert results_run_1.equals(results_run_2)

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_class, results_run_string_1))
            logFile.flush()


        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
