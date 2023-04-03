import itertools
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from typing import List
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout

from Metrics import turn_predictions_into_binaries, create_classification_report_custom_binary_threshold


def compute_cf_recommendation(id: int, user_similarity: pd.DataFrame,
                              feature: pd.DataFrame,similarity_range:int) -> pd.Series:
    most_similar_clients = user_similarity.loc[id].dropna().sort_values(
        ascending=False).reset_index()
    most_similar_clients = most_similar_clients[
        most_similar_clients['customer_gpkenn'] != id]
    n_most_similar_clients_with_score = most_similar_clients.loc[:similarity_range]
    features_n_most_similar_clients = feature.loc[
        n_most_similar_clients_with_score['customer_gpkenn'].values]  #.loc[id]
    return features_n_most_similar_clients.multiply(n_most_similar_clients_with_score[id].values,axis='rows').mean(axis='rows')\
    #- feature.loc[id]


def compute_cf_recommendation_for_all_clients(feature,similarity_range) -> pd.DataFrame:
    user_similarity_active_clients = feature.T.corr()
    recommendation_scores_per_product = []
    for id in user_similarity_active_clients.index.values:
#    for id in tqdm(user_similarity_active_clients.index.values):    
        recommendation_scores_per_product.append(
            compute_cf_recommendation(id, user_similarity_active_clients,
                                      feature,similarity_range))
    recommender_scores_active_clients = pd.DataFrame(
        recommendation_scores_per_product, index=feature.index).fillna(0)
    return recommender_scores_active_clients

def calculate_and_merge_cf_predictions(cf_data_formatter,similarity_range):
    predictions = []
    for time_window in cf_data_formatter.time_windows:
        feature = cf_data_formatter.select_products_for_timeframe(
            time_window['feature'])
        predictions.append(compute_cf_recommendation_for_all_clients(feature,similarity_range))
    predictions = pd.concat(predictions)
    return train_test_split(predictions,test_size=0.125,shuffle=False)[1]
    
def evaluate_hparam_combination_collaborative_filter(hparams,cf_data_formatter,metric="f1-score"):
    merged_predictions = calculate_and_merge_cf_predictions(cf_data_formatter,hparams[0])
    merged_targets = cf_data_formatter.give_merged_targets()
    classification_report = create_classification_report_custom_binary_threshold(
        turn_predictions_into_binaries(merged_predictions.copy(), 0.05).to_numpy(),
        merged_targets.to_numpy(), cf_data_formatter.target_columns)
    return classification_report.loc["micro avg"][metric]

def grid_search_tune_collaborative_filter(hparams,cf_data_formatter,metric="f1-score") ->pd.DataFrame:
    all_hp_combinations = [*itertools.product(*hparams.values())]
    results = []
    for hparams_combination in tqdm(all_hp_combinations):
        evaluation = evaluate_hparam_combination_collaborative_filter(hparams_combination,cf_data_formatter,metric)
        results.append({"hparams":hparams_combination,"metric":evaluation})
    return pd.DataFrame(results).sort_values("metric",ascending=False).reset_index(drop=True)


def evaluate_hparam_combination_cs(hparams,input_data,metric="f1-score"):
    dtree_classifier = DecisionTreeClassifier(criterion=hparams[0], max_depth=hparams[1], min_samples_split=hparams[2], \
                        min_samples_leaf=hparams[3],random_state=42)
    dtree_classifier = dtree_classifier.fit(input_data['X_train'],input_data['Y_train'])
    report_cross_section_approach = create_classification_report_custom_binary_threshold(
    turn_predictions_into_binaries(dtree_classifier.predict(input_data['X_test']), 0.5),
    input_data['Y_test'],csa_data.all_products)
    return report_cross_section_approach.loc["micro avg"][metric]

def randomly_tune_dtree(hparams,input_data,csa_data,runs,metric="f1-score") ->pd.DataFrame:
    results = []
    for _ in tqdm(range(runs)):
        random_hparams_combination = random.choice([*itertools.product(*hparams.values())])
        evaluation = evaluate_hparam_combination_cs(random_hparams_combination,input_data,metric)
        results.append({"hparams":random_hparams_combination,"metric":evaluation})
    return pd.DataFrame(results).sort_values("metric",ascending=False).reset_index(drop=True)

def grid_search_tune_dtree(hparams,input_data,metric="f1-score") ->pd.DataFrame:
    all_hp_combinations = [*itertools.product(*hparams.values())]
    results = []
    for hparams_combination in tqdm(all_hp_combinations):
        evaluation = evaluate_hparam_combination_cs(hparams_combination,input_data,metric)
        results.append({"hparams":hparams_combination,metric:evaluation})
    return pd.DataFrame(results).sort_values(metric,ascending=False).reset_index(drop=True)


def build_lstm_for_tuning_rmsprop_generic(hp):
    activation=['tanh','sigmoid']
    lossfct='binary_crossentropy'
    hidden_units_first_layer = hp.Choice('neurons first layer',[32,64,128,256,512,1024])
    lr = hp.Choice('learning_rate', [0.005])
    momentum = hp.Choice('momentum', [0.9])
    epsilon = hp.Choice('epsilon', [1e-06])
    model = Sequential()
    model.add(LSTM(hidden_units_first_layer,input_shape=(24, 236),activation=activation[0]))
    model.add(Dense(units=21, activation=activation[1]))
    optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=lr,momentum=momentum,epsilon=epsilon)
    model.compile(loss=lossfct, optimizer= optimizer,metrics=[tf.keras.metrics.Accuracy(),\
                                                          tf.keras.metrics.BinaryAccuracy(),\
                                                          tf.keras.metrics.MeanSquaredError(),\
                                                          tf.keras.metrics.Precision(),\
                                                          tf.keras.metrics.Recall(),\
                                                          tf.keras.metrics.TruePositives(),
                                                          tf.keras.metrics.AUC(multi_label=True)
                                                          ])
    return model



def build_lstm_for_tuning_rmsprop_specific(hp):
    activation=['tanh','sigmoid']
    lossfct='binary_crossentropy'
    hidden_units_first_layer = hp.Choice('neurons first layer',[1024])
    lr = hp.Choice('learning_rate', [0.005])
    momentum = hp.Choice('momentum', [0.9])
    epsilon = hp.Choice('epsilon', [1e-06])
    model = Sequential()
    model.add(LSTM(hidden_units_first_layer,input_shape=(24, 236),activation=activation[0]))
    model.add(Dense(units=21, activation=activation[1]))
    optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=lr,momentum=momentum,epsilon=epsilon)
    model.compile(loss=lossfct, optimizer= optimizer,metrics=[tf.keras.metrics.Accuracy(),\
                                                          tf.keras.metrics.BinaryAccuracy(),\
                                                          tf.keras.metrics.MeanSquaredError(),\
                                                          tf.keras.metrics.Precision(),\
                                                          tf.keras.metrics.Recall(),\
                                                          tf.keras.metrics.TruePositives(),
                                                          tf.keras.metrics.AUC(multi_label=True)
                                                          ])
    return model