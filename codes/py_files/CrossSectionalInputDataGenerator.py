import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class CrossSectionalInputDataGenerator():
    product_col_name = 'target_product'
    all_products = [
        'Asset Management', 'Aval', 'Betriebliche Altersvorsorge',
        'Bond-Emissionen', 'Bürgschaften und Garantien', 'Cash Pooling',
        'Commerz Real - Mobilienleasing',
        'Export Dokumentengeschäft', 'Export- und Handelsfinanzierung',
        'Forderungsmanagement', 'Geldmarktkredit', 'Global Payment Plus',
        'Import Dokumentengeschäft', 'KK-Kredit', 'Kapitalanlagen',
        'Rohstoffmanagement', 'Sichteinlagen', 'Termin-Einlage',
        'Unternehmensfinanzierung', 'Währungsmanagement', 'Zinsmanagement'
    ]
    excluded_products = ['Commerz Real - Immobilienleasing']
    transaction_variables = [
        'EUR', 'USD', 'GBP', 'Other_Currencies',
        'Abfallentsorgung und Rueckgewinnung', 'Bauindustrie und Handwerk',
        'Beherbergung und Gastronomie', 'Beratungsunternehmen', 'Bergbau',
        'Chemische Erzeugnisse', 'Energieversorgung',
        'Erziehung und Unterricht', 'Forstwirtschaft', 'Gebaeudebetreuung',
        'Gesundheitswesen', 'Glas und Glaswaren',
        'Grundstuecks- und Wohnungswesen', 'Handel mit Kraftfahrzeugen',
        'Herst. von Kraftwagen', 'Herst. von Papier',
        'Herst. von elektrischen Ausruestungen', 'Landwirtschaft',
        'Maschinenbau', 'Metallerzeugung und -bearbeitung',
        'Mit Finanz- und VersicherungsDienstl. verbundene Taetigkeiten',
        'Moebel', 'Oeffentliche Verwaltung',
        'Personen- und Gueterbefoerderung / Lagerei',
        'Pharmazeutische Erzeugnisse', 'Reisebueros und Reiseveranstalter',
        'Sozialwesen', 'Textilien und Bekleidung', 'Untagged',
        'Unterhaltungsmedien wie Film', 'Verlagswesen',
        'Vermittlung und Ueberlassung von Arbeitskraeften', 'Versicherungen',
        'Wasserversorgung und Abwasserentsorgung', 'Werbung und Marktforschung'
    ]
    categorical_features = ["gp_rckdgrp_", "gp_bonirati", "gp_wzbran2_"]
    recency_features = [*all_products, *transaction_variables]
    frequency_features = [*all_products, *transaction_variables]
    monetary_value_features = [*transaction_variables, 'Valutasaldo_avg']
    time_windows = [
        {
            'feature': [0, 24],
            'target': [24, 29]
        },
        {
            'feature': [6, 30],
            'target': [30, 35]
        },
        {
            'feature': [12, 36],
            'target': [36, 41]
        },
    ]

    test_and_valid_samples_per_train_sample = 0.25
    valid_samples_per_test_sample = 0.5
    client_ids = []
    tolerated_idle_clients = 10000

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.set_index(['customer_gpkenn', 'dateYYYYMM'])
        self.exclude_idle_clients()
        self.client_ids = list(self.data.copy().reset_index().customer_gpkenn.unique())
        random.shuffle(self.client_ids)
        self.prepare_categorical_features(
            ["gp_rckdgrp_", "gp_bonirati", "gp_wzbran2_"])

    def get_client_ids_who_bought_at_least_n_products(
            self, relevant_client_ids: List[str],
            minimum_products: int) -> List[str]:
        client_id_occurence_count = self.data.reset_index(
        )['customer_gpkenn'].value_counts()
        all_client_ids_with_more_than_n_products = client_id_occurence_count[
            client_id_occurence_count >= minimum_products].index
        self.excluded_client_ids = list(
            set(relevant_client_ids) -
            set(all_client_ids_with_more_than_n_products))
        return list(
            set(relevant_client_ids)
            & set(all_client_ids_with_more_than_n_products))

    def create_input_data(self, subsample_size=0) -> Dict[str, np.array]:
        """Loops through given client_ids and in each iteration creates X/Y data in the right format for modelling and concatenates this data with previous X/Y data. Safes formatted X/Y for all selected clients as attribute.
        """
        if subsample_size: 
            self.client_ids = random.sample(self.client_ids, subsample_size)
        x_data, y_data = [], []
        for customer_gpkenn in tqdm(self.client_ids):
            client_data = self.get_client_data(customer_gpkenn)
            for time_window in self.time_windows:
                y_data.append(
                    self.slice_delta_products(client_data, time_window))
                x_data.append([
                    self.create_cross_sectional_features(
                        client_data, time_window)
                ])
        self.input_data = self.create_dict_of_train_valid_test_split_no_shuffle(
            x_data, y_data)

    def create_dict_of_train_valid_test_split_no_shuffle(
            self, x_data: np.array, y_data: np.array) -> Dict[str, np.array]:
        X_train, X_test_valid, Y_train, Y_test_valid = train_test_split(
            np.array(x_data),
            np.array(y_data),
            test_size=self.test_and_valid_samples_per_train_sample,
            shuffle=False)
        X_test, X_valid, Y_test, Y_valid = train_test_split(
            X_test_valid,
            Y_test_valid,
            test_size=self.valid_samples_per_test_sample,
            shuffle=False)
        return dict(
            zip([
                'X_train', 'X_valid', 'X_test', 'Y_train', 'Y_valid', 'Y_test'
            ], [X_train, X_valid, X_test, Y_train, Y_valid, Y_test]))

    def slice_delta_products(self, client_data: pd.DataFrame,
                             time_window: Dict[str, int]) -> np.array:
        target_over_time = client_data[self.all_products]
        target_changes_over_time = self.get_all_deltas_from_zero_to_one(
            target_over_time)
        col = target_changes_over_time.to_numpy().tolist()
        return [
            max(x) for x in zip(
                *col[time_window['target'][0]:time_window['target'][1]])
        ]

    def create_cross_sectional_features(
            self, client_data: pd.DataFrame,
            time_window: Dict[str, int]) -> np.array:
        client_data_slice = client_data.reset_index(
            drop=True)[time_window['feature'][0]:time_window['feature'][1]]
        categorical_features = self.select_categorical_features(
            client_data_slice)
        recency_features = self.calculate_recency_features(client_data_slice)
        frequency_features = self.calculate_frequency_features(
            client_data_slice)
        monetary_value_features = self.calculate_monetary_value_featues(
            client_data_slice)
        return pd.concat([
            categorical_features, recency_features, frequency_features,
            monetary_value_features
        ]).to_numpy()

    def select_categorical_features(
            self, client_data_slice: pd.DataFrame) -> pd.DataFrame:
        return client_data_slice[self.categorical_features].loc[
            client_data_slice.index[-1]]

    def calculate_recency_features(
            self, client_data_slice: pd.DataFrame) -> pd.DataFrame:
        data_slice_recency_features = client_data_slice[self.recency_features]
        return data_slice_recency_features.apply(
            lambda column: CrossSectionalInputDataGenerator.
            calculate_recency_for_one_column(column),
            axis=0,
            raw=True)

    @staticmethod
    def calculate_recency_for_one_column(column: np.array) -> int:
        """Returns the inverse position of the last non-zero value in a np.ndarray of numerics.
        If the last value is non-zero, returns 1. If all values are non-zero, returns 0."""
        non_zero_values_of_col = np.nonzero(column)[0]
        if not non_zero_values_of_col.any():
            return 0
        return len(column) - non_zero_values_of_col[-1]

    def calculate_frequency_features(
            self, client_data_slice: pd.DataFrame) -> pd.DataFrame:
        return client_data_slice[self.frequency_features][
            ~client_data_slice[self.frequency_features].isnull()].sum()

    def calculate_monetary_value_featues(
            self, client_data_slice: pd.DataFrame) -> pd.DataFrame:
        return client_data_slice[self.monetary_value_features].mean()

    def get_input_data(self) -> Dict[str, np.array]:
        return self.input_data

    def get_all_dummy_columns(self, all_columns: List[str],
                              feature_names: List[str]) -> List[str]:
        """this function has the following logic wihtin a double list comprehensions with two if statements to get the names of all dummy columns:
        dummy_columns = []
        for feature_name in feature_names:
            for column in all_columns:
                if feature_name in column:
                    if column != feature_name:
                        dummy_columns.append(column)
        In words this means that for each feature_name the fct collects all columns that contain the name of the feature but are not the parent feature themselves. Eg. "bonirati" is a feature name and "bonirati_10.0" is a dummy name.
        """
        return [
            column for feature_name in feature_names for column in all_columns
            if feature_name in column if column != feature_name
        ]

    def track_categorical_features(self, feature_col_names: List[str]) -> None:
        self.categorical_features = self.get_all_dummy_columns(
            self.data.columns, feature_col_names)

    def prepare_categorical_features(self,
                                     feature_col_names: List[str]) -> None:
        """Creates categorical dummies and tracks these features as categorical. Also deletes source columns.
        """
        self.data[feature_col_names] = self.data[feature_col_names].astype(str)
        self.data = pd.concat(
            [self.data,
             pd.get_dummies(self.data[feature_col_names])], axis=1)
        self.track_categorical_features(feature_col_names)
        self.data = self.data.drop(feature_col_names, axis='columns')

    def get_client_data(self, id_one_client: str) -> pd.DataFrame:
        """Takes advantage of set index in self.data for higher performance"""
        return self.data.loc[id_one_client]

    def slice_y_data_sliding_window(self,
                                    data_matrix: pd.DataFrame) -> np.array:
        """Turns pandas product matrix into a multidimensional array. Uses a window of size 12 that moves 1 month forward per iteration to slice the product matrix and returns the next N months as an array (shape=(j,l), where j is the number of windows that fit into this product matrix and l is the number of products). This target array contains the information which products have been used within the target time winow at least once."""
        col = data_matrix.to_numpy().tolist()
        Y = []
        for time_window in self.time_windows:
            Y.append([
                max(x) for x in zip(
                    *col[time_window['target'][0]:time_window['target'][1]])
            ])
        return np.array(Y)

    def slice_x_data_sliding_window(self,
                                    data_matrix: pd.DataFrame) -> np.array:
        """Turns pandas product matrix into a multidimensional array. Uses a window of size 12 that moves 1 month forward per iteration to slice the product matrix and returns the results as an array (shape=(j,k,l), where j is the number of windows that fit into this product matrix, k is the window length in months and l is the number of products).
        """
        col = data_matrix.to_numpy().tolist()
        X = []
        for time_window in self.time_windows:
            X.append(col[time_window['feature'][0]:time_window['feature'][1]])
        return np.array(X)

    def calculate_slice_positions(self,
                                  data_matrix: pd.DataFrame) -> List[int]:
        return range(
            0,
            len(data_matrix) + 1 - self.feature_window_size -
            self.target_window_size, self.sliding_window_step_size)

    @staticmethod
    def get_all_deltas_from_zero_to_one(
            product_matrix: pd.DataFrame) -> pd.DataFrame:
        """Turns
               A  B  C
            0  0  1  0
            1  0  1  1
            2  1  1  1
            3  1  1  0
            4  0  0  1
        into:
               A  B  C
            0  0  0  0
            1  0  0  1
            2  1  0  0
            3  0  0  0
            4  0  0  1
        """
        matrix = product_matrix.copy()
        return matrix.diff().clip(0).fillna(0).astype(int)

    def exclude_idle_clients(self):
        '''excludes all clients that have targets_delta_sum of 0, meaning they bought no new product in the target timeframe (18 last months).'''
        data_unindexed = self.data.reset_index()
        all_idle_client_ids = list(data_unindexed[
            data_unindexed['targets_delta_sum'] == 0].customer_gpkenn.unique())
        random.shuffle(all_idle_client_ids)
        self.data = data_unindexed[~data_unindexed["customer_gpkenn"].isin(
            all_idle_client_ids[self.tolerated_idle_clients:])].set_index(
                ['customer_gpkenn', 'dateYYYYMM'])