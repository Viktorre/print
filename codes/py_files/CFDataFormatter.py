import pandas as pd
import random
from sklearn.model_selection import train_test_split

class CFDataFormatter():

    target_columns = [
        'Asset Management', 'Aval', 'Betriebliche Altersvorsorge',
        'Bond-Emissionen', 'Bürgschaften und Garantien', 'Cash Pooling',
        'Commerz Real - Mobilienleasing',
        'Export Dokumentengeschäft', 'Export- und Handelsfinanzierung',
        'Forderungsmanagement', 'Geldmarktkredit', 'Global Payment Plus',
        'Import Dokumentengeschäft', 'KK-Kredit', 'Kapitalanlagen',
        'Rohstoffmanagement', 'Sichteinlagen', 'Termin-Einlage',
        'Unternehmensfinanzierung', 'Währungsmanagement', 'Zinsmanagement'
    ]
    excluded_target_columns = ['Commerz Real - Immobilienleasing']
    tolerated_idle_clients = 10000
    
    time_windows = [
        {
            'feature': [202007,202008,202009,202010,202011,202012],
            'target': [202101,202102,202103,202104,202105,202106]
        },
        {
            'feature': [202101,202102,202103,202104,202105,202106],
            'target': [202107,202108,202109,202110,202111,202112]
        },
        {
            'feature': [202107,202108,202109,202110,202111,202112],
            'target': [202201,202202,202203,202204,202205,202206]
        },
    ]

    def __init__(self, products_and_features_table: pd.DataFrame) -> None:
        products_and_features_table_only_relevant_clients = self.exclude_idle_clients(products_and_features_table)
        products_delta = products_and_features_table_only_relevant_clients.groupby([
            'customer_gpkenn'
        ])[self.target_columns].diff().clip(0).fillna(0).astype(int)
        self.products_delta = pd.concat([products_and_features_table_only_relevant_clients[['customer_gpkenn', 'dateYYYYMM']],\
                products_delta],axis=1)

    def exclude_idle_clients(self, data) ->pd.DataFrame:
        '''excludes all clients that have targets_delta_sum of 0, meaning they bought no new product in the target timeframe (18 last months).'''
        data_unindexed = data
        all_idle_client_ids = list(data_unindexed[
            data_unindexed['targets_delta_sum'] == 0].customer_gpkenn.unique())
        random.shuffle(all_idle_client_ids)
        return data_unindexed[~data_unindexed["customer_gpkenn"].isin(
            all_idle_client_ids[self.tolerated_idle_clients:])]


    def give_merged_targets(self) ->pd.DataFrame: 
        merged_targets = pd.concat([self.select_products_for_timeframe(time_window['target']) \
               for time_window in self.time_windows])
        return train_test_split(merged_targets,test_size=0.125,shuffle=False)[1]
    
    def select_products_for_timeframe(self, feature_dates):
        feature = self.products_delta[self.products_delta['dateYYYYMM'].isin(
            feature_dates)]
        feature = feature[[
            'customer_gpkenn', 'dateYYYYMM', *self.target_columns
        ]]
        feature = feature.groupby(['customer_gpkenn'
                                   ])[self.target_columns].max()
        return feature