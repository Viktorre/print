import difflib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
# import tensorflow as tf
from typing import List, Dict


def call_all_training_plots(history) -> None:
    all_metric_names = history.history.keys()
    for metrics_that_go_together_in_one_plot in [
        ['accuracy',
         'val_accuracy'], ['binary_accuracy', 'val_binary_accuracy'],
        ['loss', 'val_loss'],
        ['precision', 'recall', 'val_precision', 'val_recall','auc','val_auc'],
        ['mean_squared_error', 'val_mean_squared_error']
    ]:
        plot_training_history(
            history,
            get_most_similar_metric_name(metrics_that_go_together_in_one_plot,
                                         all_metric_names))

def plot_training_history(history,
                          metric_names: List[str]) -> None:
    pd.DataFrame(history.history)[metric_names].plot(figsize=(8, 5))
    plt.show()

def get_most_similar_metric_name(metric_names_to_be_changed: List[str],
                                 all_metric_names: List[str]) -> str:
    return [
        difflib.get_close_matches(metric_name, all_metric_names)[0]
        for metric_name in metric_names_to_be_changed
    ]

def turn_predictions_into_binaries(prediction: np.array, threshold=0.5) -> np.array:
    prediction[prediction >= threshold] = 1
    prediction[prediction < threshold] = 0
    return prediction


def create_classification_report_custom_binary_threshold(prediction:np.array,test_data_y:np.array,all_products:List[str]) -> pd.DataFrame:
    from sklearn.metrics import classification_report
    metrics = pd.DataFrame(
        classification_report(test_data_y,
                              prediction,
                              zero_division=0,
                              output_dict=True))
    metric_column_names_with_products_and_averages = [
        *all_products, *metrics.columns[-4:]
    ]
    metrics.columns = metric_column_names_with_products_and_averages
    metrics = metrics.drop(columns=["samples avg","weighted avg","macro avg"])
    metrics = metrics.T
    metrics['pr_auc'] = calculate_pr_auc(prediction,test_data_y,all_products)[0].values
    metrics = metrics[["precision", "recall", "f1-score", "pr_auc","support"]]
    return metrics


def plot_multilabel_pr_curve( y_score:np.array,y_test: np.array,product_names:List[str],plot_name:str) -> None:
    pr_curves = dict()
    for i in range(len(product_names)):
        pr_curves[product_names[i]] = pd.DataFrame(precision_recall_curve(y_test[:, i], y_score[:, i]),index=['Precision','Recall',"Threshold"]).T.set_index("Threshold")
    create_2d_plot_of_metrics_for_avgs_and_per_class(
        pr_curves, y_test, "Precision Recall Cruve", "Precision", "Recall (TPR)",product_names,plot_name)
    
def create_2d_plot_of_metrics_for_avgs_and_per_class(
        pr_curves, y_test: np.array, plot_title: str, label_y: str, label_x: str,product_names:List[str],plot_name:str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5),dpi=300)
    color = plt.cm.gist_rainbow(np.linspace(0, 1, 22))
    linestyles = ["-",'-.',":","--",]
    for i in range(len(product_names)):
        product = product_names[i]
        ax.plot(pr_curves[product]['Recall'], pr_curves[product]['Precision'], lw=0.75,label="{0}".format(product),c=color[i],linestyle=linestyles[i%4])
        default_th = pr_curves[product].loc[min(pr_curves[product].index.tolist(), key=lambda x:abs(x-0.5))]
        ax.scatter(default_th["Recall"],default_th["Precision"],s=15,marker='x',color=color[i])     
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    #plt.title(plot_title)
    plt.legend(loc=(1.02,0),prop={'size': 6.05})
    plt.savefig(plot_name,bbox_inches='tight')
    plt.show()
    
def check_if_test_data_has_ones_for_label(test_data: np.array,
                                          label_index: int) -> bool:
    return np.any(test_data.transpose()[label_index])

def calculate_pr_auc(y_pred: np.array,y_test: np.array,product_names:List[str]) -> Dict:
    n_classes = y_test.shape[1]
    precision = dict()
    recall = dict()
    threshold = dict()
    precision_recall_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(
            y_test[:, i], y_pred[:, i])
        precision_recall_auc[product_names[i]] = auc(np.sort(precision[i]),
                                      np.sort(recall[i]))

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_pred.ravel())
    precision_recall_auc["micro"] = auc(np.sort(precision["micro"]),
                                        np.sort(recall["micro"]))
    return pd.DataFrame(precision_recall_auc.values(),index=precision_recall_auc.keys())

def plot_and_export_pr_tradeoffs_micro_average(y_pred: np.array,y_test: np.array,approach_name:str) ->None:
    pr_tradesoffs_micro = pd.DataFrame(precision_recall_curve(y_test.ravel(), y_pred.ravel()),index=['Precision','Recall',"Threshold"]).T.set_index("Threshold")
    pr_tradesoffs_micro.to_pickle(f"{approach_name}_pr_tradesoffs_micro_avg.pkl")
    fig, ax = plt.subplots(dpi=300)
    pr_tradesoffs_micro[['Precision','Recall']].set_index("Recall").plot(ax=ax)
    default_th = pr_tradesoffs_micro.loc[min(pr_tradesoffs_micro.index.tolist(), key=lambda x:abs(x-0.5))]
    ax.scatter(default_th["Recall"],default_th["Precision"],s=25,marker='x',color='black')



