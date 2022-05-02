import numpy as np
import os
from plotly.tools import FigureFactory as ff
import pickle
from training import dataset_names
def matrix_to_str(matrix):
    matrix = np.round(matrix, 2)
    return np.array(matrix).astype(str).copy()

def plot_confusion(confusion_matrix, dataset_names, given_title='title'):
    cf_matrix_list_str = matrix_to_str(confusion_matrix)
    # print(f"cf_matrix_list_str: {cf_matrix_list_str}")
    fig = ff.create_annotated_heatmap(confusion_matrix, x=dataset_names[::-1], y=dataset_names,
        annotation_text=cf_matrix_list_str, colorscale=[
            [0.0, 'rgb(255, 129, 204)'],
            [.6, 'rgb(255, 153, 153)'],
            # [.7, 'rgb(255, 153, 255)'],
            [.8, 'rgb(178, 102, 255)'],
            # [.9, 'rgb(151, 153, 255)'],
            [1.0, 'rgb(178, 255, 200)'],
            ])
    #https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
    # add title
    fig.update_layout(title_text=("Model: " + given_title),
                      xaxis = dict(title='Validated On'),
                      yaxis = dict(title='Trained On')
                     )
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.data[0].update(zmin=0, zmax=100)
    if not os.path.isdir("./Results"):
        os.mkdir("./Results")
    fig.write_image('./Results/' + given_title + '.png')
    fig.write_image(str(given_title) + '_confusion.png')

def main(results):
    #Make dictionary into needed list
    # model_names = ['bert-large-cased']
    # rows = ["tweebank", "TPANN", "ark"]
    print(results.keys())
    for model_used in results.keys():
        matrix = [[0.0 for _ in dataset_names] for _ in dataset_names]
        for dataset_trained_on in results[model_used].keys():
            for dataset_validated_on, acc in results[model_used][dataset_trained_on].items():
                # print(f"model_used: {model_used}, dataset_trained_on: {dataset_trained_on}, dataset_validated_on: {dataset_validated_on}, acc: {results[model_used][dataset_trained_on][dataset_validated_on]}%")
                matrix[dataset_names.index(dataset_trained_on)][dataset_names.index(dataset_validated_on)] = float(acc)

        plot_confusion(matrix[::-1][:], dataset_names, given_title=model_used.split('/')[-1])

if __name__ == "__main__":
    # in_path = 'model_out.pkl'
    in_path = 'psuedolabel_out.pkl'
    with open(in_path, 'rb') as f:
        results = pickle.load(f)
    # results = {'bert-large-cased': {'TPANN': {'tweebank': 77.97622165191432, 'GUM': 84.81560499847608, 'TPANN': 92.44871235268441}, 'GUM': {'tweebank': 78.97135075682186, 'GUM': 97.59219750076197, 'TPANN': 15.102575294631166}, 'tweebank': {'tweebank': 94.23872623474571, 'GUM': 18.11033221578787, 'TPANN': 77.6080314273243}}}
    main(results)
