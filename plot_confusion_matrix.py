import numpy as np
from plotly.tools import FigureFactory as ff
import pickle
from training import dataset_names
def matrix_to_str(matrix):
    return np.array(matrix).astype(str).copy()

def plot_confusion(confusion_matrix, dataset_names, given_title='title'):
    cf_matrix_list_str = matrix_to_str(confusion_matrix)
    # print(f"cf_matrix_list_str: {cf_matrix_list_str}")
    fig = ff.create_annotated_heatmap(confusion_matrix, x=dataset_names, y=dataset_names,
        annotation_text=cf_matrix_list_str, autocolorscale=True, range_color=[.5,1])
    #https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
    # add title
    fig.update_layout(title_text=("Model: " + given_title),
                      xaxis = dict(title='Validated On'),
                      yaxis = dict(title='Trained On')
                     )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis titl
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.write_image('./Results/' + given_title + '.png')
    fig.show()

def main():
    with open('model_out.pkl', 'wb') as f:
        results = pickle.load(f)
    #Make dictionary into needed list
    matrix = list()
    # model_names = ['bert-large-cased']
    # rows = ["tweebank", "TPANN", "ark"]
    for model_used in results.keys():
        matrix.append(list())
        for dataset_trained_on in results[model_used].keys():
            matrix[-1].append(list())
            for dataset_validated_on, acc in results[model_used][dataset_trained_on].items():
                # print(f"model_used: {model_used}, dataset_trained_on: {dataset_trained_on}, dataset_validated_on: {dataset_validated_on}, acc: {results[model_used][dataset_trained_on][dataset_validated_on]}%")
                matrix[-1][-1].append(acc)

    plot_confusion(matrix[-1], dataset_names, given_title=model_used.split('/')[-1])
    print()
    