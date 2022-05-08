import numpy as np
import os
import pickle
from training import dataset_names
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

def main(results, path):
    if not os.path.isdir("./Results"):
        os.mkdir("./Results")
    print(results.keys())
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.375)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    if len(results.keys()) != 4:
        print("Expected 4 models", file=sys.stderr)
    for model_i, model_used in enumerate(sorted(results.keys())):
        matrix = [[0.0 for _ in dataset_names] for _ in dataset_names]
        for dataset_trained_on in results[model_used].keys():
            for dataset_validated_on, acc in results[model_used][dataset_trained_on].items():
                # print(f"model_used: {model_used}, dataset_trained_on: {dataset_trained_on}, dataset_validated_on: {dataset_validated_on}, acc: {results[model_used][dataset_trained_on][dataset_validated_on]}%")
                i = dataset_names.index(dataset_trained_on)
                j = dataset_names.index(dataset_validated_on)
                matrix[i][j] = float(acc)
        axi = model_i % 2
        axj = model_i // 2
        
        # Plot confusion matrix
        df = pd.DataFrame(matrix,
            index=dataset_names,
            columns=dataset_names)

        heatmap = seaborn.heatmap(df, annot=True,
            fmt='.2f', ax=axs[axi][axj], vmin=0, vmax=100, cbar_ax=cbar_ax)
        heatmap.set_title(model_used)
        if axi==1:
            heatmap.set_xlabel("Validated On")
        if axj==0:
            heatmap.set_ylabel("Trained On")
    plt.savefig(path)
    print("Saved to " +  path)

if __name__ == "__main__":
    # in_path = 'model_out.pkl'
    in_path = 'psuedolabel_out.pkl'
    with open(in_path, 'rb') as f:
        results = pickle.load(f)
    #results = {'bert-large-cased': {'TPANN': {'tweebank': 77.97622165191432, 'GUM': 84.1633648277964, 'TPANN': 92.44871235268441}, 'GUM': {'tweebank': 78.97135075682186, 'GUM': 97.59219750076197, 'TPANN': 75.29463116542995}, 'tweebank': {'tweebank': 94.23872623474571, 'GUM': 92.99603779335568, 'TPANN': 77.6080314273243}}, 'gpt2': {'TPANN': {'tweebank': 73.29143754909663, 'GUM': 80.58518744285279, 'TPANN': 88.21475338280227}, 'GUM': {'tweebank': 74.49594134590207, 'GUM': 93.85553185004571, 'TPANN': 74.29070274989088}, 'tweebank': {'tweebank': 89.41084053417126, 'GUM': 88.40597378847912, 'TPANN': 76.77869925796595}}, 'vinai/bertweet-large': {'TPANN': {'tweebank': 75.74757789997382, 'GUM': 86.40658335873209, 'TPANN': 94.84941073766913}, 'GUM': {'tweebank': 79.40822204765645, 'GUM': 97.48247485522707, 'TPANN': 77.91357485814055}, 'tweebank': {'tweebank': 95.65854935847081, 'GUM': 93.27034440719292, 'TPANN': 81.36185072020952}}, 'roberta-large': {'TPANN': {'tweebank': 75.30243519245876, 'GUM': 86.98567509905517, 'TPANN': 94.7184635530336}, 'GUM': {'tweebank': 78.70123068866195, 'GUM': 96.7997561718988, 'TPANN': 77.91357485814055}, 'tweebank': {'tweebank': 95.26577638125163, 'GUM': 93.27644010972264, 'TPANN': 78.13182016586644}}}
    path = './Results/' + in_path.split('.')[0]
    main(results, path)
