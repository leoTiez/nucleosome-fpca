import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

from loadData import load_transcript_data, load_all_data, create_clusters


def parse_args(args):
    ap = argparse.ArgumentParser('Prediction and correlation test between clusters and other nuclear properties')
    ap.add_argument('--save_fig', action='store_true', dest='save_fig',
                    help='If set, figures are saved as file.')
    ap.add_argument('--save_prefix', type=str, default='',
                    help='Identifying prefix for saved figures and files')
    ap.add_argument('--setup', type=str, default='large',
                    help='Gene setup to be considered. Choose between all | very_large | large | small')
    ap.add_argument('--all_clusters', action='store_true', dest='all_clusters',
                    help='If set, base analysis on clusters where all genes were considered.')
    ap.add_argument('--do_multivariate', action='store_true', dest='do_multivariate',
                    help='If set, use pairs to predict clusters.')
    ap.add_argument('--do_not_equalize', action='store_true', dest='do_not_equalize',
                    help='If set, clusters are not equalised in size.')
    return ap.parse_args(args)


def learn(X, y):
    y_bias = y.copy()
    y_bias[y_bias == 0] = -1.
    return (np.sum(X == y.reshape(-1, 1), axis=0) - np.sum(X != y.reshape(-1, 1), axis=0)) / X.shape[0], np.mean(y_bias)


def pred(X, w, b):
    X_in = X.copy()
    X_in[X_in == 0] = -1.
    result = np.sign(w.dot(X_in.T) + b)
    result[result == -1] = 0.
    return result


def run(data_profiles, single_pred=True, save_fig=False, save_prefix=''):
    X_data = [
                'Pol2',
                'Sth',
                'Med',
                'AT',
                'size',
                'orientation',
                'NDR length',
            ]
    target_data = 'cluster'

    for data_name, dp in data_profiles.items():
        print('\t$%s' % data_name)
        corr_data = dp[X_data]
        clusters = dp[target_data].to_numpy() - 1
        errors = []
        names = []
        for i in range(len(X_data) - 1):
            if single_pred:
                X = corr_data[X_data[i]].to_numpy().reshape(-1, 1)
                w, _ = learn(X, clusters)
                names.append('%s' % X_data[i])
                errors.append((pred(X, w, 0.) - clusters) ** 2)
                continue
            for j in range(i + 1, len(X_data)):
                X = np.vstack([
                    corr_data[X_data[i]].to_numpy(),
                    corr_data[X_data[j]].to_numpy(),
                ]).T
                w, _ = learn(X, clusters)
                names.append('%s:%s' % (X_data[i], X_data[j]))
                errors.append((pred(X, w, 0.) - clusters)**2)
        if single_pred:
            X = corr_data[X_data[-1]].to_numpy().reshape(-1, 1)
            w, _ = learn(X, clusters)
            names.append('%s' % X_data[-1])
            errors.append((pred(X, w, 0.) - clusters) ** 2)
        plt.figure(figsize=(8, 4))
        percentage = np.mean(errors, axis=1)
        plt.bar(np.arange(len(names)) + .3, 1. - percentage, 0.4, bottom=0, color='tab:green')
        plt.bar(np.arange(len(names)) + .3, percentage, 0.4, bottom=1. - percentage, color='tab:red')
        plt.plot(np.arange(len(names) + 1), .5 * np.ones(len(names) + 1), linestyle='--', color='black')
        plt.xticks(np.arange(len(names)) + .3, names, rotation=90, fontsize=14)
        plt.title('%s' % data_name.replace('_', ' '), fontsize=21)
        plt.tight_layout()
        if save_fig:
            Path('figures/prediction/').mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/prediction/%s_%s_prediction_%s.png' % (
                save_prefix, data_name, 'single' if single_pred else 'double'))
            plt.close()
        else:
            plt.show()


def main(args):
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    single_pred = not args.do_multivariate
    equalize_clusters = not args.do_not_equalize
    min_size = None
    max_size = None
    if args.setup.lower() == 'small':
        dir_path = 'data/mat_small_genes'
        max_size = 1000
    elif args.setup.lower() == 'large':
        dir_path = 'data/mat_large_genes'
        min_size = 1000
    elif args.setup.lower() == 'very_large':
        dir_path = 'data/mat'
        min_size = 3000
    elif args.setup.lower() == 'all':
        dir_path = 'data/mat'
    else:
        raise ValueError('Setup not understood. Choose between small | large | very_large | all')

    if args.all_clusters:
        dir_path = 'data/mat'

    print('Load transcript data')
    transcript_data = load_transcript_data(min_size=min_size, max_size=max_size)
    print('Load nucleosome profiles')
    data_profiles = load_all_data(transcript_data, equalize_clusters=equalize_clusters, dir_path=dir_path)
    print('Estimate data')
    run(data_profiles, single_pred=single_pred, save_fig=save_fig, save_prefix=save_prefix)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))


