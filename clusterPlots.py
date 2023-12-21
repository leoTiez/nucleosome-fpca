import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np

import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from scipy.spatial.distance import jensenshannon
from scipy.stats.distributions import beta
from scipy.stats import kstest

from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from skfda.exploratory.visualization import FPCAPlot

from sklearn import svm

from loadData import load_transcript_data, load_all_data, create_clusters, smooth
from multivariatePrediction import learn, pred

COLORS = np.array([[0., 0., 1., 1., ], [1., .6, .2, 1., ]])
COLORS_POL2 = np.array([[0., 0., 1., 1.], [1., 1., 0., 1.]])
COLORS_MED = np.array([[.5, .5, 0., 1.], [0., .5, 1., 1.]])
COLORS_STH = np.array([[.1, .5, .8, 1.], [.8, .5, 0., 1.]])
COLORS_SIZE = np.array([[1., .0, 1., 1.], [.5, 1., 0., 1.]])
COLORS_SEQ = np.array([[.0, .5, .5, 1.], [1., .5, .5, 1.]])
COLORS_NDR_SIZE = np.array([[.0, .7, .7, 1.], [1., .5, 1., 1.]])
COLORS_ORIENTATION = np.array([[1., 1., 0., 1.], [0., 0., .0, 1.]])


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
    ap.add_argument('--all_cond_one_plot', action='store_true', dest='all_cond_one_plot',
                    help='If set, plot all other nuclear properties in single plot.')
    ap.add_argument('--do_equalize', action='store_true', dest='do_equalize',
                    help='If set, clusters are equalised in size.')
    ap.add_argument('--merge_replicate', type=str, default='A',
                    help='Define the replicate for which all cluster scatter plots are set side by side in one figure. '
                         'Choose between A | B | none')
    ap.add_argument('--use_all_cluster', action='store_true', dest='use_all_cluster',
                    help='If set, use clustering that was obtained using all transcripts. '
                         'Only changes behaviour if setup is not all.')
    ap.add_argument('--dir_path', type=str, default=None,
                    help='If set, load profiles saved in this directory. Otherwise, take default which is linked to '
                         'the passed setup.')
    return ap.parse_args(args)


def create_cluster_hist(
        data_profiles: DataFrame,
        save_fig: bool = True,
        save_prefix: str = '',
        save_dir: str = ''
):
    plt.figure(figsize=(8, 7))
    for data_name, dp in data_profiles.items():
        plt.hist(dp['cluster'], bins=2, histtype='step')
    plt.xticks([1.25, 1.75], [1, 2])
    plt.title('Cluster distribution')
    if save_fig:
        Path('figures/cluster/hist/%s' % save_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig('figures/cluster/hist/%s%s_cluster_hist.png' % (save_dir, save_prefix))
        plt.close('all')
    else:
        plt.show()


def pearson_plot(
        data_profiles: DataFrame,
        n_bins: int = 50,
        n_repeat: int = 500,
        p_thresh: float = 5e-2,
        n_subsample: int = 500,
        save_fig: bool = True,
        save_prefix: str = '',
        save_dir: str = ''
):
    drop_list = [
        'cluster',
        'Pol2',
        'Sth',
        'Med',
        'AT',
        'GC',
        'ORF',
        'size',
        'orientation',
        'chr',
        'start',
        'end',
        'NDR length'
    ]
    for data_name, dp in data_profiles.items():
        print('\t%s' % data_name)
        n_c1 = np.sum(dp['cluster'] == 1)
        n_c2 = np.sum(dp['cluster'] == 2)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        pearson = np.corrcoef(dp.drop(drop_list, axis=1).to_numpy())
        cluster_correlation = pearson[dp['cluster'] == 1][:, dp['cluster'] == 2].reshape(-1)
        hist_cluster, _, _ = ax[0].hist(
            np.abs(cluster_correlation),
            bins=n_bins,
            histtype='step',
            density=True,
            label='Cluster',
            color='purple',
            lw=2,
            alpha=.8
        )
        ax[1].plot(
            np.sort(np.abs(cluster_correlation)),
            np.cumsum(np.sort(np.abs(cluster_correlation))) / np.sum(np.abs(cluster_correlation)), 
            label='Cluster',
            alpha=0.8,
            color='purple',
            lw=2
        )
        ax[0].set_title('Pearson correlation between cluster', fontsize=16)
        ax[0].set_xlabel('Absolute Pearson coefficient', fontsize=12)
        ax[0].set_ylabel('#Values', fontsize=12)

        ax[1].set_title('CDF Pearson correlation', fontsize=16)
        ax[1].set_xlabel('Absolute Pearson coefficient', fontsize=12)
        ax[1].set_ylabel('CDF', fontsize=12)

        ks_pvalues = []
        for i in range(n_repeat):
            rand_cluster_idc = np.random.choice(
                [1, 2],
                p=[n_c1 / (n_c1 + n_c2), n_c2 / (n_c1 + n_c2)],
                size=len(dp.index),
                replace=True
            )
            rand_cluster = pearson[rand_cluster_idc == 1][:, rand_cluster_idc == 2].reshape(-1)
            cluster_subsample = np.random.choice(cluster_correlation.size, replace=False, size=n_subsample)
            rand_cluster_subsample = np.random.choice(rand_cluster.size, replace=False, size=n_subsample)
            ks_pvalues.append(kstest(
                np.abs(cluster_correlation)[cluster_subsample],
                np.abs(rand_cluster)[rand_cluster_subsample],
                alternative='greater'
            ).pvalue)

        ks_pvalues = np.asarray(ks_pvalues)
        print('P-value %.9f' % np.mean(ks_pvalues))
        random_pearson = np.sort(np.abs(rand_cluster))
        random_cdf = np.cumsum(np.sort(np.abs(rand_cluster))) / np.sum(np.abs(rand_cluster))
        ax[0].hist(np.abs(rand_cluster), bins=n_bins, density=True, histtype='step', color='orange', lw=2, alpha=.8, label='Random')
        ax[1].plot(random_pearson, random_cdf, color='orange', lw=2, alpha=.8, label='Random')
        ax[0].legend(loc='upper right', fontsize=12)
        ax[1].legend(loc='upper left', fontsize=12)
        fig.suptitle('Pearson clusters %s' % data_name.replace('_', ' '))
        fig.tight_layout()

        if save_fig:
            Path('figures/cluster/sig/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/cluster/sig/%s%s_sig_%s.png' % (save_dir, save_prefix, data_name))
            plt.close(fig)
        else:
            fig.show()


def plot_profile_heatmap(
        data_profiles: Dict[str, DataFrame],
        cmap: str = 'copper',
        save_fig: bool = True,
        save_prefix: str = '',
        save_dir: str = '',
):
    drop_list = [
        'cluster',
        'Pol2',
        'Sth',
        'Med',
        'AT',
        'GC',
        'ORF',
        'size',
        'orientation',
        'chr',
        'start',
        'end',
        'NDR length'
    ]
    for data_name, dp in data_profiles.items():
        print('\t%s' % data_name)
        dp_array = dp.drop(drop_list, axis=1).to_numpy()
        cluster_mask = dp['cluster'] == 1
        plt.figure(figsize=(3, 2.7))
        plt.imshow(dp_array[cluster_mask], cmap=cmap, aspect='auto')
        plt.xlabel('Position', fontsize=10)
        plt.ylabel('Genes (%s)' % np.sum(cluster_mask), fontsize=10)
        plt.xticks(np.arange(0, 1201, 200), [-200, +1, 200, 400, 600, 800, 1000], fontsize=8)
        plt.yticks([])
        plt.title('Heatmap Cluster 1', fontsize=12)
        plt.tight_layout()
        if save_fig:
            Path('figures/cluster/heatmap/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/cluster/heatmap/%s%s_heatmap1_%s.png' % (save_dir, save_prefix, data_name), dpi=300)
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(3, 2.7))
        plt.imshow(dp_array[~cluster_mask], cmap=cmap, aspect='auto')
        plt.xlabel('Position', fontsize=10)
        plt.ylabel('Genes (%s)' % np.sum(~cluster_mask), fontsize=10)
        plt.xticks(np.arange(0, 1201, 200), [-200, +1, 200, 400, 600, 800, 1000], fontsize=8)
        plt.yticks([])
        plt.title('Heatmap Cluster 2', fontsize=12)
        plt.tight_layout()
        if save_fig:
            Path('figures/cluster/heatmap/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/cluster/heatmap/%s%s_heatmap2_%s.png' % (save_dir, save_prefix, data_name), dpi=300)
            plt.close()
        else:
            plt.show()


def plot_clustering(
        scores: np.ndarray,
        profiles: DataFrame,
        name: str,
        n_bins: bool = 100,
        save_fig: bool = True,
        save_prefix: str = '',
        make_single: bool = True,
        cluster_ax: plt.Axes = None,
        boundaries=(-20, 20),
        do_remove: bool = False,
        save_dir: str = ''
):
    def make_score_plot(ax_score, clust, c, w, b, line_col='grey'):
        ax_score.scatter(*scores.T, color=c[clust], marker='.')
        condition = w is not None and b is not None

        if not condition and not do_remove:
            raise ValueError('weight and bias have not been set')

        if do_remove:
            condition = condition and not any([name.replace('Ocampo_', '').replace('_A', '').replace(
                '_B', '').replace('_', ' ').lower() == r for r in remove])

        if condition:
            if line_col == 'grey':
                peffect = [pe.Stroke(linewidth=8, foreground='black'), pe.Normal()]
            else:
                peffect = []
            ax_score.plot(
                np.arange(*boundaries),
                w * np.arange(*boundaries) + b,
                color=line_col,
                path_effects=peffect,
                linestyle='--',
                lw='4'
            )

        ax_score.set_ylim(boundaries)
        ax_score.set_xlim(boundaries)

    def make_hist(ax1, ax2, score1_c1, score1_c2, score2_c1, score2_c2, c):
        hist11, bins11 = np.histogram(score1_c1, bins=n_bins, density=True, range=boundaries)
        hist12, bins12 = np.histogram(score1_c2, bins=n_bins, density=True, range=boundaries)
        hist21, bins21 = np.histogram(score2_c1, bins=n_bins, density=True, range=boundaries)
        hist22, bins22 = np.histogram(score2_c2, bins=n_bins, density=True, range=boundaries)
        hist11, hist12, hist21, hist22 = smooth(hist11, hist12, hist21, hist22)
        ax1.plot(bins11[:-1] + (bins11[1] - bins11[0]), hist11, color=c[0])
        ax1.plot(bins12[:-1] + (bins12[1] - bins12[0]), hist12, color=c[1])
        ax1.set_xlim(boundaries)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('JS: %.3f' % jensenshannon(hist11, hist12), fontsize=14)

        if make_single:
            ax2.plot(hist21, bins21[:-1] + (bins21[1] - bins21[0]), color=c[0])
            ax2.plot(hist22, bins22[:-1] + (bins22[1] - bins22[0]), color=c[1])
            ax2.yaxis.set_label_position('right')
        else:
            ax2.plot(bins21[:-1] + (bins21[1] - bins21[0]), hist21, color=c[0])
            ax2.plot(bins22[:-1] + (bins22[1] - bins22[0]), hist22, color=c[1])
        ax2.set_ylim((-20, 20))
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.set_ylabel('JS: %.3f' % jensenshannon(hist21, hist22), rotation=270, labelpad=20, fontsize=14)

    did_flip = False
    if np.sum(scores[profiles['cluster'] == 1].T[1]) <= np.sum(scores[profiles['cluster'] == 2].T[1]):
        colors = np.flip(COLORS, axis=0)
        did_flip = True
    else:
        colors = COLORS

    if not make_single:
        fig, ax = plt.subplots(4, 2, figsize=(6, 11))
        fig_hist, ax_hist = plt.subplots(4, 2, figsize=(8, 11))

    all_clusters = [
        profiles['cluster'].to_numpy() - 1,
        profiles['Pol2'].to_numpy(),
        profiles['Med'].to_numpy(),
        profiles['Sth'].to_numpy(),
        profiles['size'].to_numpy(),
        profiles['AT'].to_numpy(),
        profiles['NDR length'].to_numpy(),
        profiles['orientation'].to_numpy()
    ]
    all_colors = [
        colors,
        COLORS_POL2,
        COLORS_MED,
        COLORS_STH,
        COLORS_SIZE,
        COLORS_SEQ,
        COLORS_NDR_SIZE,
        COLORS_ORIENTATION
    ]
    all_names = [
        'Cluster',
        'Pol2',
        'Med',
        'Sth1',
        'Size',
        'AT content',
        'NDR size',
        'Orientation'
    ]
    handle_names = [
        ['cluster 1', 'cluster 2'],
        ['Low pol 2', 'High pol 2'],
        ['Low med', 'High med'],
        ['Low sth1', 'High sth1'],
        ['Short', 'Long'],
        ['Low AT', 'High AT'],
        ['Short', 'Long'],
        ['Opposite', 'Tandem'],
    ]
    remove = ['wt', 'isw1', 'isw2', 'isw1 chd1', 'rsc8 chd1', 'isw1 isw2', 'isw1 isw2 chd1']
    weight, bias = None, None
    for i_num, (cluster, cl, n, hn) in enumerate(zip(all_clusters, all_colors, all_names, handle_names)):
        if make_single:
            fig_c = plt.figure(figsize=(6, 6), layout="constrained")
            spec_c = fig_c.add_gridspec(4, 4)
            ax_score_c = fig_c.add_subplot(spec_c[:-1, :-1])
            ax_hist_c1 = fig_c.add_subplot(spec_c[-1, :-1])
            ax_hist_c2 = fig_c.add_subplot(spec_c[:-1, -1])
        else:
            ax_score_c = ax[i_num // 2, i_num % 2]
            ax_hist_c1 = ax_hist[i_num // 2, i_num % 2]
            ax_hist_c2 = ax_hist[i_num // 2, i_num % 2].twinx().twiny()

        if n.lower() == 'cluster':
            clf = svm.LinearSVC().fit(scores, cluster)
            weight, bias = -clf.coef_[..., 0] / clf.coef_[..., 1], -clf.intercept_ / clf.coef_[..., 1]

        make_score_plot(
            ax_score_c,
            clust=cluster,
            c=cl,
            w=weight,
            b=bias,
            line_col='black' if n.lower() == 'cluster' else 'grey'
        )
        if cluster_ax is not None:
            if n.lower() == 'cluster':
                make_score_plot(
                    cluster_ax,
                    clust=cluster,
                    c=cl,
                    w=weight,
                    b=bias,
                    line_col='black' if n.lower() == 'cluster' else 'grey'
                )

        make_hist(
            ax_hist_c1,
            ax_hist_c2,
            scores.T[0][cluster == 0],
            scores.T[0][cluster == 1],
            scores.T[1][cluster == 0],
            scores.T[1][cluster == 1],
            c=cl
        )
        handles = [Line2D([0], [0], color=c, lw=4, marker='.') for cnum, c in enumerate(cl)]
        if n.lower() == 'cluster':
            error = 1. - clf.score(scores, cluster)
        else:
            discrim_w, _ = learn(cluster.reshape(-1, 1), all_clusters[0])
            error = np.mean((pred(cluster.reshape(-1, 1), discrim_w, 0.) - all_clusters[0])**2)

        if make_single:
            ax_score_c.set_title(n, fontsize=18)
        else:
            ax_score_c.set_title(n, fontsize=18)
        if n.lower() == 'cluster':
            path_effects = []
            color = 'black'
        else:
            path_effects = [pe.Stroke(linewidth=4, foreground='black'), pe.Normal()]
            color = 'grey'
        ax_score_c.set_xlabel('fPC 1', fontsize=18)
        ax_score_c.set_ylabel('fPC 2', fontsize=18)
        ax_score_c.set_xlim((-20., 20))
        ax_score_c.set_ylim((-20., 20))
        ax_score_c.set_xticks([])
        ax_score_c.set_yticks([])

        if not make_single:
            ax_hist_c1.legend(handles, hn)
            ax_hist_c1.set_title(n)
            ax_hist_c1.set_xlabel('fPC 1')
            ax_hist_c1.set_ylabel('fPC 2')
            ax_hist_c1.set_xlim((-20., 20))
            ax_hist_c1.set_ylim((-20., 20))
        if make_single:
            fig_c.suptitle(name.replace('_', ' '), fontsize=21)
            fig_c.tight_layout()
            handles.append(Line2D([0], [0], color=color, path_effects=path_effects, linestyle='--'))
            hn.append('%.3fx + %.3f, Error: %.1f%%' % (weight, bias, error * 100.))
            if save_fig:
                fig_leg = plt.figure(figsize=(4.5, 1.2))
                fig_leg.legend(handles, hn, fontsize=14, loc='center')
                Path('figures/cluster/fpc_cluster/%s' % save_dir).mkdir(exist_ok=True, parents=True)
                fig_c.savefig('figures/cluster/fpc_cluster/%s%s_fpc_scores_%s_single_%s.png'
                              % (save_dir, save_prefix, name, n))
                fig_leg.savefig('figures/cluster/fpc_cluster/%s%s_fpc_scores_%s_single_%s_legend.png'
                                % (save_dir, save_prefix, name, n))
                plt.close(fig_c)
                plt.close(fig_leg)
            else:
                ax_score_c.legend(handles, hn, fontsize=14)
                fig_c.show()

    if not make_single:
        if cluster_ax is None:
            fig.suptitle('fPCA Scores %s' % name.replace('_', ' '))
            fig_hist.suptitle('fPCA Histograms %s' % name.replace('_', ' '))
            fig.tight_layout()
            fig_hist.tight_layout()

            if save_fig:
                Path('figures/cluster/fpc_cluster/%s' % save_dir).mkdir(exist_ok=True, parents=True)
                fig.savefig('figures/cluster/fpc_cluster/%s%s_fpc_scores_%s.png' % (save_dir, save_prefix, name))
                fig_hist.savefig('figures/cluster/fpc_cluster/%s%s_fpc_hist_%s.png' % (save_dir, save_prefix, name))
                plt.close(fig)
                plt.close(fig_hist)
            else:
                fig_hist.show()
                fig.show()

    return (
        weight,
        bias,
        did_flip
    ) if not do_remove else (None, None, None)


def fpca_plot(
        data_profiles: Dict[str, DataFrame],
        n_basis: int = 20,
        save_fig: bool = True,
        save_prefix: str = '',
        make_single: bool = True,
        merge_replicate: str = '_A',
        do_remove: bool = False,
        save_dir: str = ''
):
    bspline_basis = BSplineBasis(n_basis=n_basis)
    drop_list = [
                'cluster',
                'Pol2',
                'Sth',
                'Med',
                'AT',
                'GC',
                'ORF',
                'size',
                'orientation',
                'chr',
                'start',
                'end',
                'NDR length'
            ]
    if merge_replicate != 'none':
        merge_fig, merge_ax = plt.subplots(4, 4, figsize=(8, 9))
        ax_dict = {
            'wt': (0, 0),
            'chd1': (0, 1),
            'isw1': (0, 2),
            'isw2': (0, 3),
            'rsc8': (1, 0),
            'isw1_chd1': (1, 1),
            'isw2_chd1': (1, 2),
            'rsc8_chd1': (1, 3),
            'isw1_isw2': (2, 0),
            'isw1_rsc8': (2, 1),
            'isw2_rsc8': (2, 2),
            'isw1_isw2_chd1': (2, 3),
            'isw1_rsc8_chd1': (3, 0),
            'isw2_rsc8_chd1': (3, 1),
            'isw1_isw2_rsc8': (3, 2),
            'isw1_isw2_rsc8_chd1': (3, 3),
        }
    for data_name, dp in data_profiles.items():
        print('\t%s' % data_name)
        x_scale = np.arange(dp.drop(drop_list, axis=1).to_numpy().shape[1])
        dp_fd = FDataGrid(
            data_matrix=dp.drop(drop_list, axis=1).to_numpy(),
            grid_points=x_scale,
        )
        bspline_data = dp_fd.to_basis(bspline_basis)
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        ax[0].plot(dp.drop(drop_list, axis=1).to_numpy().T)
        bspline_data.plot(chart=ax[1])
        bspline_data.mean().plot(chart=ax[2], linestyle='--')
        fig.suptitle('MNase profiles %s' % data_name.replace('_', ' '))
        ax[0].set_title('Original')
        ax[1].set_title('B-spline')
        ax[2].set_title('Mean Bspline')
        ax[0].set_xlabel('Position')
        ax[1].set_xlabel('Position')
        ax[2].set_xlabel('Position')
        ax[0].set_ylabel('Amplitude')
        ax[1].set_ylabel('Amplitude')
        ax[2].set_ylabel('Amplitude')
        plt.tight_layout()
        if save_fig:
            Path('figures/cluster/bspline/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/cluster/bspline/%s%s_bspline_profile_%s.png' % (save_dir, save_prefix, data_name))
            plt.close(fig)
        else:
            fig.show()

        fpca = FPCA(n_components=2)
        fpc_scores = fpca.fit_transform(bspline_data)
        print('Number of data points that are not shown: %d' % np.any(np.abs(fpc_scores) > 20, axis=1).sum())

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        fpca.components_.plot(chart=ax)
        fig.suptitle('fPCA all profiles', fontsize=21)
        ax.set_xlabel('Position', fontsize=16)
        ax.set_ylabel('Amplitude', fontsize=16)
        fig.legend(['fPC 1', 'fPC 2'], fontsize=16)
        ax.set_xticks(np.arange(0, 1201, 200))
        ax.set_yticks([])
        ax.set_xticklabels([-200, '+1', 200, 400, 600, 800, 1000])
        ax.tick_params(axis='x', labelsize=14)
        fig.tight_layout()
        if save_fig:
            Path('figures/cluster/fpc/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            fig.savefig('figures/cluster/fpc/%s%s_fpc_%s.png' % (save_dir, save_prefix, data_name))
            plt.close(fig)
        else:
            fig.show()

        fig = plt.figure(figsize=(6, 2 * 3))
        FPCAPlot(
            bspline_data.mean(),
            fpca.components_,
            factor=20,
            fig=fig,
            n_rows=2,
        ).plot()
        ax_list = fig.axes
        ax_list[0].set_title('fPC 1', fontsize=18)
        ax_list[1].set_title('fPC 2', fontsize=18)
        for a in ax_list:
            a.set_yticks([])
            a.set_xticks(np.arange(0, 1201, 200))
            a.set_xticklabels([-200, '+1', 200, 400, 600, 800, 1000])
            a.tick_params(axis='x',  labelsize=14)
            a.get_lines()[0].set_color('black')
            a.get_lines()[0].set_linestyle('--')
            a.get_lines()[0].set_linewidth(2)
            a.get_lines()[1].set_color('magenta')
            a.get_lines()[1].set_linewidth(2)
            a.get_lines()[2].set_color('limegreen')
            a.get_lines()[2].set_linewidth(2)
        fig.suptitle('%s' % data_name.replace('_', ' '), fontsize=21)
        fig.tight_layout()

        if save_fig:
            Path('figures/cluster/fpc_effect/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            fig.savefig('figures/cluster/fpc_effect/%s%s_fpc_effect_%s.png' % (save_dir, save_prefix, data_name))
            plt.close(fig)
        else:
            fig.show()

        cluster_ax = None
        if merge_replicate in data_name:
            sample_name = data_name.replace(merge_replicate, '').replace('Ocampo_', '').lower()
            try:
                cluster_ax = merge_ax[ax_dict[sample_name]]
                cluster_ax.set_title(sample_name.replace('_', ' '), fontsize=16)
                cluster_ax.set_xticks([])
                cluster_ax.set_yticks([])
            except:
                cluster_ax = None

        (
            weight,
            bias,
            did_flip
        ) = plot_clustering(
            fpc_scores,
            dp,
            data_name,
            save_fig=save_fig,
            save_prefix=save_prefix,
            make_single=make_single,
            cluster_ax=cluster_ax,
            do_remove=do_remove,
            save_dir=save_dir
        )
        if weight is None:
            continue
        upper_precision = (bspline_data.mean() + 5. * (
                weight * fpca.components_[0]
                + fpca.components_[1]
        ) / (np.abs(weight) + 1.))(x_scale).reshape(-1)

        lower_precision = (bspline_data.mean() - 5. * (
                weight * fpca.components_[0]
                + fpca.components_[1]
        ) / (np.abs(weight) + 1.))(x_scale).reshape(-1)

        cluster_1_scores = np.median(fpc_scores[dp['cluster'] == 1], axis=0)
        cluster_2_scores = np.median(fpc_scores[dp['cluster'] == 2], axis=0)
        average_fpc_c1 = bspline_data.mean() + (
                cluster_1_scores[:1] * fpca.components_[0]  # Need array format for weight
                + cluster_1_scores[1:] * fpca.components_[1]  # Need array format for weight
        )
        average_fpc_c2 = bspline_data.mean() + (
            cluster_2_scores[:1] * fpca.components_[0]  # Need array format for weight
            + cluster_2_scores[1:] * fpca.components_[1]  # Need array format for weight
        )
        colors = np.flip(COLORS, axis=0) if did_flip else COLORS
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        bspline_data.mean().plot(chart=ax, color='black', lw=2, label='Mean')
        ax.plot(lower_precision, color='grey')
        ax.plot(upper_precision, color='grey')
        ax.fill_between(
            x_scale,
            lower_precision,
            upper_precision,
            color='black',
            alpha=.15,
            label='Discriminator'
        )
        average_fpc_c1.plot(chart=ax, color=colors[0], linestyle='--', lw=2, label='Cluster 1')
        average_fpc_c2.plot(chart=ax, color=colors[1], linestyle='--', lw=2, label='Cluster 2')
        ax.set_title('Cluster fPCs', fontsize=21)
        # ax.legend(fontsize=16)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 1201, 200))
        ax.set_xticklabels([-200, '+1', 200, 400, 600, 800, 1000])
        ax.set_xlabel('Position', fontsize=16)
        ax.set_ylabel('Amplitude', fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        fig.tight_layout()
        if save_fig:
            Path('figures/cluster/discriminator/%s' % save_dir).mkdir(exist_ok=True, parents=True)
            fig.savefig('figures/cluster/discriminator/%s%s_discriminator_%s.png' % (save_dir, save_prefix, data_name))
            plt.close(fig)
        else:
            fig.show()

    if save_fig:
        merge_fig.suptitle('%s' % save_prefix.replace('_', ' '))
        merge_fig.tight_layout()
        Path('figures/cluster/fpc_cluster/%s' % save_dir).mkdir(exist_ok=True, parents=True)
        merge_fig.savefig('figures/cluster/fpc_cluster/%s%s_all_cluster.png' % (save_dir, save_prefix))
        plt.close(merge_fig)
    else:
        merge_fig.show()


def main(args):
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    make_single = not args.all_cond_one_plot
    equalize_clusters = args.do_equalize
    use_all_cluster = args.use_all_cluster
    merge_replicate = args.merge_replicate
    if merge_replicate.lower() == 'none':
        merge_replicate = 'none'
    elif merge_replicate.lower() == 'a':
        merge_replicate = '_A'
    elif merge_replicate.lower() == 'b':
        merge_replicate = '_B'
    else:
        raise ValueError('Merge replicate not understood. Choose between A | B | none')
    min_size = None
    max_size = None
    do_remove = False
    save_dir = 'all/' if use_all_cluster else 'size-specific/'
    if args.setup.lower() == 'small':
        dir_path = 'data/mat_small_genes'
        max_size = 1000
        save_dir += 'small/'
        if use_all_cluster:
            do_remove = True
    elif args.setup.lower() == 'large':
        dir_path = 'data/mat_large_genes'
        save_dir += 'large/'
        min_size = 1000
    elif args.setup.lower() == 'very_large':
        dir_path = 'data/mat'
        save_dir += 'very_large/'
        min_size = 3000
    elif args.setup.lower() == 'all':
        dir_path = 'data/mat'
        save_dir += 'all/'
    else:
        raise ValueError('Setup not understood. Choose between small | large | very_large | all')

    # Overwrite path if all_cluster flag is set
    if use_all_cluster:
        dir_path = 'data/mat'
    if args.dir_path is not None:
        dir_path = args.dir_path
    print('Load transcript data')
    transcript_data = load_transcript_data(min_size=min_size, max_size=max_size)
    print('Load nucleosome profiles')
    data_profiles = load_all_data(transcript_data, equalize_clusters=equalize_clusters, dir_path=dir_path)
    print('Plot profile heatmap')
    plot_profile_heatmap(data_profiles, save_dir=save_dir, save_prefix=save_prefix, save_fig=save_fig)
    print('Create cluster histograms')
    create_cluster_hist(data_profiles, save_fig=save_fig, save_prefix=save_prefix, save_dir=save_dir)
    print('Create pearson plots')
    pearson_plot(data_profiles, save_fig=save_fig, save_prefix=save_prefix, save_dir=save_dir)
    print('Perform fpca')
    fpca_plot(
        data_profiles,
        save_fig=save_fig,
        save_prefix=save_prefix,
        make_single=make_single,
        merge_replicate=merge_replicate,
        do_remove=do_remove,
        save_dir=save_dir
    )


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))


