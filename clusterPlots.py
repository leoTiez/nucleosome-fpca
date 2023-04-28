import sys
import argparse
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from scipy.spatial.distance import jensenshannon
from scipy.stats.distributions import gamma

from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from skfda.exploratory.visualization import FPCAPlot

from sklearn import svm

from loadData import load_transcript_data, load_all_data, create_clusters, smooth
from multivariatePrediction import learn, pred


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
    ap.add_argument('--do_not_merge', action='store_true', dest='do_not_merge',
                    help='If set, different wt and mutant conditions are not plot together in single figure.')
    ap.add_argument('--use_all_cluster', action='store_true', dest='use_all_cluster',
                    help='If set, use clustering that was obtained using all transcripts. '
                         'Only changes behaviour if setup is not all.')
    return ap.parse_args(args)


def create_cluster_hist(data_profiles, save_fig: bool = True, save_prefix: str = ''):
    plt.figure(figsize=(8, 7))
    for data_name, dp in data_profiles.items():
        plt.hist(dp['cluster'], bins=2, histtype='step')
    plt.xticks([1.25, 1.75], [1, 2])
    plt.title('Cluster distribution')
    if save_fig:
        Path('figures/cluster/hist/').mkdir(exist_ok=True, parents=True)
        plt.savefig('figures/cluster/hist/%s_cluster_hist.png' % save_prefix)
        plt.close('all')
    else:
        plt.show()


def pearson_plot(data_profiles, n_bins: int = 100, n_repeat: int = 500, save_fig: bool = True, save_prefix: str = ''):
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
        hist_c1, _, _ = ax[0].hist(
            pearson[dp['cluster'] == 1].reshape(-1),
            bins=n_bins,
            histtype='step',
            density=True
        )
        hist_c2, _, _ = ax[0].hist(
            pearson[dp['cluster'] == 2].reshape(-1),
            bins=n_bins,
            histtype='step',
            density=True
        )
        ax[0].set_title('Jensen-Shannon: %.4f' % jensenshannon(hist_c1, hist_c2))
        ax[0].set_xlabel('Value')
        ax[0].set_ylabel('#Coefficients')

        js_distances = []
        for i in range(n_repeat):
            rand_cluster = np.random.choice(
                [1, 2],
                p=[n_c1 / (n_c1 + n_c2), n_c2 / (n_c1 + n_c2)],
                size=len(dp.index),
                replace=True
            )
            rand_hist1, _ = np.histogram(pearson[rand_cluster == 1].reshape(-1), bins=100, range=(-1., 1.))
            rand_hist2, _ = np.histogram(pearson[rand_cluster == 2].reshape(-1), bins=100, range=(-1., 1.))
            js_distances.append(jensenshannon(rand_hist1, rand_hist2))

        js_distances = np.asarray(js_distances)
        lower, upper = np.percentile(js_distances, (0., 99.))
        js_distances = js_distances[np.logical_and(js_distances >= lower, js_distances < upper)]
        fit_alpha, fit_loc, fit_beta = gamma.fit(js_distances)
        ax[1].hist(js_distances, bins=100, density=True, label='Random clusters')
        ax[1].plot(
            np.arange(0, .1, 0.001),
            gamma.pdf(np.arange(0, .1, 0.001), fit_alpha, loc=fit_loc, scale=fit_beta),
            label='Gamma fit'
        )
        ax[1].vlines(
            jensenshannon(hist_c1, hist_c2),
            ymin=0.,
            ymax=200,
            linestyle='--',
            color='black',
            label='Cluster JS'
        )
        ax[1].vlines(
            gamma.interval(1. - 5e-2, fit_alpha, loc=fit_loc, scale=fit_beta),
            ymin=0.,
            ymax=200,
            linestyle='--',
            color='tab:orange',
            label='PI 5e-2'
        )
        ax[1].legend()
        ax[1].set_title('Random Jensen-Shannon Distances')
        ax[1].set_xlabel('JS distance')
        ax[1].set_ylabel('#Values')
        fig.suptitle('Pearson Clusters %s' % data_name.replace('_', ' '))
        fig.tight_layout()

        if save_fig:
            Path('figures/cluster/JS/').mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/cluster/JS/%s_random_js_%s.png' % (save_prefix, data_name))
            plt.close(fig)
        else:
            plt.show()


def plot_clustering(
        scores,
        profiles,
        name,
        n_bins: bool = 100,
        save_fig: bool = True,
        save_prefix: str = '',
        make_single: bool = True,
        cluster_ax: plt.Axes = None,
        boundaries=(-20, 20),
):
    def make_score_plot(ax_score, clust, c, w, b, line_col='grey'):
        ax_score.scatter(*scores.T, color=c[clust], marker='.')
        if w is not None and b is not None:
            # and not any([name.replace('Ocampo_', '').replace('_A', '').replace('_B', '').replace('_', ' ').lower()
            # == r for r in remove]):
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
        else:
            # pass
            raise ValueError('weight and bias have not been set')
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
        ax1.set_xlabel('JS distance: %.3f' % jensenshannon(hist11, hist12))

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
        ax2.set_ylabel('JS distance: %.3f' % jensenshannon(hist21, hist22), rotation=270, labelpad=20)

    colors = np.array([[0., 0., 1., 1., ], [1., .6, .2, 1., ]])
    colors_pol2 = np.array([[0., 0., 1., 1.], [1., 1., 0., 1.]])
    colors_med = np.array([[.5, .5, 0., 1.], [0., .5, 1., 1.]])
    colors_sth = np.array([[.1, .5, .8, 1.], [.8, .5, 0., 1.]])
    colors_size = np.array([[1., .0, 1., 1.], [.5, 1., 0., 1.]])
    colors_seq = np.array([[.0, .5, .5, 1.], [1., .5, .5, 1.]])
    colors_orientation = np.array([[1., 1., 0., 1.], [0., 0., .0, 1.]])
    colors_ndr_size = np.array([[.0, .7, .7, 1.], [1., .5, 1., 1.]])

    if np.sum(scores.T[0]) <= np.sum(scores.T[1]):
        colors = np.flip(colors, axis=0)

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
        colors_pol2,
        colors_med,
        colors_sth,
        colors_size,
        colors_seq,
        colors_ndr_size,
        colors_orientation
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
    # remove = ['wt', 'isw1', 'isw2', 'isw1 chd1', 'rsc8 chd1', 'isw1 isw2', 'isw1 isw2 chd1']
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
        handles.append(Line2D([0], [0], color=color, path_effects=path_effects, linestyle='--'))
        hn.append('%.3fx + %.3f, Error: %.1f%%' % (weight, bias, error * 100.))
        ax_score_c.legend(handles, hn)
        ax_score_c.set_xlabel('fPC 1')
        ax_score_c.set_ylabel('fPC 2')
        ax_score_c.set_xlim((-20., 20))
        ax_score_c.set_ylim((-20., 20))
        if not make_single:
            ax_hist_c1.legend(handles, hn)
            ax_hist_c1.set_title(n)
            ax_hist_c1.set_xlabel('fPC 1')
            ax_hist_c1.set_ylabel('fPC 2')
            ax_hist_c1.set_xlim((-20., 20))
            ax_hist_c1.set_ylim((-20., 20))
        if make_single:
            fig_c.suptitle(name.replace('_', ' '), fontsize=21)
            if save_fig:
                Path('figures/cluster/fpc_cluster').mkdir(exist_ok=True, parents=True)
                fig_c.savefig('figures/cluster/fpc_cluster/%s_fpc_scores_%s_single_%s.png' % (save_prefix, name, n))
                plt.close(fig_c)
            else:
                plt.show()

    if not make_single:
        if cluster_ax is None:
            fig.suptitle('fPCA Scores %s' % name.replace('_', ' '))
            fig_hist.suptitle('fPCA Histograms %s' % name.replace('_', ' '))
            fig.tight_layout()
            fig_hist.tight_layout()

            if save_fig:
                Path('figures/cluster/fpc_cluster').mkdir(exist_ok=True, parents=True)
                fig.savefig('figures/cluster/fpc_cluster/%s_fpc_scores_%s.png' % (save_prefix, name))
                fig_hist.savefig('figures/cluster/fpc_cluster/%s_fpc_hist_%s.png' % (save_prefix, name))
                plt.close(fig)
                plt.close(fig_hist)
            else:
                plt.show()


def fpca_plot(
        data_profiles,
        n_basis: int = 20,
        save_fig: bool = True,
        save_prefix: str = '',
        make_single: bool = True,
        merge_conditions: bool = False,
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
    if merge_conditions:
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
        dp_fd = FDataGrid(
            data_matrix=dp.drop(drop_list, axis=1).to_numpy(),
            grid_points=np.arange(dp.drop(drop_list, axis=1).to_numpy().shape[1]),
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
            Path('figures/cluster/bspline').mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/cluster/bspline/%s_bspline_profile_%s.png' % (save_prefix, data_name))
            plt.close(fig)
        else:
            plt.show()

        fpca = FPCA(n_components=2)
        fpc_scores = fpca.fit_transform(bspline_data)

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        fpca.components_.plot(chart=ax)
        fig.suptitle('fPCA all profiles')
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')
        fig.legend(['fPC 1', 'fPC 2'])
        fig.tight_layout()
        if save_fig:
            Path('figures/cluster/fpc').mkdir(exist_ok=True, parents=True)
            fig.savefig('figures/cluster/fpc/%s_fpc_%s.png' % (save_prefix, data_name))
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
        fig.suptitle('%s' % data_name.replace('_', ' '), fontsize=21)
        fig.tight_layout()

        if save_fig:
            Path('figures/cluster/fpc_effect').mkdir(exist_ok=True, parents=True)
            fig.savefig('figures/cluster/fpc_effect/%s_fpc_effect_%s.png' % (save_prefix, data_name))
            plt.close(fig)
        else:
            fig.show()

        cluster_ax = None
        if merge_conditions:
            if '_A' in data_name:
                sample_name = data_name.replace('_A', '').replace('Ocampo_', '').lower()
                try:
                    cluster_ax = merge_ax[ax_dict[sample_name]]
                    cluster_ax.set_title(sample_name.replace('_', ' '), fontsize=16)
                    cluster_ax.set_xticks([])
                    cluster_ax.set_yticks([])
                except:
                    cluster_ax = None

        plot_clustering(
            fpc_scores,
            dp,
            data_name,
            save_fig=save_fig,
            save_prefix=save_prefix,
            make_single=make_single,
            cluster_ax=cluster_ax
        )

    if save_fig:
        merge_fig.suptitle('%s' % save_prefix.replace('_', ' '))
        merge_fig.tight_layout()
        Path('figures/cluster/fpc_cluster').mkdir(exist_ok=True, parents=True)
        merge_fig.savefig('figures/cluster/fpc_cluster/%s_all_cluster.png' % save_prefix)
        plt.close(merge_fig)
    else:
        merge_fig.show()


def main(args):
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    make_single = not args.all_cond_one_plot
    equalize_clusters = args.do_equalize
    use_all_cluster = args.use_all_cluster
    merge_conditions = not args.do_not_merge
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

    # Overwrite path if all_cluster flag is set
    if use_all_cluster:
        dir_path = 'data/mat'
    print('Load transcript data')
    transcript_data = load_transcript_data(min_size=min_size, max_size=max_size)
    print('Load nucleosome profiles')
    data_profiles = load_all_data(transcript_data, equalize_clusters=equalize_clusters, dir_path=dir_path)
    print('Create cluster histograms')
    create_cluster_hist(data_profiles, save_fig=save_fig, save_prefix=save_prefix)
    print('Create pearson plots')
    pearson_plot(data_profiles, save_fig=save_fig, save_prefix=save_prefix)
    print('Perform fpca')
    fpca_plot(
        data_profiles,
        save_fig=save_fig,
        save_prefix=save_prefix,
        make_single=make_single,
        merge_conditions=merge_conditions
    )


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))


