import os
import warnings
from typing import Union

import numpy as np
import pandas as pd
import pyBigWig
from Bio import SeqIO
import re


def calc_sig_better(candidates, mnase_data, sig_nfold=3, is_positive=True):
    index = 0 if is_positive else -1
    sig_better = mnase_data[candidates][index] * sig_nfold < mnase_data[candidates]
    if np.sum(sig_better) > 0:
        return candidates[sig_better][index]
    else:
        return candidates[index]


def load_ndr(
        mnase_file: pyBigWig.pyBigWig,
        chromosome: str,
        start: int,
        direct: str,
        name: str,
        maxd_total: int = 1000,
        mind_nucl: int = 150,
        mind_nfr: int = 75,
        smooth_ws: int = 100,
        scaling: int = 1,
        sig_nfold: int = 3,
        is_p1_tss: bool = True
) -> np.ndarray:
    max_bp = int(mnase_file.chroms(chromosome))
    mnase_data_raw = np.nan_to_num(mnase_file.values(
        chromosome,
        int(np.maximum(start - maxd_total, 0)),
        int(np.minimum(start + maxd_total, max_bp))
    ))

    distance = np.abs(np.maximum(start - maxd_total, 0) - np.minimum(start + maxd_total, max_bp))
    # Smooth data
    mnase_data = np.convolve(mnase_data_raw, np.ones(smooth_ws) / float(smooth_ws), mode='same') * scaling

    # Compute gradient
    grad = np.gradient(mnase_data)
    grad = np.convolve(grad, np.ones(smooth_ws) / float(smooth_ws), mode='same') * scaling

    # Compute maxima and minima
    asign = np.sign(grad)
    descending = np.arange(distance)[(np.roll(asign, 1) - asign) > 0]
    if descending.size < 2:
        warnings.warn(
            'Could not find enough peaks for %s on chromosome %s. Skip.' % (name, chromosome),
            UserWarning
        )
        return np.asarray([np.nan, np.nan])
    if not is_p1_tss:
        ascending = np.arange(distance)[(np.roll(asign, 1) - asign) < 0]
        # Compute position of nfr
        if direct == '+':
            nfr_centre_candidate = ascending[np.logical_and(
                maxd_total + mind_nfr > ascending,
                ascending > 1
            )]
        else:
            nfr_centre_candidate = ascending[np.logical_and(
                maxd_total - mind_nfr < ascending,
                ascending < maxd_total * 2
            )]
        try:
            nfr_centre = nfr_centre_candidate[np.argmin(mnase_data[nfr_centre_candidate])]
        except ValueError:
            warnings.warn(
                'Could not find NFR centre for %s on chromsome %s. Skip.' % (name, chromosome),
                UserWarning
            )
            return np.asarray([np.nan, np.nan])
        # compute p1 and m1
        p1_candidates = descending[descending > nfr_centre]
        if p1_candidates.size == 0:
            p1_candidates = np.asarray([distance - 1])
        try:
            p1 = calc_sig_better(p1_candidates, mnase_data, sig_nfold)
        except IndexError:
            warnings.warn('Could not find +1 for %s on chromsome %s. Skip.' % (name, chromosome), UserWarning)
            return np.asarray([np.nan, np.nan])
        m1_candidates = descending[np.logical_and(descending < nfr_centre, descending < p1 - mind_nucl)]
        if m1_candidates.size == 0:
            m1_candidates = np.asarray([0])
        try:
            m1 = calc_sig_better(m1_candidates, mnase_data, sig_nfold, is_positive=False)
        except IndexError:
            warnings.warn('Could not find -1 for %s on chromosome %s. Skip.' % (name, chromosome), UserWarning)
            return np.asarray([np.nan, np.nan])

    else:
        try:
            p1 = descending[np.argmin(np.abs(descending - maxd_total))]
        except IndexError:
            warnings.warn('Could not find +1 for %s on chromosome %s. Skip.' % (name, chromosome), UserWarning)
            return np.asarray([np.nan, np.nan])
        # Compute position of nfr
        if direct == '+':
            m1_candidates = descending[descending <= p1 - mind_nucl]
            try:
                m1 = calc_sig_better(m1_candidates, mnase_data, sig_nfold, is_positive=False)
            except IndexError:
                warnings.warn('Could not find -1 for %s on chromosome %s. Skip.' % (name, chromosome), UserWarning)
                return np.asarray([np.nan, np.nan])
        else:
            m1_candidates = descending[descending >= p1 + mind_nucl]
            try:
                m1 = calc_sig_better(m1_candidates, mnase_data, sig_nfold)
            except IndexError:
                warnings.warn('Could not find -1 for %s on chromosome %s. Skip.' % (name, chromosome), UserWarning)
                return np.asarray([np.nan, np.nan])

    return np.asarray([m1, p1])


def load_transcript_data(
        ref_genome: str = 'data/ref/SacCer3.fa',  # UCSC
        tss_path: str = 'data/ref/GSE49026_S-TSS.txt',  # Park et al.
        pas_path: str = 'data/ref/GSE49026_S-PAS.txt',  # Park et al.
        pol2_path: str = 'data/seq/L5_18.BT2.SacCer3.RPM.Norm.bw',  # Georges et al.
        med_path: str = 'data/seq/Med14_Med14-Myc_WT.bin25.norm.bw',  # Andre et al.
        sth_path: str = 'data/seq/IP-Myc_Sth1-Myc_MED17.bin25.norm.bw',  # Andre et al.
        mnase_path: str = 'data/seq/MNase_WT_A.bw',  # Andre et al.
        min_size: Union[None, int] = None,
        max_size: Union[None, int] = None
):
    ref_genome = list(SeqIO.parse(ref_genome, 'fasta'))
    tss = pd.read_csv(
        tss_path,
        delimiter='\t',
        usecols=['chr', 'coordinate', 'ORF']
    )
    tss.columns = ['chr', 'start', 'ORF']
    pas = pd.read_csv(
        pas_path,
        delimiter='\t',
        usecols=['coordinate', 'ORF']
    )
    pas.columns = ['end', 'ORF']
    combined_df = pd.merge(left=tss, right=pas, left_on='ORF', right_on='ORF')
    combined_df['size'] = np.abs(combined_df['start'] - combined_df['end'])
    size_mask = np.ones(len(combined_df.index), dtype='bool')
    if min_size is not None:
        size_mask = np.logical_and(size_mask, combined_df['size'] > min_size)
    if max_size is not None:
        size_mask = np.logical_and(size_mask, combined_df['size'] <= max_size)
    combined_df = combined_df[size_mask]
    combined_df['size'] = create_clusters_array(combined_df['size'].to_numpy())
    pol2_bw = pyBigWig.open(pol2_path)
    med_bw = pyBigWig.open(med_path)
    sth_bw = pyBigWig.open(sth_path)
    mnase_bw = pyBigWig.open(mnase_path)

    pol2_data = {'ORF': [], 'Pol2': []}
    med_data = {'ORF': [], 'Med': []}
    sth_data = {'ORF': [], 'Sth': []}
    seq_data = {'ORF': [], 'AT': [], 'GC': []}
    orientation = {'ORF': [], 'orientation': []}
    ndr_length = {'ORF': [], 'NDR length': []}

    for chromosome in np.unique(combined_df['chr']):
        chr_pos = combined_df[combined_df['chr'] == chromosome]
        orientation['ORF'].extend(chr_pos['ORF'].to_list())
        plus_genes = np.where(chr_pos['start'] <= chr_pos['end'])[0]
        minus_genes = np.where(chr_pos['start'] > chr_pos['end'])[0]
        plus_dist = np.abs(chr_pos['start'].to_numpy().reshape(-1, 1)
                           - chr_pos['start'].iloc[plus_genes].to_numpy().reshape(1, -1))
        plus_dist[plus_genes, np.arange(len(plus_genes))] = 999999999999999
        plus_dist = np.min(plus_dist, axis=1)
        minus_dist = np.abs(chr_pos['start'].to_numpy().reshape(-1, 1)
                            - chr_pos['start'].iloc[minus_genes].to_numpy().reshape(1, -1))
        minus_dist[minus_genes, np.arange(len(minus_genes))] = 999999999999999
        minus_dist = np.min(minus_dist, axis=1)
        closest_strand = np.argmin([plus_dist, minus_dist], axis=0)
        direct_nrf = np.zeros(len(chr_pos), dtype='int')
        direct_nrf[plus_genes] = np.where(closest_strand[plus_genes] == 0, 1, 0)
        direct_nrf[minus_genes] = np.where(closest_strand[minus_genes] == 1, 1, 0)
        orientation['orientation'].append(direct_nrf)
        for _, gp in chr_pos.iterrows():
            pol2_data['ORF'].append(gp['ORF'])
            med_data['ORF'].append(gp['ORF'])
            sth_data['ORF'].append(gp['ORF'])
            seq_data['ORF'].append(gp['ORF'])
            ndr_length['ORF'].append(gp['ORF'])
            direct = '+'
            if gp['start'] <= gp['end']:
                pol2_data['Pol2'].append(
                    np.nanmean(pol2_bw.values(gp['chr'], int(gp['start']), int(gp['start']) + 1200))
                )
            else:
                pol2_data['Pol2'].append(
                    np.nanmean(pol2_bw.values(gp['chr'], int(gp['start']) - 1200, int(gp['start'])))
                )
                direct = '-'
            med_data['Med'].append(
                np.nanmean(med_bw.values(gp['chr'], int(gp['start'] - 500), int(gp['start']) + 500))
            )
            sth_data['Sth'].append(
                np.nanmean(sth_bw.values(gp['chr'], int(gp['start'] - 500), int(gp['start']) + 500))
            )
            n_gc, n_at = sequence_composition(gp['chr'], gp['start'], gp['end'], ref_genome)
            seq_data['GC'].append(n_gc)
            seq_data['AT'].append(n_at)

            nucl_pos = load_ndr(mnase_bw, gp['chr'], int(gp['start']), direct, gp['ORF'])
            if np.all(~np.isnan(nucl_pos)):
                ndr_length['NDR length'].append(np.abs(nucl_pos[0] - nucl_pos[1]))
            else:
                ndr_length['NDR length'].append(np.nan)

    pol2_data['Pol2'] = create_clusters_array(np.asarray(pol2_data['Pol2']))
    med_data['Med'] = create_clusters_array(np.asarray(med_data['Med']))
    sth_data['Sth'] = create_clusters_array(np.asarray(sth_data['Sth']))
    seq_data['AT'] = create_clusters_array(np.asarray(seq_data['AT']))
    seq_data['GC'] = create_clusters_array(np.asarray(seq_data['GC']))
    ndr_length['NDR length'] = create_clusters_array(np.asarray(ndr_length['NDR length']))
    orientation['orientation'] = np.concatenate(orientation['orientation'])
    pol2_data = pd.DataFrame(pol2_data)
    med_data = pd.DataFrame(med_data)
    sth_data = pd.DataFrame(sth_data)
    seq_data = pd.DataFrame(seq_data)
    ndr_length = pd.DataFrame(ndr_length)
    orientation = pd.DataFrame(orientation)
    combined_df = pd.merge(combined_df, pol2_data, left_on='ORF', right_on='ORF')
    combined_df = pd.merge(combined_df, med_data, left_on='ORF', right_on='ORF')
    combined_df = pd.merge(combined_df, sth_data, left_on='ORF', right_on='ORF')
    combined_df = pd.merge(combined_df, orientation, left_on='ORF', right_on='ORF')
    combined_df = pd.merge(combined_df, seq_data, left_on='ORF', right_on='ORF')
    combined_df = pd.merge(combined_df, ndr_length, left_on='ORF', right_on='ORF')
    return combined_df


def sequence_composition(chrom, start, end, dna_seq):
    chrom_seq = list(filter(lambda x: x.id == chrom, dna_seq))[0]
    combinations_gc = ['G', 'C']
    combinations_at = ['A', 'T']
    size = float(np.abs(start - end))
    s = start if start <= end else end
    e = end if start <= end else start
    seq = chrom_seq.seq
    num_gc = 0
    num_at = 0
    for gc, at in zip(combinations_gc, combinations_at):
        num_gc += len([m.start() for m in re.finditer(gc, str(seq[s:e]))])
        num_at += len([m.start() for m in re.finditer(at, str(seq[s:e]))])
    return num_gc / size, num_at / size


def load_profile(mat_path: str, equalize_clusters=False):
    nucl_profiles = pd.read_csv(
        mat_path + '/AlignedProfile.txt',
        delimiter='\s+',
        header=None,
        index_col=None
    )
    nucl_profiles['cluster'] = np.loadtxt(mat_path + '/cidx.txt', delimiter='\t').astype('int')
    orf_column = pd.read_csv(mat_path + '/ORF.txt', sep='\t', header=None, names=['ORF'])
    nucl_profiles['ORF'] = orf_column
    if equalize_clusters:
        n_c1, n_c2 = np.sum(nucl_profiles['cluster'] == 1), np.sum(nucl_profiles['cluster'] == 2)
        if n_c1 != n_c2:
            axis = 0 if n_c1 < n_c2 else 1
            dominant_cluster = 1 if n_c1 > n_c2 else 2
            inferior_cluster = 1 if n_c1 < n_c2 else 2
            pearson = np.corrcoef(nucl_profiles.drop(['cluster', 'ORF'], axis=1).to_numpy())
            cluster_sorting = np.argsort(
                np.mean(pearson[nucl_profiles['cluster'] == 1][:, nucl_profiles['cluster'] == 2], axis=axis))
            idc = np.arange(pearson.shape[0], dtype='int')
            idc = idc[nucl_profiles['cluster'] == dominant_cluster]
            idc = idc[cluster_sorting[-np.abs(n_c1 - n_c2) // 2:]]
            nucl_profiles.iloc[idc, nucl_profiles.columns.get_loc('cluster')] = inferior_cluster
    return nucl_profiles.sort_values(by='ORF')


def load_all_data(merge_data: Union[None, pd.DataFrame] = None, dir_path: str = 'data/mat', equalize_clusters=False):
    data = {}
    for root, dirs, files in os.walk(dir_path):
        for dir_name in dirs:
            if 'for' not in dir_name \
                    and 'MACOSX' not in dir_name\
                    and 'BOWTIE.SacCer3.pe.sorted.bam' not in dir_name\
                    and 'Our_wild_type' not in dir_name:
                data[dir_name] = load_profile(os.path.join(root, dir_name), equalize_clusters)
                if merge_data is not None:
                    data[dir_name] = pd.merge(data[dir_name], merge_data, left_on='ORF', right_on='ORF')
    return data


def create_clusters_array(array):
    clusters = np.zeros(array.shape[0], dtype='int')
    clusters[array < np.nanmedian(array)] = 0
    clusters[array >= np.nanmedian(array)] = 1
    clusters[np.isnan(clusters)] = 0
    return clusters


def create_clusters(data_type, profiles):
    clusters = np.zeros(profiles.shape[0], dtype='int')
    clusters[profiles[data_type].to_numpy() < np.nanmedian(profiles[data_type].to_numpy())] = 0
    clusters[profiles[data_type].to_numpy() >= np.nanmedian(profiles[data_type].to_numpy())] = 1
    return clusters


def smooth(*data, sm_size=31):
    if sm_size % 2 == 0:
        sm_size += 1
    sm_window = np.ones(sm_size) / float(sm_size)
    return np.array([np.convolve(d, sm_window, mode='full')[sm_size//2:-sm_size//2 + 1] for d in data])

