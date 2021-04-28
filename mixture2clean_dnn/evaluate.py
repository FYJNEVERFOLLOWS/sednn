"""
Summary:  Calculate PESQ and overal stats of enhanced speech. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import argparse
import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import soundfile as sf
from pystoi import stoi

def plot_training_stat(args):
    """Plot training and testing loss. 
    
    Args: 
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files. 
    """
    workspace = args.workspace
    tr_snr = args.tr_snr
    bgn_iter = args.bgn_iter
    fin_iter = args.fin_iter
    interval_iter = args.interval_iter

    tr_losses, te_losses, iters = [], [], []
    
    # Load stats. 
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    for iter in range(bgn_iter, fin_iter, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = pickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])
        
    # Plot
    line_tr, = plt.plot(tr_losses, c='b', label="Train")
    line_te, = plt.plot(te_losses, c='r', label="Test")
    plt.axis([0, len(iters), 0, max(tr_losses)])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(handles=[line_tr, line_te])
    plt.xticks(np.arange(len(iters)), iters)
    plt.savefig(workspace + '/training_process.png', dpi=600)
    plt.show()


def calculate_pesq(args):
    """Calculate PESQ of all enhaced speech. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of clean speech. 
      te_snr: float, testing SNR. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    te_snr = args.te_snr
    
    # Remove already existed file. 
    os.system('rm _pesq_itu_results.txt')
    os.system('rm _pesq_results.txt')
    
    # Calculate PESQ of all enhaced speech. 
    enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr))
    names = os.listdir(enh_speech_dir)

    for (cnt, na) in enumerate(names):
        print(cnt, na)
        enh_path = os.path.join(enh_speech_dir, na)
        
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)
        
        # Call executable PESQ tool. 
        cmd = ' '.join(["pesq", "+16000", speech_path, enh_path])

        if os.system(cmd) != 0:
            print("xxxxxx") # 调用失败时会打印"xxxxxx"

            
def calculate_stoi(args):
    """Calculate STOI of all enhaced speech.

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of clean speech.
      te_snr: float, testing SNR.
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    te_snr = args.te_snr

    # Remove already existed file.
    os.system('rm _stoi_results.txt')

    with open('_stoi_results.txt', 'a') as f:
        # Calculate STOI of all enhaced speech.
        enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr))
        names = os.listdir(enh_speech_dir)

        for (cnt, na) in enumerate(names):
            print(cnt, na)
            enh_path = os.path.join(enh_speech_dir, na)

            speech_na = na.split('.')[0]
            speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)

            clean, fs = sf.read(speech_path)
            denoised, fs = sf.read(enh_path)

            len_clean = len(clean)
            len_denoised = len(denoised)
            if len_denoised < len_clean:
                clean = clean[0: len_denoised]
            elif len_clean < len_denoised:
                denoised = denoised[0: len_clean]

            # Clean and denoised should have the same length, and be 1D
            # stoi requires numpy == 1.15.0
            res = stoi(clean, denoised, fs, extended=False)
            f.write(na + '\t{}\n'.format(res))


def get_stats_stoi(args):
    """Calculate stats of STOI.
    """
    stoi_path = "_stoi_results.txt"
    with open(stoi_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader) # len(lis) = 2521

    stoi_dict = {}
    for i1 in range(0, len(lis) - 1): # [0, 2519]
        li = lis[i1]
        na = li[0]
        stoi = float(li[1])
        noise_type = na.split('.')[1]
        if noise_type not in stoi_dict.keys():
            stoi_dict[noise_type] = [stoi]
        else:
            stoi_dict[noise_type].append(stoi)

    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("Noise", "STOI"))
    print("---------------------------------")
    for noise_type in stoi_dict.keys():
        stois = stoi_dict[noise_type]
        avg_stoi = np.mean(stois)
        std_stoi = np.std(stois)
        avg_list.append(avg_stoi)
        std_list.append(std_stoi)
        print(f.format(noise_type, "%.4f +- %.4f" % (avg_stoi, std_stoi)))
    print("---------------------------------")
    print(f.format("Avg.", "%.4f +- %.4f" % (np.mean(avg_list), np.mean(std_list))))
        
        
def get_stats(args):
    """Calculate stats of PESQ. 
    """
    pesq_path = "_pesq_results.txt"
    with open(pesq_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    pesq_dict = {}
    for i1 in range(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        pesq = float(li[1])
        noise_type = na.split('.')[1]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
        else:
            pesq_dict[noise_type].append(pesq)
        
    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("Noise", "PESQ"))
    print("---------------------------------")
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        print(f.format(noise_type, "%.2f +- %.2f" % (avg_pesq, std_pesq)))
    print("---------------------------------")
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_plot_training_stat = subparsers.add_parser('plot_training_stat')
    parser_plot_training_stat.add_argument('--workspace', type=str, required=True)
    parser_plot_training_stat.add_argument('--tr_snr', type=float, required=True)
    parser_plot_training_stat.add_argument('--bgn_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--fin_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--interval_iter', type=int, required=True)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    
    parser_calculate_pesq = subparsers.add_parser('calculate_stoi')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)

    parser_get_stats = subparsers.add_parser('get_stats_stoi')
    parser_get_stats = subparsers.add_parser('get_stats')
    
    args = parser.parse_args()
    
    if args.mode == 'plot_training_stat':
        plot_training_stat(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    elif args.mode == 'calculate_stoi':
        calculate_stoi(args)
    elif args.mode == 'get_stats_stoi':
        get_stats_stoi(args)
    elif args.mode == 'get_stats':
        get_stats(args)
    else:
        raise Exception("Error!")
