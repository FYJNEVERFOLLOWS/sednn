"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: - 
"""
import os
import soundfile
import librosa
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import h5py
from sklearn import preprocessing

import prepare_data as pp_data
import config as cfg


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_mixture_csv(args):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      mixed_dir: str, path of mixed data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    mixed_dir = args.mixed_dir
    data_type = args.data_type
    fs = cfg.sample_rate
    
    speech_names = [na for na in os.listdir(speech_dir)]
    # mixed_names = [na for na in os.listdir(mixed_dir) if na.lower().endswith(".wav")]
    
    rs = np.random.RandomState(0)
    out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    pp_data.create_folder(os.path.dirname(out_csv_path))
    
    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\n" % "speech_name")
    for speech_na in speech_names:
        # Read speech. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        len_speech = len(speech_audio)       

        if cnt % 100 == 0:
            print(cnt)
            
        cnt += 1
        f.write("%s\n" % speech_na) 
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)

def calculate_mixture_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      mixed_dir: str, path of mix data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    mixed_dir = args.mixed_dir    
    data_type = args.data_type
    snr = args.snr
    fs = cfg.sample_rate
    
    # Open mixture csv. 
    mixture_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    with open(mixture_csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
    
    t1 = time.time()
    cnt = 0
    for i1 in range(1, len(lis)):
        [speech_na] = lis[i1]

        
        # Read speech audio. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        
        # Read mix audio. 
        mixed_path = os.path.join(mixed_dir, speech_na)
        (mixed_audio, _) = read_audio(mixed_path, target_fs=fs)



        # Get normalized mixture, speech.         
        alpha = 1. / np.max(np.abs(mixed_audio))
        mixed_audio *= alpha
        speech_audio *= alpha


        # Extract spectrogram. 
        mixed_complx_x = calc_sp(mixed_audio, mode='complex')
        speech_x = calc_sp(speech_audio, mode='magnitude')

        # Write out features. 
        out_feat_path = os.path.join(workspace, "features", "spectrogram", 
            data_type, "%ddb" % int(snr), "%s.p" % os.path.splitext(speech_na)[0])
        create_folder(os.path.dirname(out_feat_path))
        data = [mixed_complx_x, speech_x, alpha, os.path.splitext(speech_na)[0]]
        pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        cnt += 1

    print("Extracting feature time: %s" % (time.time() - t1))

    
def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x
    
###
def pack_features(args):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames. 
    """
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop
    
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)
    
    cnt = 0
    t1 = time.time()
    
    # Load all features. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", data_type, "%ddb" % int(snr))
    names = os.listdir(feat_dir)
    for na in names:
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_x, alpha, na] = data
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values. 
        n_pad = int((n_concat - 1) / 2)
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
    
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x_all.append(mixed_x_3d)
        
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, int((n_concat - 1) / 2), :]
        y_all.append(y)
    
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        # if cnt == 3: break
        cnt += 1
        
    x_all = np.concatenate(x_all, axis=0)   # (n_segs, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
    
    x_all = log_sp(x_all).astype(np.float32)
    y_all = log_sp(y_all).astype(np.float32)
    
    # Write out data to .h5 file. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
    
    print("Write out to %s" % out_path)
    print("Pack features finished! %s s" % (time.time() - t1,))
    
def log_sp(x):
    return np.log(x + 1e-08)
    
def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)

def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)

###
def compute_scaler(args):
    """Compute and write out scaler of data. 
    """
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    
    # Load data. 
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')     
        x = np.array(x)     # (n_segs, n_concat, n_freq)
    
    # Compute scaler. 
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    print(scaler.mean_)
    print(scaler.scale_)
    
    # Write out scaler. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))
    
    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))
    
def scale_on_2d(x2d, scaler):
    """Scale 2D array data. 
    """
    return scaler.transform(x2d)
    
def scale_on_3d(x3d, scaler):
    """Scale 3D array data. 
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d
    
def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data. 
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]
    
###
def load_hdf5(hdf5_path):
    """Load hdf5 data. 
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)        
    return x, y

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
    
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--mixed_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--mixed_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)
    
    args = parser.parse_args()
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)       
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")