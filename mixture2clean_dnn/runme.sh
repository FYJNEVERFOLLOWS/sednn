#!/bin/bash

MINIDATA=0
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="/Users/fuyanjie/Desktop/Coding/PycharmProjects/SE/sednn_xu/mixture2clean_dnn/workspace"
#  mkdir $WORKSPACE
  TR_SPEECH_DIR="/Users/fuyanjie/Desktop/Coding/PycharmProjects/SE/sednn_xu/mixture2clean_dnn/mini_data/train_speech"
  TR_NOISE_DIR="/Users/fuyanjie/Desktop/Coding/PycharmProjects/SE/sednn_xu/mixture2clean_dnn/mini_data/train_noise"
  TE_SPEECH_DIR="/Users/fuyanjie/Desktop/Coding/PycharmProjects/SE/sednn_xu/mixture2clean_dnn/mini_data/test_speech"
  TE_NOISE_DIR="/Users/fuyanjie/Desktop/Coding/PycharmProjects/SE/sednn_xu/mixture2clean_dnn/mini_data/test_noise"
  echo "Using mini data. "
else
  WORKSPACE="..\metadata_workspace"
  TR_SPEECH_DIR=".\metadata\train_speech"
  TR_NOISE_DIR=".\metadata\train_noise"
  TE_SPEECH_DIR=".\metadata\sub_test_speech"
  TE_NOISE_DIR=".\metadata\test_noise"
  echo "Using full data. "
fi

# Create mixture csv. 
#python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=2
#python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test

# Calculate mixture features.
TR_SNR=0
TE_SNR=0 
#python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
#python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR

# Pack features. 
N_CONCAT=7
N_HOP=3
#python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP
#python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP

# Compute scaler. 
#python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR

# Train. 
LEARNING_RATE=1e-4
#CUDA_VISIBLE_DEVICES=3
#python main_dnn.py train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE

# Plot training stat. 
#python evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=10001 --interval_iter=1000

# Inference, enhanced wavs will be created. 
ITERATION=10000
#CUDA_VISIBLE_DEVICES=3
#python main_dnn.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION --visualize

# Calculate PESQ of all enhanced speech. 
#python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

# Calculate STOI of all enhanced speech.
#python evaluate.py calculate_stoi --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

# Calculate stoi overall stats.
#python evaluate.py get_stats_stoi

# Calculate overall stats. 
#python evaluate.py get_stats
read -p "press enter end"
