import pandas as pd
import os
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile

def extract_features(input_folder: str):
    # ekstrak mfcc dan filter bank fitur dari file suara .wav
  
    # get train_df and test_test
    df = pd.DataFrame(columns=[ "label","file_name", "mfcc", "filter_bank"])

    # parsing melalui direktori input
    for folder in os.listdir(input_folder):
        # mengambil seluruh path dari folder
        full_path = os.path.join(input_folder, folder)

        # jika tidak ada folder, tolak
        if not os.path.isdir(full_path):
            continue

      
        label = full_path[full_path.rfind('/') + 1:]

        # pengulangan dalam folder mad
        for file_name in os.listdir(full_path):
            if not file_name.endswith('.wav'):
                continue
                
            file_path = os.path.join(full_path, file_name)

            #ekstaksi fitur
            sampling_freq, audio = wavfile.read(file_path)
            mfcc_features = mfcc(audio, sampling_freq, nfft=1103)
            filterbank_features = logfbank(audio, sampling_freq, nfft=1103)

            df = df.append({"file_name": file_path, "label": label, "mfcc":mfcc_features, "filter_bank":filterbank_features}, ignore_index= True)
            
    return df