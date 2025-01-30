import mne
import numpy as np
import pywt
import pandas as pd
from scipy import signal
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

quantified_condition = {"normal": [0], "increase": [1]}

def load_eeg_from_folder_basic(folder_path):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.set'):
            file_path = os.path.join(folder_path, file_name)
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            data.append(raw.get_data())  # Extracting data as a NumPy array
            if (condition == "increase"):
                number_condition = 1
            else:
                number_condition = 0
            #labels.append(folder_path.split('/')[-1])  # Label based on folder name
            labels.append(number_condition)
    print("************Length All Data and All Labels*********", len(data), len(labels))
    return data, labels

def load_eeg_from_folder(folder_path, montage='standard_1020'):
    """Loads EEG data, handles missing channels with renaming."""
    data = {"normal": [], "increase": []}
    data_labels = {"normal": [], "increase": []}
    montage = mne.channels.make_standard_montage(montage)
    for condition in ["normal", "increase"]:
        condition_path = os.path.join(folder_path, condition)
        if os.path.exists(condition_path) and os.path.isdir(condition_path):
            for filename in os.listdir(condition_path):
                if filename.endswith(".set"):
                    file_path = os.path.join(condition_path, filename)
                    print("File Path", file_path)

                    try:
                        raw = mne.io.read_raw_eeglab(file_path, preload=True)
                        """
                        # Handle missing channels by renaming
                        original_ch_names = raw.ch_names
                        try:
                            raw.set_montage(montage, on_missing='raise')  # Try setting montage directly
                        except ValueError as e: # Catch if channels are missing
                            print(f"Montage mismatch for {filename}: {e}. Attempting channel renaming.")
                            mapping = {}
                            for orig_ch in original_ch_names:
                                for mont_ch in montage.ch_names:
                                    if orig_ch.upper().replace('Z', 'z').replace('.', '') == mont_ch.upper().replace('Z', 'z').replace('.', ''): #More robust matching
                                        mapping[orig_ch] = mont_ch
                                        break
                            if len(mapping) == len(original_ch_names): #only rename if all channels are found
                                raw.rename_channels(mapping)
                                raw.set_montage(montage, on_missing='raise')
                                print(f"Renamed channels for {filename}")
                            else:
                                print(f"Could not automatically rename all channels for {filename}. Skipping.")
                                continue
                        """
                        data[condition].append(raw)
                        if (condition == "increase"):
                            number_condition = 1
                        else:
                            number_condition = 0
                        #data_labels[condition].append(file_path.split('/')[-1])  # Label based on folder name
                        data_labels[condition].append(number_condition)

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: Subfolder '{condition}' not found in {folder_path}")
    print("************Length All Data and All Labels*********", len(data), len(data_labels))
    return data, data_labels

def preprocess_eeg(raw, l_freq=1.0, h_freq=45.0, notch_freq=50.0):
    """Preprocesses EEG data."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=True)
    raw.notch_filter(notch_freq, verbose=True)
    raw.set_montage('standard_1020', verbose=True)
    raw.set_eeg_reference('average', verbose=True)
    return raw

def wavelet_coherence(data1, data2, wavelet='cmor1.5-1.0'):
    """Computes wavelet coherence."""
    scales = np.arange(1, 128)
    coef1, freqs = pywt.cwt(data1, scales, wavelet)
    coef2, freqs = pywt.cwt(data2, scales, wavelet)
    coherence = np.abs(np.mean(coef1 * np.conj(coef2), axis=1)) / (np.std(coef1, axis=1) * np.std(coef2, axis=1))
    return coherence, freqs

def compute_psd(data, sfreq):
    """Computes Power Spectral Density."""
    freqs, psd = signal.welch(data, sfreq, nperseg=int(sfreq), average='median')
    return freqs, psd

def extract_features(psd, condition, freqs, bands={'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}):
    """Extracts band power features."""
    features = {}
    labels = {}
    for band, (fmin, fmax) in bands.items():
        idx_band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if len(idx_band) > 0:
            band_power = np.sum(psd[idx_band])
            features[band] = band_power
            if (condition == "increase"):
                number_condition = 1
            else:
                number_condition = 0
            labels[band] = number_condition
        else:
            features[band] = 0
            labels[band] = 0
    return features, labels

def extract_features_mean_stddev(data):
    # features = pd.DataFrame()
    features_power = {}
    features_power['mean'] = data.mean(axis=1)
    features_power['std'] = data.std(axis=1)
    # Power in alpha band (8-12 Hz) as an example
    features_power['alpha_power'] = data.apply(lambda x: np.sum(psd[(f >= 8) & (f <= 12)]), axis=1)
    features_power['delta_power'] = data.apply(lambda x: np.sum(psd[(f >= 1) & (f <= 3)]), axis=1)
    features_power['theta_power'] = data.apply(lambda x: np.sum(psd[(f >= 4) & (f <= 7)]), axis=1)
    features_power['beta_power'] = data.apply(lambda x: np.sum(psd[(f >= 13) & (f <= 30)]), axis=1)
    return features_power

folder_path = "/Users/oveedharwadkar/Downloads/stress_data_full"

all_data, all_labels = load_eeg_from_folder(folder_path)
print("Length of All Data and All Labels", len(all_data), len(all_labels))

if all_data:
    results = {"normal": [], "increase": []}
    for condition, eeg_list in all_data.items():
        for raw in eeg_list:
            try:
                raw = preprocess_eeg(raw)
                sfreq = raw.info['sfreq']

                # Example: Use first two channels (check if available)
                if len(raw.ch_names) >= 2:
                    channel1 = raw.ch_names[0]
                    channel2 = raw.ch_names[1]
                    data1 = raw.get_data(picks=channel1)[0]
                    data2 = raw.get_data(picks=channel2)[0]

                    coherence, freqs_wav = wavelet_coherence(data1, data2)
                    freqs_psd, psd = compute_psd(data1, sfreq)
                    features,labels = extract_features(psd, condition, freqs_psd)
                    results[condition].append(features)
                    #all_labels[condition].append(labels)

                    #features_mean_power = extract_features_mean_stddev(data1)
                    #results[condition].append(features_mean_power)

                    # Example Plotting (first patient of each condition)
                    if len(results[condition]) == 1:
                        plt.figure()
                        plt.plot(freqs_wav, coherence)
                        plt.xlabel("Frequency")
                        plt.ylabel("Coherence")
                        plt.title(f"Wavelet Coherence - {condition}")
                        plt.show()

                        plt.figure()
                        plt.plot(freqs_psd, psd)
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Power/Frequency (V^2/Hz)")
                        plt.title(f"Power Spectral Density - {condition}")
                        plt.show()
                else:
                    print(f"Not enough channels for wavelet coherence in a file from {condition}. Skipping")

            except Exception as e:
                print(f"Error processing a file in {condition}: {e}")

    # Convert results to pandas DataFrames
    df_normal = pd.DataFrame(results["normal"])
    df_increase = pd.DataFrame(results["increase"])
    df_labels_normal = pd.DataFrame(all_labels["normal"])
    df_labels_increase = pd.DataFrame(all_labels["increase"])

    print("Normal Data Features:")
    print(df_normal.head())
    print("\nIncreased Data Features:")
    print(df_increase.head())

    print("Normal Labels:")
    print(df_labels_normal.head())
    print("\nIncreased Labels:")
    print(df_labels_increase.head())

    #all_labels = np.array(all_labels)
    #all_labels.reshape(-1,1)
    #results = np.array(results)
    #   results.reshape(-1,1)

    
    # Now you can perform statistical analysis or machine learning

    # Initialize a random forest classifier
    clf = RandomForestClassifier()

    # Sequential Backward Selection
    sbs = SequentialFeatureSelector(clf, direction='backward', n_features_to_select='auto')
    #sbs.fit(results['increase'], all_labels['increase'])  # Assuming 'labels' are provided
    sbs.fit(df_normal, df_labels_normal)  # Assuming 'labels' are provided
    #sbs.fit(df_normal, df_increase)  # Assuming 'labels' are provided
    selected_normal_features = sbs.get_support()
    
    #sbs.fit(df_increase, df_labels_increase)  # Assuming 'labels' are provided
    #selected_increase_features = sbs.get_support()

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(results[selected_normal_features], all_labels, test_size=0.2)

    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression()
    }

    # Training and evaluating classifiers
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
 
else:
    print("Error loading data from folder.")
