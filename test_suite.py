import os
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import pickle
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from wvmos import get_wvmos
from scipy.stats import gaussian_kde
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from scipy.stats import ttest_ind, f_oneway
import librosa
from sklearn.mixture import GaussianMixture
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_RAVDESS_one_speaker
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts_to_file(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0, prosody_creativity=0.4, architecture ="CFM"):
    
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, architecture=architecture)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor, prosody_creativity=prosody_creativity)
    del tts


def read_text(model_id, sentence, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0, prosody_creativity=0.4, architecture="CFM"):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, architecture=architecture)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    
    phone, durations, pitch, energy = tts.get_prosody_values(text=sentence, duration_scaling_factor=duration_scaling_factor, prosody_creativity=prosody_creativity)
    
    return phone, durations, pitch, energy

def get_wave(model_id, sentence, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    
    wave, sr = tts.get_wave(sentence)
    # Extract pitch (F0) using librosa's pyin function
    f0, voiced_flag, voiced_probs = librosa.pyin(wave, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0, sr


def create_freq_samples(version, model_ids=["Meta"], gpu_id = 1, speaker_reference=None, samples=100):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples", exist_ok=True)
    
    data = {
        'model': [],
        'time': [],
        'pitch': []
        }

    for model_id in model_ids:
        print("Computing Samples for the model: ", model_id)
        for i in tqdm(range(samples)):
        
            f0, sr = get_wave(model_id=model_id,
                            sentence="It snowed, rained, and hailed the same morning.",
                            device=device,
                            speaker_reference=speaker_reference)

            times = librosa.times_like(f0, sr=sr)
            
            # Store the results in the data list
            data['model'].append(model_id)
            data['pitch'].append(pickle.dumps(f0)) # Serialize the array
            data['time'].append(pickle.dumps(times)) # Serialize the array

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Save the DataFrame to disk
    df.to_csv(f"samples/{version}_pitch.csv", index=False)
    print(df.head())
    

def plot_freq(version, path_to_data="samples/test_freq_pitch.csv"):   

    
    df = pd.read_csv(path_to_data) 

    for model, group_df in df.groupby('model'):
        plt.figure(figsize=(10, 4))
        colors = plt.cm.viridis_r(1. * np.arange(len(group_df)) / len(group_df))  # Generate colors
        

        for idx, row in group_df.iterrows():
            row_times = pickle.loads(eval(row['time']))
            f0 = pickle.loads(eval(row['pitch']))

             # Plotting the pitch (F0) contours
            color = colors[idx]  # Get color for this plot
        
            # Plotting the pitch (F0) contours with different colors
            plt.plot(row_times, f0, label=f'Row {idx}', color=color)


        plt.xlabel('Time (s)')
        plt.ylabel('F0 (Hz)')
        plt.title('Pitch Contours over Time')
        plt.ylim([0, 400])  # Adjust as necessary
        plt.grid(True)
        version = version.replace("/", "_")
        model = model.replace("/", "_")
        plt.savefig(f"visualizations/{version}_{model}_freq.png")
        plt.close()



def variance_test(version, dir , model_id="Meta", exec_device="cpu", samples = 40, speaker_reference=None, prosody_creativity=0.4, architecture="CFM"):
    os.makedirs(f"audios/{dir}/{version}", exist_ok=True)
    print("speaker ", speaker_reference)
    for i in tqdm(range(samples)):
        read_texts_to_file(model_id=model_id,
                sentence=["It snowed, rained, and hailed the same morning."],
                filename=f"audios/{dir}/{version}/{version}-{i}.wav",
                device=exec_device,
                language="eng",
                speaker_reference=speaker_reference,
                prosody_creativity=prosody_creativity,
                architecture=architecture,
                duration_scaling_factor=3)





def create_prosody_samples_per_phone(version, model_ids=["Meta"], gpu_id = 1, speaker_reference=None, samples=100):
    #torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples", exist_ok=True)

    # Check if the device is set correctly
    print(f"Using device: {device}")
    data = {
    'model': [],
    'phone': [],
    'pitch': [],
    'energy': [],
    'duration': []
    }
    
    for model_id in model_ids:
        print("Computing Samples for the model: ", model_id)
        for i in tqdm(range(samples)):
            phone, duration, pitch, energy = read_text(model_id=model_id,
                sentence="It snowed, rained, and hailed the same morning.",
                device=device,
                language="eng",
                speaker_reference=speaker_reference)
            for ph, p, e, d in zip(phone, pitch, energy, duration):
                data['model'].append(model_id)
                data['phone'].append(ph)
                data['pitch'].append(p.cpu().numpy())
                data['energy'].append(e.cpu().numpy())
                data['duration'].append(d.cpu().numpy())
    df = pd.DataFrame(data)
    df.to_csv(f"{version}data_samples.csv", index=False)
    print(df.head())

def create_prosody_samples_per_sentence(version, model_ids=["Meta"], device = "cpu", speaker_reference=None, samples=100):
    
    os.makedirs("samples", exist_ok=True)

    # Check if the device is set correctly
    print(f"Using device: {device}")
    data = {
    'model': [],
    'pitch_mean': [],
    'pitch_var': [],
    'energy_mean': [],
    'energy_var': [],
    'duration_mean': [],
    'duration_var': []
    }
    
    for model_id in model_ids:
        print("Computing Samples for the model: ", model_id)
        for i in tqdm(range(samples)):
            phones, durations, pitches, energies = read_text(model_id=model_id,
                sentence="It snowed, rained, and hailed the same morning.",
                device=device,
                language="eng",
                speaker_reference=speaker_reference)
            data['model'].append(model_id)
            data['pitch_mean'].append(pitches.cpu().numpy().mean())
            data['pitch_var'].append(pitches.cpu().numpy().var())
            data['energy_mean'].append(energies.cpu().numpy().mean())
            data['energy_var'].append(energies.cpu().numpy().var())
            data['duration_mean'].append(durations.cpu().numpy().mean())
            data['duration_var'].append(durations.cpu().numpy().var())
    df = pd.DataFrame(data)
    df.to_csv(f"samples/{version}_data_samples_sentence.csv", index=False)
    print(df.head())

def plot_boxplot_per_sentence(version, path_to_data="samples/CFM_testdata_samples.csv"):
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    os.makedirs("visualizations", exist_ok=True)
    
    df = pd.read_csv(path_to_data)
    
    
    # Replace the model names
    df['model'] = df['model'].str.replace('Libri_Prosody/CFM/energy_pitch_duration', 'e-p-d')
    df['model'] = df['model'].str.replace('Libri_Prosody/CFM/pitch_energy_duration', 'p-e-d')


    print(df.head())
    print("")
    
    
    # Melt the DataFrame
    df_melted = df.melt(id_vars=['model'], 
                        value_vars=['pitch_var', 'energy_var', 'duration_var'],
                        var_name='variable', 
                        value_name='value')

    # Create a boxplot for each variable grouped by model
    g = sns.catplot(
        x="model", y="value", col="variable", 
        kind="box", data=df_melted,
        height=4, aspect=0.7, sharey=False
    )
    g.set_axis_labels("Model", "Value")
    g.set_titles("{col_name}") 
    
    # adjust axis
    for ax, variable in zip(g.axes.flat, df_melted['variable'].unique()):
        data = df_melted[df_melted['variable'] == variable]['value']
        ax.set_ylim(0, data.max() * 1.1)


    plt.savefig(f"visualizations/{version}_boxplot.png")

    # Histograms
    g = sns.FacetGrid(df_melted, col="variable", hue="model", sharex=False, sharey=False, height=4, aspect=0.7)
    g.map(sns.histplot, 'value', kde=True, bins=20)
    g.add_legend()
    g.set_titles("{col_name} Histogram")
    plt.savefig(f"visualizations/{version}_histogram.png")

    # Line Plots
    for variable in ['pitch', 'energy', 'duration']:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=df.index, y=variable, hue='model')
        plt.title(f'Line Plot of {variable}')
        plt.xlabel('Index')
        plt.ylabel(variable.capitalize())
        plt.legend(title='Model')
        plt.savefig(f"visualizations/{version}_{variable}_lineplot.png")



    grouped = df.groupby(['model'])
    
    # Calculate statistics
    statistics = grouped.agg(
        pitch_mean=('pitch_var', 'mean'),
        pitch_std=('pitch_var', 'std'),
        pitch_range=('pitch_var', lambda x: x.max() - x.min()),
        energy_mean=('energy_var', 'mean'),
        energy_std=('energy_var', 'std'),
        energy_range=('energy_var', lambda x: x.max() - x.min()),
        duration_mean=('duration_var', 'mean'),
        duration_std=('duration_var', 'std'),
        duration_range=('duration_var', lambda x: x.max() - x.min())
    ).reset_index()

    # Display the results
    pd.set_option('display.max_columns', 10)
    print("#"*72)
    print(statistics)
    print("#"*72)
    print("")
    # Initialize a list to hold the results
    results = []

    # Group by model and perform statistical tests
    models = df['model'].unique()

    if len(models) == 2:  # Perform t-test only if there are exactly two models
        model1_data = df[df['model'] == models[0]]
        model2_data = df[df['model'] == models[1]]

        for metric in ['pitch_var', 'energy_var', 'duration_var']:
            t_stat, p_val = ttest_ind(model1_data[metric], model2_data[metric], equal_var=False)
            significant = p_val < 0.001  # Typical threshold for significance
            results.append({
                'metric': metric,
                'model_1': models[0],
                'model_2': models[1],
                't_stat': t_stat,
                'p_val': p_val,
                'significant': significant
            })
    else:
        for metric in ['pitch_var', 'energy_var', 'duration_var']:
            model_data = [df[df['model'] == model][metric] for model in models]
            f_stat, p_val = f_oneway(*model_data)
            significant = p_val < 0.05  # Typical threshold for significance
            results.append({
                'metric': metric,
                'models': models,
                'f_stat': f_stat,
                'p_val': p_val,
                'significant': significant
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display the results
    print(results_df)
    #plt.savefig(f"visualizations/{version}testing.png")


def get_mean_var(path_to_data, per_sample=False): 
    df = pd.read_csv(path_to_data)
    print(df.head())
    print("")
    
    if per_sample:
        return df["pitch_mean"], df["pitch_var"], df["energy_mean"], df["energy_var"], df["duration_mean"], df["duration_var"]
    
    grouped = df.groupby(['model'])
    # Calculate statistics
    statistics = grouped.agg(
        pitch_mean=('pitch_var', 'mean'),
        pitch_std=('pitch_var', 'std'),
        pitch_range=('pitch_var', lambda x: x.max() - x.min()),
        energy_mean=('energy_var', 'mean'),
        energy_std=('energy_var', 'std'),
        energy_range=('energy_var', lambda x: x.max() - x.min()),
        duration_mean=('duration_var', 'mean'),
        duration_std=('duration_var', 'std'),
        duration_range=('duration_var', lambda x: x.max() - x.min())
    ).reset_index()


    return statistics["pitch_mean"], statistics["pitch_std"], statistics["energy_mean"], statistics["energy_std"], statistics["duration_mean"], statistics["duration_std"]

def plot_boxplot_per_phone(version, path_to_data="samples/CFM_testdata_samples.csv"):
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 63)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    os.makedirs("visualizations", exist_ok=True)


    
    df = pd.read_csv(path_to_data)
    print(df.head())
    
    df_melted = df.melt(id_vars=['model', 'phone'], 
                    value_vars=['pitch', 'energy', 'duration'],
                    var_name='variable', 
                    value_name='value')
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Create a boxplot for each variable grouped by model and phone
    g = sns.catplot(
    x="phone", y="value", col="variable", 
    hue="model", data=df_melted, kind="box",
    height=4, aspect=0.7, sharey=False
    )   
    
    # adjust axis
    for ax, variable in zip(g.axes.flat, df_melted['variable'].unique()):
        data = df_melted[df_melted['variable'] == variable]['value']
        ax.set_ylim(0, data.max() * 1.1)

    # Adjust the layout
    g.set_titles("{col_name}")
    g.set_axis_labels("Phone", "Value")
    g.add_legend(title="Model")

    grouped = df.groupby(['phone', 'model'])

    # Calculate statistics
    statistics = grouped.agg(
        pitch_mean=('pitch', 'mean'),
        pitch_median=('pitch', 'median'),
        pitch_std=('pitch', 'std'),
        pitch_range=('pitch', lambda x: x.max() - x.min()),
        energy_mean=('energy', 'mean'),
        energy_median=('energy', 'median'),
        energy_std=('energy', 'std'),
        energy_range=('energy', lambda x: x.max() - x.min()),
        duration_mean=('duration', 'mean'),
        duration_median=('duration', 'median'),
        duration_std=('duration', 'std'),
        duration_range=('duration', lambda x: x.max() - x.min())
    ).reset_index()

    # Display the results
    print(statistics)
    # Initialize a list to hold the results
    results = []

    # Group by phone
    phones = df['phone'].unique()

    for phone in phones:
        df_phone = df[df['phone'] == phone]
        models = df_phone['model'].unique()
        
        if len(models) == 2:  # Perform t-test only if there are exactly two models
            model1_data = df_phone[df_phone['model'] == models[0]]
            model2_data = df_phone[df_phone['model'] == models[1]]
            
            for metric in ['pitch', 'energy', 'duration']:
                t_stat, p_val = ttest_ind(model1_data[metric], model2_data[metric], equal_var=False)
                significant = p_val < 0.05
                results.append({
                    'phone': phone,
                    'metric': metric,
                    'model_1': models[0],
                    'model_2': models[1],
                    't_stat': t_stat,
                    'p_val': p_val,
                    'significant': significant
                })
        else:
            for metric in ['pitch', 'energy', 'duration']:
                # If there are more than two models, we can use ANOVA
                model_data = [df_phone[df_phone['model'] == model][metric] for model in models]
                f_stat, p_val = f_oneway(*model_data)
                results.append({
                    'phone': phone,
                    'metric': metric,
                    'models': models,
                    'f_stat': f_stat,
                    'p_val': p_val
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display the results
    print(results_df)
    #plt.savefig(f"visualizations/{version}testing.png")

def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id, speaker embedding
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True).float(),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            [datapoint[2] for datapoint in batch],
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            None,
            torch.stack([datapoint[8] for datapoint in batch]),
            torch.stack([datapoint[9] for datapoint in batch]),
            [datapoint[10] for datapoint in batch])


def calculate_overlap_area(data1, data2):
    if data1.nunique() == 1: # all speaker values the same
        return 0.0
    elif data2.nunique() == 1: # all speaker values the same
        return 0.0
    # KDE estimation for each dataset
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    # Calculate the x-axis range to evaluate overlap
    xmin = min(min(data1), min(data2))
    xmax = max(max(data1), max(data2))
    x = np.linspace(xmin, xmax, 1000)
    # Evaluate KDE densities
    density1 = kde1(x)
    density2 = kde2(x)
    # Calculate overlap area using numerical integration
    overlap_area = np.trapz(np.minimum(density1, density2), x)
    return overlap_area


# Function to calculate Bhattacharyya distance manually
def calculate_bhattacharyya_distance(data1, data2):
    if data1.nunique() == 1: # all speaker values the same
        return 0.0
    elif data2.nunique() == 1: # all speaker values the same
        return 0.0
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    
    xmin = min(min(data1), min(data2))
    xmax = max(max(data1), max(data2))
    x = np.linspace(xmin, xmax, 1000)
    
    density1 = kde1(x)
    density2 = kde2(x)
    
    # Calculate Bhattacharyya distance
    bc = np.sum(np.sqrt(density1 * density2))
    distance = -np.log(bc)
    
    return distance

def create_speaker_values(device):
    data_speaker = {
    'model': [],
    'pitch': [],
    'energy': [],
    'duration': [],
    'sentence': [],
    'step': []
    }
    data_speaker_sentence = {
    'model': [],
    'pitch_mean': [],
    'pitch_var': [],
    'energy_mean': [],
    'energy_var': [],
    'duration_mean': [],
    'sentence': [],
    'duration_var': []
    }
    speaker_data = prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS_one_speaker(),
                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "RAVDESS_one_speaker"),
                                        lang="eng", fine_tune_aligner=True)
    train_loader = DataLoader(dataset=speaker_data,
                            collate_fn=collate_and_pad,
                            batch_size=1)
    
    print(len(train_loader))

    for i, batch in enumerate(tqdm(train_loader)):
        text_tensors = batch[0].to(device)
        text_lengths = batch[1].squeeze().to(device)
        speech_indexes = batch[2]
        speech_lengths = batch[3].squeeze().to(device)
        gold_durations = batch[4].to(device)
        gold_pitch = batch[6].to(device)  # mind the switched order
        gold_energy = batch[5].to(device)  # mind the switched order
        lang_ids = batch[8].squeeze(1).to(device)
        path = batch[10][0]

        sentence = speaker_data.pttd[path]
        
        # Unbind tensors along dimension 1
        pitch_unbound = torch.unbind(gold_pitch, dim=1)
        energy_unbound = torch.unbind(gold_energy, dim=1)
        duration_unbound = torch.unbind(gold_durations, dim=1)
        
        
        result = 0.0
        for value in energy_unbound:
            result += value.item()
        
        mean = result / len(energy_unbound)

        data_speaker_sentence['model'].append("speaker")
        data_speaker_sentence['pitch_mean'].append(torch.stack(pitch_unbound).mean().item())
        data_speaker_sentence['pitch_var'].append(torch.stack(pitch_unbound).var().item())
        data_speaker_sentence['energy_mean'].append(mean)
        data_speaker_sentence['energy_var'].append(torch.stack(energy_unbound).var().item())
        data_speaker_sentence['duration_mean'].append(torch.stack(duration_unbound).float().mean().item())
        data_speaker_sentence['duration_var'].append(torch.stack(duration_unbound).float().var().item())
        data_speaker_sentence['sentence'].append(sentence)  

        for idx, (p, e, d) in enumerate(zip(pitch_unbound, energy_unbound, duration_unbound)):
            data_speaker['model'].append("speaker")
            data_speaker['pitch'].append(np.float32(p.cpu().item()))
            data_speaker['energy'].append(np.float32(e.cpu().item()))
            data_speaker['duration'].append(np.float32(d.cpu().item()))   
            data_speaker['sentence'].append(sentence)  
            data_speaker['step'].append(idx)
    
    df_speaker_sentence = pd.DataFrame(data_speaker_sentence)
    df_speaker = pd.DataFrame(data_speaker)
    df_speaker_sentence.to_csv("samples/speaker_sentence.csv", index=False)
    df_speaker.to_csv("samples/speaker.csv", index=False)


def create_model_samples(version, model_id, device, reference_speaker, samples=100, prosody_creativity=0.4, architecture="CFM"):
    data_model = {
    'model': [],
    'pitch': [],
    'energy': [],
    'duration': [],
    'sentence': [],
    'step': []
    }
    data_model_sentence = {
    'model': [],
    'pitch_mean': [],
    'pitch_var': [],
    'energy_mean': [],
    'energy_var': [],
    'duration_mean': [],
    'duration_var': [],
    'sentence': []
    
    }
    # Sentence for speaker ref "sentences_24_regular"        : "You can well enjoy the evening now. We'll make up for it now. The weakness of a murderer. But they wouldn't leave me alone. The telegram was from his wife."
    transcript_for_ears = [
            "Kids are talking by the door.",
            "Dogs are sitting by the door."
            ]
    
    for _ in tqdm(range(samples)):
        for sentence in transcript_for_ears:
            phones, durations, pitches, energies = read_text(model_id=model_id,
                        sentence=sentence,
                        device=device,
                        language="eng",
                        speaker_reference=reference_speaker,
                        prosody_creativity=prosody_creativity,
                        architecture=architecture)
            
            data_model_sentence['model'].append(model_id)
            data_model_sentence['pitch_mean'].append(pitches.mean().item())
            data_model_sentence['pitch_var'].append(pitches.var().item())
            data_model_sentence['energy_mean'].append(energies.mean().item())
            data_model_sentence['energy_var'].append(energies.var().item())
            data_model_sentence['duration_mean'].append(durations.float().mean().item())
            data_model_sentence['duration_var'].append(durations.float().var().item())
            data_model_sentence['sentence'].append(sentence)
            """
            data_model_sentence['pitch_mean'].append(pitches.cpu().numpy().mean())
            data_model_sentence['pitch_var'].append(pitches.cpu().numpy().var())
            data_model_sentence['energy_mean'].append(energies.cpu().numpy().mean())
            data_model_sentence['energy_var'].append(energies.cpu().numpy().var())
            data_model_sentence['duration_mean'].append(durations.cpu().numpy().mean())
            data_model_sentence['duration_var'].append(durations.cpu().numpy().var())
            """

            for idx, (ph, p, e, d) in enumerate(zip(phones, pitches, energies, durations)):
                    data_model['model'].append(model_id.rsplit('/', 1)[-1])
                    data_model['pitch'].append(np.float32(p.cpu()))
                    data_model['energy'].append(np.float32(e.cpu()))
                    data_model['duration'].append(np.float32(d.cpu()))
                    data_model['sentence'].append(sentence)
                    #data_model['phones'].append(ph)
                    data_model['step'].append(idx)
    
    df_sentence = pd.DataFrame(data_model_sentence)
    df_sentence.to_csv(f"samples/{version}_data_samples_sentence.csv", index=False)
    
    df_model = pd.DataFrame(data_model)
    df_model.to_csv(f"samples/{version}_data_samples_sentence_phones.csv", index=False)

def compare_to_reference(version, reference_speaker, model_id, samples=100, device="cpu", use_wandb = True, prosody_creativity=0.4, number=0, architecture="CFM"):
    
    #print(f"GPU {os.environ['CUDA_VISIBLE_DEVICES']} is the only visible device(s).")
    
    if not os.path.exists("samples/speaker.csv") or not os.path.exists("samples/speaker_sentence.csv"):
        create_speaker_values(device)
    
    df_speaker = pd.read_csv("samples/speaker.csv")
    df_speaker_sentence = pd.read_csv("samples/speaker_sentence.csv")
    
    if not os.path.exists(f"samples/{version}_data_samples_sentence_phones.csv") or not os.path.exists(f"samples/{version}_data_samples_sentence.csv"):
        create_model_samples(version, model_id, device, reference_speaker, samples, prosody_creativity, architecture=architecture)
    
    df_model = pd.read_csv(f"samples/{version}_data_samples_sentence_phones.csv")
    df_model_sentence = pd.read_csv(f"samples/{version}_data_samples_sentence.csv")

    df = pd.concat([df_speaker, df_model], ignore_index=True)

    df_sentence = pd.concat([df_speaker_sentence, df_model_sentence], ignore_index=True)


    
    overlap_pitch = []
    overlap_energy = []
    overlap_duration = []

    distance_pitch = []
    distance_energy = []
    distance_duration = []

    
    name = model_id.split("/")[-2] + str(prosody_creativity )             
    # visualization
    fig, axes = plt.subplots(3, 4, figsize=(30, 40))
    # Add titles for the left and right halves
    fig.suptitle(name, fontsize=86, y=0.96)
    

    for idx, sentence in enumerate(df_sentence['sentence'].unique()):
        df_current_sentence = df_sentence[df_sentence['sentence'] == sentence].copy()

        if idx == 0:
            # Add a title for the left half
            fig.text(0.25, 0.91, sentence, ha='center', fontsize=20)
        else:
            # Add a title for the right half
            fig.text(0.75, 0.91, sentence, ha='center', fontsize=20)
        
        # Plot for Pitch
        sns.violinplot(hue='model', y='pitch_mean', split=True, data=df_current_sentence, ax=axes[0, idx*2])
        axes[0, idx*2].set_title(f'Mean Pitch')
        
        # Plot for Energy
        
        #pd.set_option('display.max_rows', None)
        #print(df_sentence[['energy_mean', 'pitch_mean']])
        sns.violinplot(hue='model', y='energy_mean', split=True, data=df_current_sentence, ax=axes[1, 2*idx])
        axes[1, 2*idx].set_title(f'Mean Energy')
        
        # Plot for Duration
        sns.violinplot(hue='model', y='duration_mean', split=True, data=df_current_sentence, ax=axes[2, 2*idx])
        axes[2, 2*idx].set_title(f'Mean Duration')

        # Plot for Pitch
        sns.violinplot(hue='model', y='pitch_var', split=True, data=df_current_sentence, ax=axes[0, (1 + 2*idx)])
        axes[0, (1 + 2*idx)].set_title(f'Variance Pitch')
        
        # Plot for Energy
        sns.violinplot(hue='model', y='energy_var', split=True, data=df_current_sentence, ax=axes[1, (1 + 2*idx)])
        axes[1, (1 + 2*idx)].set_title(f'Variance Energy')

        # Plot for Duration
        sns.violinplot(hue='model', y='duration_var', split=True, data=df_current_sentence, ax=axes[2, (1 + 2*idx)])
        axes[2, (1 + 2*idx)].set_title(f'Variance Duration')
        
    # Adjust layout and display the plot
    #plt.tight_layout()

    plt.savefig(f"visualizations/{version}_speaker_violin.png")
   
    if use_wandb:
        violin_plot = wandb.Image(f"visualizations/{version}_speaker_violin.png")
        wandb.log({f"speaker_violin": violin_plot}, step = number)
    plt.clf()
    plt.close()

    for sentence in df['sentence'].unique():
        df_sentence = df[df['sentence'] == sentence]
        for step in df['step'].unique():
            
            df_step = df_sentence[df_sentence['step'] == step].copy()
            # Melt the DataFrame
            df_step.drop(columns=['sentence', 'step'], inplace=True)
            
            df_speaker = df_step[df_step['model'] == 'speaker'].copy()
            df_model =df_step[df_step['model'] != 'speaker'].copy()

            # Calculate overlap area for each variable
            overlap_pitch.append(calculate_overlap_area(df_speaker['pitch'], df_model['pitch']))
            overlap_energy.append(calculate_overlap_area(df_speaker['energy'], df_model['energy']))
            overlap_duration.append(calculate_overlap_area(df_speaker['duration'], df_model['duration']))

            # Calculate Bhattacharyya distance for each variable
            distance_pitch.append(calculate_bhattacharyya_distance(df_speaker['pitch'], df_model['pitch']))
            distance_energy.append(calculate_bhattacharyya_distance(df_speaker['energy'], df_model['energy']))
            distance_duration.append(calculate_bhattacharyya_distance(df_speaker['duration'], df_model['duration']))

            """
            # Create a boxplot for each variable grouped by model
            g = sns.catplot(
                x="model", y="value", col="variable", 
                kind="box", data=df_melted,
                height=4, aspect=0.7, sharey=False
            )
            g.set_axis_labels("Model", "Value")
            g.set_titles("{col_name}") 

            
            # adjust axis
            for ax, variable in zip(g.axes.flat, df_melted['variable'].unique()):
                data = df_melted[df_melted['variable'] == variable]['value']
                ax.set_ylim(0, data.max() * 1.1)
            
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')

            # Adjust the layout
            plt.tight_layout()


            plt.savefig(f"visualizations/{version}_speaker_boxplot.png")
            
            if use_wandb:
                box_plot = wandb.Image(f"visualizations/{version}_speaker_boxplot.png")
                wandb.log({f"{version}_speaker_boxplot": box_plot})
            """
      # Print results

    overlap_pitch_mean = np.array(overlap_pitch).mean()
    overlap_energy_mean = np.array(overlap_energy).mean()
    overlap_duration_mean = np.array(overlap_duration).mean()

    distance_pitch_mean = np.array(distance_pitch).mean()
    distance_energy_mean = np.array(distance_energy).mean()
    distance_duration_mean = np.array(distance_duration).mean()

    """
    print(f"Overlap Area (Pitch): {overlap_pitch_mean}")
    print(f"Overlap Area (Energy): {overlap_energy_mean}")
    print(f"Overlap Area (Duration): {overlap_duration_mean}")

    print(f"Bhattacharyya Distance (Pitch): {distance_pitch_mean}")
    print(f"Bhattacharyya Distance (Energy): {distance_energy_mean}")
    print(f"Bhattacharyya Distance (Duration): {distance_duration_mean}")
    """
    overlap = (overlap_pitch_mean + overlap_energy_mean + overlap_duration_mean) / 3
    distance = (distance_pitch_mean + distance_energy_mean + distance_duration_mean) / 3
    return distance, overlap
    
    


def add_config_var(original_pt_path):

    # Path to your original .pt file

    # Load the original checkpoint
    checkpoint = torch.load(original_pt_path, map_location='cpu')


    # Update or add your new_variable in the config
    checkpoint['config']['duration_log_scale'] = True

    # Save the modified checkpoint back to the .pt file
    torch.save(checkpoint, original_pt_path)

    print(f"Modified checkpoint saved to {original_pt_path}")

def plot_gaussian_ellipse(ax, mean, cov, color='b', alpha=0.3, dims=(0, 1)):
    """Plot a 2D ellipse representing a Gaussian distribution in the specified dimensions."""
    sub_cov = cov[np.ix_(dims, dims)]
    U, s, rotation = np.linalg.svd(sub_cov)
    radii = np.sqrt(s)
    
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipse = Ellipse(xy=(mean[dims[0]], mean[dims[1]]), width=2*radii[0], height=2*radii[1], angle=angle, color=color, alpha=alpha)
    ax.add_patch(ellipse)


def kl_divergence_gmm(gmm_p, gmm_q, num_samples=10000):
    """
    Compute KL divergence between two Gaussian Mixture Models (GMMs).
    
    Parameters:
    gmm_p : sklearn.mixture.GaussianMixture
        First GMM object.
    gmm_q : sklearn.mixture.GaussianMixture
        Second GMM object.
    num_samples : int, optional
        Number of samples to use for Monte Carlo estimation of KL divergence.
    
    Returns:
    kl_div : float
        KL divergence D(P || Q) between GMMs P and Q.
    """
    # Sample from GMM P
    samples_p, _ = gmm_p.sample(num_samples)
    
    # Evaluate log likelihood of samples under GMM P and GMM Q
    log_likelihood_p = gmm_p.score_samples(samples_p)
    log_likelihood_q = gmm_q.score_samples(samples_p)
    
    
    # Compute KL divergence using Monte Carlo estimation
    kl_div = log_likelihood_p.mean() - log_likelihood_q.mean()
    
    return kl_div

def compare_gmms(path_to_data):

    df = pd.read_csv(path_to_data)
    gmms = []
    models = df['model'].unique()
    print(models)

    for model in models:
        df_model = df[df['model'] == model]
        print(df_model.head())

        # Number of components
        n_components = len(df_model)

        means_list = df_model[['pitch_mean', 'energy_mean', 'duration_mean']].values.tolist()
        var_list = df_model[['pitch_var', 'energy_var', 'duration_var']].values.tolist()
        
        covariances = np.array([np.diag(var) for var in np.array(var_list)])
        means = np.array(means_list)
        # Print the resulting list of 3D means
        # Create Gaussian Mixture Model (GMM) object

        gmm = GaussianMixture(n_components=n_components) #, covariance_type='diag')

        # Initialize means and covariances of the GMM
        weights = np.ones(n_components) / n_components

        # Fit the GMM
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)) for cov in covariances])
        gmms.append(gmm)
        # Sample from the GMM
        samples, _ = gmm.sample(1000)
        # Plot the GMM components
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, n_components))
        # Convert variances to covariance matrices (if applicable)
        # Plot the GMM components in 2D for each pair of dimensions
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Pairs of dimensions to plot
        dim_pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ['(Mean, Energy)', '(Mean, Duration)', '(Energy, Duration)']
        for ax, (dim_x, dim_y), title in zip(axs, dim_pairs, titles):
            for mean, cov, color in zip(means, covariances, colors):
                plot_gaussian_ellipse(ax, mean, cov, color=color, alpha=0.5, dims=(dim_x, dim_y))
            ax.scatter(samples[:, dim_x], samples[:, dim_y], c='black', s=10, alpha=0.5, label='Samples')
            ax.set_title(title)
            ax.set_xlabel(f'Dimension {dim_x + 1}')
            ax.set_ylabel(f'Dimension {dim_y + 1}')
            ax.axis('equal')

        plt.suptitle('2D Projections of 3D GMM')
        plt.tight_layout()
        model = model.replace("/", "_")
        plt.savefig(f"visualizations/{model}_gmm.png")
        plt.close()

    print(f"KL divergence D(P || Q) between GMMs P = {models[0]} and Q = {models[1]}: ")
    print(kl_divergence_gmm(gmms[0], gmms[1], num_samples=1000))

def compare_speaker_gmm(path_to_model_data, path_to_speaker_data):
    df_model = pd.read_csv(path_to_model_data)
    df_speaker = pd.read_csv(path_to_speaker_data)

    # Number of components
    n_components = len(df_model)
    
    X = df_speaker[['pitch', 'energy', 'duration']].values

    means_list = df_model[['pitch_mean', 'energy_mean', 'duration_mean']].values.tolist()
    var_list = df_model[['pitch_var', 'energy_var', 'duration_var']].values.tolist()
    
    covariances = np.array([np.diag(var) for var in np.array(var_list)])
    means = np.array(means_list)

    # Create Gaussian Mixture Model (GMM) object
    gmm = GaussianMixture(n_components=n_components) #, covariance_type='diag')

    # Initialize means and covariances of the GMM
    weights = np.ones(n_components) / n_components

    # Fit the GMM
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)) for cov in covariances])
    
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    print("AIC: ", aic)
    print("BIC: ", bic)
    score = (aic + bic) / 2
    return score



def get_automatic_mos_score(path_to_audios, device="cpu", per_sample=False):
    model = get_wvmos(device=device)
    
    if per_sample:
        mos = []
        print(os.listdir(path_to_audios))
        for audio in os.listdir(path_to_audios):
            print(audio)
            mos.append(model.calculate_one(path_to_audios + "/" + audio))
    else:
        if path_to_audios[-4:] == '.wav':
            # Check if input_data is a string
            mos = model.calculate_one(path_to_audios[:-4])
        else:
            mos = model.calculate_dir(path_to_audios, mean=True) # infer average MOS score across .wav files in directory
        
    return mos


def rename(path_to_folder):

    for dir in os.listdir(path_to_folder):
        correct_name = path_to_folder + "/" + dir
        for file in os.listdir(correct_name):
            new_path = correct_name + "/" + dir + "-" + file.split("-")[1] 
            old_path = path_to_folder + "/" + dir + "/" + file
            os.rename(old_path, new_path)


def compare_predicted_self(version, wandb_id, per_sample=False, use_wandb=True):
    run = wandb.init(id=wandb_id, project="IMS-Toucan-Prosody-Variance", entity="prosody-variance", resume='allow')
    if not os.path.exists(f'samples/{wandb_id}/Evaluation_Table.table.json'):
        
        # Fetch the logged table
        artifact = run.use_artifact(f'prosody-variance/IMS-Toucan-Prosody-Variance/run-{wandb_id}-Evaluation_Table:v1', type='run_table')
        artifact.download(root=f"samples/{wandb_id}")
        if not use_wandb:
            run.finish()
        
    with open(f'samples/{wandb_id}/Evaluation_Table.table.json', 'r') as f:
        table_data = json.load(f)

    # Convert W&B Table to a pandas dataframe
    columns = table_data['columns']
    data = table_data['data']
    df = pd.DataFrame(data=data, columns=columns)
    print(df)
    if per_sample:
        value_vars = ['predicted', 'self_test']
        df['model'] = df['model'].apply(lambda x: x.split("-")[0])
    elif df['self_test_variance'][0] == -1:
        value_vars = ['predicted', 'self_test']
    else:
        value_vars = ['predicted', 'self_test', 'predicted_var', 'self_test_variance']

    df_melted = df.melt(id_vars=['model'], 
                    value_vars=value_vars,
                    var_name='variable', 
                    value_name='value')
    # Create a new 'source' column to distinguish between predicted and self_test
    df_melted['variable'] = df_melted['variable'].apply(lambda x: 'predicted' if 'predicted' in x else 'self_test')
    

    df_test_data = df_melted[df_melted['model']]

    plt.figure(figsize=(20, 6))
    sns.barplot(x='model', y='value', hue='variable', data=df_melted, errorbar='sd')
    # Adding labels and title
    plt.xlabel('Models')
    plt.ylabel('Score')
    name = df['predictor'][0].split("/")[-1]
    if per_sample:
        loss = torch.nn.functional.mse_loss(torch.tensor(df['predicted']), torch.tensor(df['self_test']))
    elif df['self_test_variance'][0] == -1:
        loss = torch.nn.functional.mse_loss(torch.tensor(df['predicted']), torch.tensor(df['self_test']))
    else:
        loss = torch.nn.functional.mse_loss(torch.tensor(df[['predicted', 'predicted_var']].values), torch.tensor(df[['self_test', 'self_test_variance']].values))
    plt.title(f"{name}; LOSS: {loss}")
    plt.xticks(rotation=45,ha='right')

    # Adjust the plot layout to prevent cutting off labels
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"visualizations/{version}_comparison_{name}_results.png")
    if use_wandb:
        bar_plot = wandb.Image(f"visualizations/{version}_comparison_{name}_results.png")
        wandb.log({"Comparison barplot": bar_plot})

if __name__ == '__main__':
    #exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"running on {exec_device}")

    # merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]
    
    """
    create_prosody_samples_per_sentence(version="NF_test",
              model_ids=["NF/NF_epd_log_c8_l6_k5_d0.2"],
              gpu_id=2,
             speaker_reference="audios/speaker_reference/100_121669_000013_000000.wav",
             samples=10)
    
    """
    #add_config_var("Models/ToucanTTS_Libri_Prosody/CFM/epd_log_v4/best.pt")
    #plot_boxplot_per_sentence("CFM_", path_to_data="samples/CFM__data_samples_sentence.csv")
    #create_freq_samples(version="CFM_log_freq", model_ids = ["Libri_Prosody/CFM/epd_log_v4"], gpu_id=0, speaker_reference="audios/speaker_reference/100_121669_000013_000000.wav", samples=10)
    #compare_to_reference("testing", reference_speaker="audios/speaker_reference/sentences_24_regular.wav", samples=3,model_id="Libri_Prosody/CFM/epd_log_v4")
    #plot_freq("CFM_log_freq", path_to_data="samples/CFM_log_freq_pitch.csv")
    #compare_gmms("samples/CFM_gmm_data_samples_sentence.csv")
    #self_test("audios")
    #print(get_automatic_mos_score("audios/", cuda=True))
    #rename("audios/eval_combined")
    compare_predicted_self("eval_combined_VAR", "3abeptps", per_sample=False)