import os
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, f_oneway
import librosa

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts_to_file(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor)
    del tts

def read_text(model_id, sentence, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    
    phone, durations, pitch, energy = tts.get_prosody_values(text=sentence, duration_scaling_factor=duration_scaling_factor)
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



def variance_test(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts_to_file(model_id=model_id,
               sentence=["It snowed, rained, and hailed the same morning.",
                         "It snowed, rained, and hailed the same morning.",
                         "It snowed, rained, and hailed the same morning.",
                         "It snowed, rained, and hailed the same morning.",],
               filename=f"audios/{version}_variance_test.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)





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

def create_prosody_samples_per_sentence(version, model_ids=["Meta"], gpu_id = 1, speaker_reference=None, samples=100):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples", exist_ok=True)

    # Check if the device is set correctly
    print(f"Using device: {device}")
    data = {
    'model': [],
    'pitch': [],
    'energy': [],
    'duration': []
    }
    
    for model_id in model_ids:
        print("Computing Samples for the model: ", model_id)
        for i in tqdm(range(samples)):
            phones, durations, pitches, energies = read_text(model_id=model_id,
                sentence="It snowed, rained, and hailed the same morning.",
                device=device,
                language="eng",
                speaker_reference=speaker_reference)
            for ph, p, e, d in zip(phones, pitches, energies, durations):
                data['model'].append(model_id)
                data['pitch'].append(p.cpu().numpy().mean())
                data['energy'].append(e.cpu().numpy().mean())
                data['duration'].append(d.cpu().numpy().mean())
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
                        value_vars=['pitch', 'energy', 'duration'],
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
        pitch_mean=('pitch', 'mean'),
        pitch_std=('pitch', 'std'),
        pitch_range=('pitch', lambda x: x.max() - x.min()),
        energy_mean=('energy', 'mean'),
        energy_std=('energy', 'std'),
        energy_range=('energy', lambda x: x.max() - x.min()),
        duration_mean=('duration', 'mean'),
        duration_std=('duration', 'std'),
        duration_range=('duration', lambda x: x.max() - x.min())
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

        for metric in ['pitch', 'energy', 'duration']:
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
        for metric in ['pitch', 'energy', 'duration']:
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

    


        

if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    # merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]
    """
    create_prosody_samples_per_sentence(version="CFM_test_2",
              model_ids=["Libri_Prosody/CFM/energy_pitch_duration", "Libri_Prosody/CFM/pitch_energy_duration"],
              gpu_id=0,
             speaker_reference="audios/speaker_reference/100_121669_000013_000000.wav",
             samples=2)
    """
    plot_boxplot_per_sentence("test_", path_to_data="samples/CFM_test_data_samples_sentence.csv")
    #create_freq_samples(version="test_freq", model_ids = ["Libri_Prosody/CFM/epd_log_v2"], gpu_id=0, speaker_reference="audios/speaker_reference/100_121669_000013_000000.wav", samples=10)

    #plot_freq("test")