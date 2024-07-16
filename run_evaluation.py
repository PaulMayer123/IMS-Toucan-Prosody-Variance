import time
import wandb
import argparse
import os
import pandas as pd
from tqdm import tqdm
import test_suite as test
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def run_eval(path_to_models, use_wandb=True, gpu_id=None):
    if gpu_id is None:
        device = torch.device("cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")
    if use_wandb:
        wandb.init(
            name=f"Evaluation_{time.strftime('%Y%m%d-%H%M%S')}")
    results = {
        'model': [],
        'self_test': [],
        'distance_speaker': [],
        'overlap': [],
        'gmm': [],
        'variance': [],
        'wv_mos': []
        }
    model_ids = os.listdir(path_to_models)
    print(model_ids)
    for model_id in tqdm(model_ids):
        print(f"Evaluating Model {model_id}")
        full_model_path = path_to_models + "/" + model_id + "/best.pt"
        audio_samples=40
        if not os.path.exists(f"audios/eval2/{model_id}/{model_id}-0.wav"):
            # create data for self-test
            print("Creating sample audios")
            
            test.variance_test(f"{model_id}", samples= audio_samples, model_id=full_model_path, exec_device=device, speaker_reference="audios/RAVDESS_one/Actor_19/03-01-01-01-01-01-19.wav")
        
        # get distance to speaker
        print("Compare to speaker")
        speaker_distance, speaker_overlap = test.compare_to_reference(version=f"{model_id}", reference_speaker="audios/RAVDESS_one/Actor_19/03-01-01-01-01-01-19.wav", samples=100 ,model_id=full_model_path, device=device, use_wandb=use_wandb)
        print("Speaker Score: ", speaker_distance)
        print("Speaker overlap: ", speaker_overlap)
        

        pitch_mean, pitch_std, energy_mean, energy_std, duration_mean, duration_std = test.get_mean_var(f"samples/{model_id}_data_samples_sentence.csv")

        print(pitch_mean)
        print(pitch_mean.values[0])
        # TODO combine in meaningful way!
        variance_score = pitch_mean.values[0] + pitch_std.values[0] + energy_mean.values[0] + energy_std.values[0] + duration_mean.values[0] + duration_std.values[0]
        # get GMM score
        gmm_score = test.compare_speaker_gmm(path_to_model_data=f"samples/{model_id}_data_samples_sentence.csv", path_to_speaker_data="samples/speaker.csv")    
        print("gmm_score ", gmm_score)       
        # get WV-Mos
        print("variance score: ", variance_score)
        print("Computing MOS score")
        mos_score = test.get_automatic_mos_score(f"audios/eval2/{model_id}", device = device)
        print("Mos: ", mos_score)
        # TODO normalization

        results["model"].append(model_id)
        results["self_test"].append(-1)
        results["distance_speaker"].append(speaker_distance)
        results["overlap"].append(speaker_overlap)
        results["gmm"].append(gmm_score)
        results["variance"].append(variance_score)
        results["wv_mos"].append(mos_score)
        
    # Create DataFrame
    df = pd.DataFrame(results)
    
    pd.set_option('display.max_columns', 18)
    # Metrics
    metrics = ["distance_speaker", "gmm", "variance", "wv_mos", "overlap"]
    
    # Standardize the metrics
    for metric in metrics:
        if metric == "distance_speaker":  # Bhattacharyya distance (lower is better)
            df[metric + "_transformed"] = -df[metric].abs()  # Negate to treat lower as better
        elif metric == "gmm":  # Lower is better
            df[metric + "_transformed"] = -df[metric]  # Negate to treat lower as better
        elif metric == "overlap":  # Should be close to 1
            df[metric + "_transformed"] = -(df[metric] - 1).abs()  # Negate the absolute deviation from 1
        else:  # Higher is better (variance and wv_mos)
            df[metric + "_transformed"] = df[metric]
    
    # Normalize each metric to z-scores
    for metric in metrics:
        transformed_metric = metric + "_transformed"
        df[metric + "_zscore"] = (df[transformed_metric] - df[transformed_metric].mean()) / df[transformed_metric].std()
   
    # Assign weights (equal weights for simplicity)
    weights = {metric: 1/len(metrics) for metric in metrics}

    # Compute the total score using DataFrame operations
    df['total'] = sum(df[metric + "_zscore"] * weights[metric] for metric in metrics)


    # Sort the DataFrame by the total score
    df_sorted = df.sort_values(by='total', ascending=False)

    df_melt = df_sorted.melt(id_vars=['model'], 
                        value_vars=['total'],
                        var_name='variable', 
                        value_name='value')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='value', data=df_melt, palette='coolwarm')

    # Adding labels and title
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Score for Different Models')

    # Adding horizontal line at y=0 for reference
    plt.axhline(0, color='gray', linestyle='--')
    plt.savefig(f"visualizations/results.png")


    print(df_sorted)
    if use_wandb:
        table = wandb.Table(dataframe=df_sorted)

        wandb.log({"Evaluation_Table": table})
        bar_plot = wandb.Image(f"visualizations/results.png")
        wandb.log({f"Result barplot": bar_plot})

        wandb.finish()  

    # save used data as csv
    df_sorted.to_csv(f"samples/evaluation_{time.strftime('%Y%m%d-%H%M%S')}.csv")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--model_dir',
                        type=str,
                        help="Path to Models.",
                        default=None)
    
    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU(s) to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    parser.add_argument('--wandb',
                        action="store_true",
                        help="Whether to use weights and biases to track training runs. Requires you to run wandb login and place your auth key before.",
                        default=False)

    args = parser.parse_args()

    run_eval(path_to_models = args.model_dir, use_wandb = args.wandb, gpu_id = args.gpu_id)
