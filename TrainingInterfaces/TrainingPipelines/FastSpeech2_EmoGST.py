"""
This is the setup with which the embedding model is trained. After the embedding model has been trained, it is only used in a frozen state.
"""

import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop_with_embed import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    use_wandb = os.path.isfile("Utility/wandb.key")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_EmoGST")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()
    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "ravdess"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "esds"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "libri_all_clean"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "vctk"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "hifi"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "Nancy"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "LJSpeech"),
                                              lang="en"))
    train_set = ConcatDataset(datasets)

    model = FastSpeech2(lang_embs=None)
    if use_wandb:
        wandb.init(name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}")
        wandb.watch(model, log_graph=True)
    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=32,
               lang="en",
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=4000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
