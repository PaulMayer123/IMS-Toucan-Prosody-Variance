"""
Train non-autoregressive spectrogram inversion model on a combination of multiple large datasets

In theory, spectrogram inversion should be language and
speaker independent, so throwing together all datasets
that have caches should work.
"""

import os
import random
import warnings

import torch
from torch.utils.data import ConcatDataset

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    model_save_dir = "Models/MelGAN/MultiSpeaker/Combined"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    cache_dir_libri = "Corpora/LibriTTS"
    assert os.path.exists(cache_dir_libri)
    cache_dir_lj = "Corpora/LJSpeech"
    assert os.path.exists(cache_dir_lj)
    cache_dir_css10de = "Corpora/CSS10_DE"
    assert os.path.exists(cache_dir_css10de)

    train_dataset_libri = MelGANDataset(list_of_paths=[],
                                        cache_dir=os.path.join(cache_dir_libri, "melgan_train_cache.json"))
    valid_dataset_libri = MelGANDataset(list_of_paths=[],
                                        cache_dir=os.path.join(cache_dir_libri, "melgan_valid_cache.json"))
    train_dataset_lj = MelGANDataset(list_of_paths=[],
                                     cache_dir=os.path.join(cache_dir_lj, "melgan_train_cache.json"))
    valid_dataset_lj = MelGANDataset(list_of_paths=[],
                                     cache_dir=os.path.join(cache_dir_lj, "melgan_valid_cache.json"))
    train_dataset_css10de = MelGANDataset(list_of_paths=[],
                                          cache_dir=os.path.join(cache_dir_css10de, "melgan_train_cache.json"))
    valid_dataset_css10de = MelGANDataset(list_of_paths=[],
                                          cache_dir=os.path.join(cache_dir_css10de, "melgan_valid_cache.json"))

    train_dataset = ConcatDataset([train_dataset_libri, train_dataset_lj, train_dataset_css10de])
    valid_dataset = ConcatDataset([valid_dataset_libri, valid_dataset_lj, valid_dataset_css10de])

    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batchsize=64,
               epochs=600000,  # just kill the process at some point
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_dataset,
               valid_dataset=valid_dataset,
               device=torch.device("cuda:1"),
               generator_warmup_steps=200000,
               model_save_dir=model_save_dir)
