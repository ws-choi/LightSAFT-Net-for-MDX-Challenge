#!/usr/bin/env python

# a light-weight version of LaSAFT+GPoCM

# Quick Tutorial - How to submit?: see https://github.com/ws-choi/music-demixing-challenge-starter-kit/blob/master/README.md#evaluation
# Please read README.md from the following link for details :)
# Github Repository: https://github.com/ws-choi/music-demixing-challenge-starter-kit
# Github Repository of the original paper: https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT
# Lasaft Demo site: https://lasaft.github.io/

import soundfile as sf
import torch
from torch.utils.data import DataLoader

from evaluator.music_demixing import MusicDemixingPredictor
from lasaft.data.musdb_wrapper import SingleTrackSet
from lasaft.pretrained.load_pretrained_nets import PreTrainedLightSAFTNet

class LightSAFTPredictor(MusicDemixingPredictor):
    def prediction_setup(self):
        self.model = PreTrainedLightSAFTNet('lightsaft_small_2020')
        self.model.eval()
        self.model.freeze()

    def separator(self, audio, rate):
        pass

    def prediction(
            self,
            mixture_file_path,
            bass_file_path,
            drums_file_path,
            other_file_path,
            vocals_file_path,
    ):
        mix, rate = sf.read(mixture_file_path, dtype='float32')
        # mix = np.concatenate((mix, mix), axis=0)
        device = self.model.device
        dataset = SingleTrackSet(mix, self.model.hop_length, self.model.num_frame)

        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size)

        trim_length = dataset.trim_length
        total_length = mix.shape[0]
        window_length = self.model.hop_length * (self.model.num_frame - 1)
        true_samples = window_length - 2 * trim_length

        results = {0: [], 1: [], 2: [], 3: []}

        with torch.no_grad():
            for mixture, window_ids, offsets in dataloader:
                target_hats = self.model.separate(mixture, offsets)[:, trim_length:-trim_length]  # B, T, 2
                for target_hat, offset in zip(target_hats, offsets):
                    results[offset.item()].append(target_hat)
                # input_conditions.append(input_condition)

        vocals, drums, bass, other = [torch.cat(results[i])[:total_length].cpu().detach().numpy() for i in range(4)]

        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }

        for target, target_name in zip([vocals, drums, bass, other], ['vocals', 'drums', 'bass', 'other']):
            sf.write(target_file_map[target_name], target, samplerate=44100)


if __name__ == "__main__":
    submission = LightSAFTPredictor()
    submission.run()
    print("Successfully generated predictions!")
