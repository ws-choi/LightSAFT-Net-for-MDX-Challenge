from test_lightsaft import LightSAFTPredictor

# a light-weight version of the following paper

"""
@inproceedings{choi2021lasaft,
  title={LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation},
  author={Choi, Woosung and Kim, Minseok and Chung, Jaehwa and Jung, Soonyoung},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={171--175},
  year={2021},
  organization={IEEE}
}
"""

# Github Repository: https://github.com/ws-choi/music-demixing-challenge-starter-kit
# Github Repository of the original paper: https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT

lightsaft_predictor = LightSAFTPredictor()
submission = lightsaft_predictor
submission.run()
print("Successfully completed music demixing...")
