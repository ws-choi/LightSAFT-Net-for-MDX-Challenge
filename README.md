This implementation is a very light version of the [LaSAFT+GPoCM](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) model we proposed in our previous paper.

Due to the response time limitation of the DMX challenge, we reduced [its size](https://github.com/ws-choi/music-demixing-challenge-starter-kit/blob/15127603f909a738dc745cdf82dbe5f3d26e6058/lasaft/pretrained/load_pretrained_nets.py#L61) and trained it.

We call this version ```LightSAFT + GPoCM```.


## Reproduction

### Training (Optional, if you want to reproduce checkpoints for yourself)

You can re-train ```LightSAFT + GPoCM``` as follows:

1. clone https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/tree/lightsaft4dmx_9_256
2. ```python train_lightsaft4dmx.py --musdb_root ../repos/musdb18HQ --gpus 4 --distributed_backend ddp --precision 16 --sync_batchnorm True --pin_memory True --num_workers 64 --seed 2021 --batch_size 16 --log wandb --deterministic True```
3. epoch=694.ckpt => lightsaft_small_2020.ckpt

### Evaluation
1. clone this repository
2. bash run.py to test if it works
3. Follow the [official guideline](https://github.com/AIcrowd/music-demixing-challenge-starter-kit/blob/master/docs/SUBMISSION.md)

## Hyperparmeter comparison

|                          | LightSAFT + GPoCM                        | LaSAFT+GPoCM                 | LaSAFT+GPoCM (large)         |
|--------------------------|----------------------------------|------------------------------|------------------------------|
| n_fft                    | 2048                             | 2048                         | 4096                         |
| hop_length               | 1024                             | 1024                         | 1024                         |
| num_frame                | 256                              | 128                          | 128                          |
| frequency transformation | [LightSaFT](https://github.com/ws-choi/music-demixing-challenge-starter-kit/blob/15127603f909a738dc745cdf82dbe5f3d26e6058/lasaft/source_separation/conditioned/LaSAFT.py#L34) with 16 Latent Sources | [LaSAFT](https://github.com/ws-choi/music-demixing-challenge-starter-kit/blob/15127603f909a738dc745cdf82dbe5f3d26e6058/lasaft/source_separation/conditioned/LaSAFT.py#L9) with 6 Latent Sources | [LaSAFT](https://github.com/ws-choi/music-demixing-challenge-starter-kit/blob/15127603f909a738dc745cdf82dbe5f3d26e6058/lasaft/source_separation/conditioned/LaSAFT.py#L9) with 6 Latent Sources |
| # of intermediate layers | 9                                | 7                            | 9                            |
| # internal channels      | 16                               | 24                           | 24                           |
| dk and embedding dim     | 64                               | 32                           | 64                           |
| DMX time constraints     | passed                           | failed                       | failed                       |

## Authors

- Woosung Choi
- Minseok Kim

## Bibtex

```bibtex
@inproceedings{choi2021lasaft,
  title={LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation},
  author={Choi, Woosung and Kim, Minseok and Chung, Jaehwa and Jung, Soonyoung},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={171--175},
  year={2021},
  organization={IEEE}
}
```
