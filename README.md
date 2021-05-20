This implementation is a very light version of the [LaSAFT+GPoCM](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) model we proposed in our previous paper.

Due to the response time limitation of the DMX challenge, we reduced [its size](https://github.com/ws-choi/music-demixing-challenge-starter-kit/blob/15127603f909a738dc745cdf82dbe5f3d26e6058/lasaft/pretrained/load_pretrained_nets.py#L61) and trained it.

We call this version ```lightsaft```.


## Reproduction

### Training (Optional, if you want to reproduce checkpoints for yourself)

You can re-training as follows:

1. clone https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/tree/lightsaft4dmx_9_256
2. ```python train_lightsaft4dmx.py --musdb_root ../repos/musdb18HQ --gpus 4 --distributed_backend ddp --precision 16 --sync_batchnorm True --pin_memory True --num_workers 64 --seed 2021 --batch_size 16 --log wandb --deterministic True```
3. epoch=694.ckpt => lightsaft_small_2020.ckpt

### Evaluation
1. clone this repository
2. bash run.py to test if it works
3. Follow the [official guideline](https://github.com/AIcrowd/music-demixing-challenge-starter-kit/blob/master/docs/SUBMISSION.md)

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
