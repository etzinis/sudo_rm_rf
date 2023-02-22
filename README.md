# Sudo rm -rf: Efficient Networks for Universal Sound Source Separation

TLDR; The main contribution of this paper is to jointly consider audio separation performance in multiple tasks, in addition to computational complexity and memory requirements.  We develop a family of models that take into account the following elements:

1. The number of floating points operations per fixed unit of input time.
2. Memory requirements that also include intermediate variables on the GPU.
3. Execution time for completing a forward and/or a backward pass.
4. The number of parameters for the deployed model.

Moreover, we also propose a convolutional architecture which is capable of capturing long-term temporal audio structure by using successive downsampling and resampling operations which can be efficiently implemented. Our experiments on both speech and environmental sound separation datasets show that SuDoRM-RF performs comparably and even surpasses various state-of-the-art approaches with significantly higher computational resource requirements

[![YouTube sudo rm -rf presentation](http://img.youtube.com/vi/ftc0-tTf4O8/0.jpg)](https://www.youtube.com/watch?v=ftc0-tTf4O8 "sudo rm -rf presentation")

You can find our paper here: https://arxiv.org/abs/2007.06833 alongside its journal extended version with experiments on FUSS and the new group communication variation of sudo rm -rf: https://arxiv.org/pdf/2103.02644.pdf.

Please cite as:
```BibTex
@inproceedings{tzinis2020sudo,
  title={Sudo rm-rf: Efficient networks for universal audio source separation},
  author={Tzinis, Efthymios and Wang, Zhepei and Smaragdis, Paris},
  booktitle={2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}

@article{tzinis2022compute,
  title={Compute and Memory Efficient Universal Sound Source Separation},
  author={Tzinis, Efthymios and Wang, Zhepei and Jiang, Xilin and Smaragdis, Paris},
  journal={Journal of Signal Processing Systems},
  year={2022},
  volume={94},
  number={2},
  pages={245--259},
  publisher={Springer}
}
```


## Table of contents

- [Pre-trained models and easy-to-use recipes](#pre-trained-models-and-easy-to-use-recipes)
- [Model complexity and results](#model-complexity-and-results)
- [Sudo rm -rf architecture](#sudo-rm--rf-architecture)
- [Short - How to run the best models](#short-how-to-run-the-best-models)
- [Extended - How to run previous versions](#extended-how-to-run-previous-versions)
- [Copyright and license](#copyright-and-license)


## Pre-trained models and easy-to-use recipes
You can find all the available pre-trained models below.
| Training Data | Sudo rm -rf version | U-ConvBlocks | Number of encoder bases | Pre-trained model file |
| :---          | :---          |    :----:   |   :----:  |    :----:  |
| WSJ0-2mix     | Group Comm   |  8          | 512             |  [download](https://zenodo.org/record/6299852/files/GroupCom_Sudormrf_U8_Bases512_WSJ02mix.pt?download=1) |
| WSJ0-2mix    | Improved     | 16          | 512     |  [download](https://zenodo.org/record/6299852/files/Improved_Sudormrf_U16_Bases512_WSJ02mix.pt?download=1) |
| WSJ0-2mix    | Improved     | 36          | 2048     |  [download](https://zenodo.org/record/6299852/files/Improved_Sudormrf_U36_Bases2048_WSJ02mix.pt?download=1) |
| WHAMR!    | Improved     | 16          | 2048     |  [download](https://zenodo.org/record/6299852/files/Improved_Sudormrf_U16_Bases2048_WHAMRexclmark.pt?download=1) |
| WHAMR!    | Improved     | 36          | 4096     |  [download](https://zenodo.org/record/6299852/files/Improved_Sudormrf_U36_Bases4096_WHAMRexclmark.pt?download=1) |

Because of issues with git-LFS it would be much easier to download all the pre-trained models from zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6299852.svg)](https://doi.org/10.5281/zenodo.6299852)
 and place them in the corresponding pretrained directory using the following command.
```bash
➜  pretrained_models git:(master) ✗ pwd
/home/thymios/projects/sudo_rm_rf/pretrained_models
➜  pretrained_models git:(master) ✗ bash download_pretrained_models.sh
```

We have also prepared an easy to use example for the pre-trained sudo rm -rf models here [python-notebook](https://github.com/etzinis/sudo_rm_rf/blob/master/sudo_rm_rf/notebooks/sudormrf_how_to_use.ipynb) so you can take all models for a spin 🏎️.. Simply normalize the input audio and infer!
```python
import sudo_rm_rf.dnn.experiments.utils.mixture_consistency as mixture_consistency
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 as sudormrf_gc_v2

# Load a pretrained model
separation_model = torch.load(anechoic_model_p)

anechoic_separation_model_config = os.path.basename(anechoic_model_p)
type_of_model = anechoic_separation_model_config.split('_')[0]

if type_of_model == "Improved":
    model = improved_sudormrf.SuDORMRF(
            out_channels=anechoic_separation_model.out_channels,
            in_channels=anechoic_separation_model.in_channels,
            num_blocks=anechoic_separation_model.num_blocks,
            upsampling_depth=anechoic_separation_model.upsampling_depth,
            enc_kernel_size=anechoic_separation_model.enc_kernel_size,
            enc_num_basis=anechoic_separation_model.enc_num_basis,
            num_sources=anechoic_separation_model.num_sources,)
else:
    model = sudormrf_gc_v2.GroupCommSudoRmRf(
            out_channels=anechoic_separation_model.out_channels,
            in_channels=anechoic_separation_model.in_channels,
            num_blocks=anechoic_separation_model.num_blocks,
            upsampling_depth=anechoic_separation_model.upsampling_depth,
            enc_kernel_size=anechoic_separation_model.enc_kernel_size,
            enc_num_basis=anechoic_separation_model.enc_num_basis,
            num_sources=anechoic_separation_model.num_sources,)
model.load_state_dict(anechoic_separation_model.state_dict())

# Normalize the waveform and apply the model
input_mix_std = separation_model.std(-1, keepdim=True)
input_mix_mean = separation_model.mean(-1, keepdim=True)
input_mix = (separation_model - input_mix_mean) / (input_mix_std + 1e-9)

# Apply the model
rec_sources_wavs = model(input_mix.unsqueeze(1))

# Rescale the input sources with the mixture mean and variance
rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean

# In case you are using the pre-trained models with Group communication
# please also use the mixture consistency right after the estimated waveforms
if "GroupCom" in anechoic_model_p:
    rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs, input_mix.unsqueeze(1))
```

One of the main points that sudo rm -rf models have brought forward is that focusing only on the reconstruction fidelity performance and ignoring all other computational metrics, such as: *execution time* and *actual memory consumption* is an ideal way of wasting resources for getting almost neglidgible performance improvement. To that end, we show that the Sudo rm -rf models can provide a very effective alternative for a range of separation tasks while also being respectful to users who do not have access to immense computational power or researchers who prefer not to train their models for weeks on a multitude of GPUs.

### Results on WSJ0-2mix (anechoic 2-source speech separation) 
| Model version | Batch <br> Size | For. <br> CPU <br> (ex/sec) <br> ⬆️ | For. <br> GPU <br> (ex/sec) <br> ⬆️  |  For. <br> GPU <br> Mem. <br> (GB) <br> ⬇️ |  Back. <br> GPU <br> (ex/sec) <br> ⬆️  |  Back. <br> GPU <br> Mem. <br> (GB) <br> ⬇️   | #Params <br> (10^6) <br> ⬇️  | Mean <br> SI-SDRi <br> (dB)  <br> ⬆️ | 
| :---          |    :----:   | :----:   |   :----:  |    :----:  | :----:   |   :----:  |    :----:  | :----:  |   
|  Group Com <br> *sudo rm-rf* <br> 16 U-ConvBlocks <br> 512 enc. bases |  1 <br> 4 | 1.5 <br> **0.3** 🥇| **43.9** 🥇<br> **78.9** 🥇| **0.06** 🥇<br> **0.25** 🥇| **31.9** 🥇<br> **18.1** 🥇| **1.45**🥇 <br> **5.94**🥇 | **0.51** 🥇 | 13.1 |
|  Improved <br> *sudo rm-rf* <br> 16 U-ConvBlocks <br> 512 enc. bases |  1 <br> 4 | **3.9**🥇 <br> 0.2 | 26.2 <br> 53.3 | 0.08 <br> **0.25**🥇 | 21.8 <br> 11.8 | 2.1 <br> 8.43 | 5.02 | 17.3 |
|  Improved <br> *sudo rm-rf* <br> 36 U-ConvBlocks <br> 2048 enc. bases |  1 <br> 2 | 1.3 <br> OOM | 9.8 <br> OOM | 0.23 <br> OOM | 2.2 <br> OOM | 5.26 <br> OOM | 23.24 | 19.5 |
|  [Sepformer<br>(literature SOTA)](https://arxiv.org/pdf/2202.02884.pdf) |  1 <br> 2 | 0.1 <br> OOM | 10.6 <br> OOM | 0.39 <br> OOM | 3.5 <br> OOM | 8.16 <br> OOM | 25.68 | **22.4** 🥇 |


### Results on WHAMR! (noisy and reverberant 2-source speech separation) 
| Model version | Batch <br> Size | For. <br> CPU <br> (ex/sec) <br> ⬆️ | For. <br> GPU <br> (ex/sec) <br> ⬆️  |  For. <br> GPU <br> Mem. <br> (GB) <br> ⬇️ |  Back. <br> GPU <br> (ex/sec) <br> ⬆️  |  Back. <br> GPU <br> Mem. <br> (GB) <br> ⬇️   | #Params <br> (10^6) <br> ⬇️  | Mean <br> SI-SDRi <br> (dB)  <br> ⬆️ | 
| :---          |    :----:   | :----:   |   :----:  |    :----:  | :----:   |   :----:  |    :----:  | :----:  |   
|  Improved <br> *sudo rm-rf* <br> 16 U-ConvBlocks <br> 2048 enc. bases |  1 <br> 4 | **3.3** 🥇 <br> **0.2** 🥇 | **26.2** 🥇 <br> **48.7** 🥇 | **0.16** 🥇 <br> **0.55** 🥇 | **21.3** 🥇 <br> **11.2** 🥇 | **2.44** 🥇 <br> **9.7** 🥇 | 6.36| 12.1 |
|  Improved <br> *sudo rm-rf* <br> 36 U-ConvBlocks <br> 4096 enc. bases |  1 <br> 2 | 0.3 <br> OOM | 10.1 <br> OOM | 0.37 <br> OOM | 2.2 <br> OOM | 5.73 <br> OOM | 26.61 <br> OOM | **13.5** 🥇  |
|  [DPTNET - SRSSN](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9670704&casa_token=vhrs3lpffDUAAAAA:TBtuZZ84P58l8WaZx-L5uRNuwa_MmE0p72QKocH1h6_mqyHT7s5DS_iiyLmrg8djSmeRuStGTo0&tag=1) |  1 | - | - | - | - | - | **5.7** 🥇  | 12.3 |
|  [Wavesplit<br>(previous SOTA)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9670704&casa_token=vhrs3lpffDUAAAAA:TBtuZZ84P58l8WaZx-L5uRNuwa_MmE0p72QKocH1h6_mqyHT7s5DS_iiyLmrg8djSmeRuStGTo0&tag=1) |  1 | - | - | - | - | - | 29 | 13.2 |

Thus, Sudo rm- rf models are able to perform adequately with SOTA and even surpass it in certain cases with minimal computational overhead in terms of both **time** and **memory**. Also, the importance of reporting all the above metrics when proposign a new model becomes apparent. We have conducted all the experiments assuming 8kHz sampling rate and 4 seconds of input audio on a server with an NVIDIA GeForce RTX 2080 Ti (11 GBs) and an 12-core Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz. OOM means out of memory for the corresponding configuration. A value of **Z** ex/sec corresponds to the throughput of each model, in other words, for each second that passes, the model is is capable of processing (either forward or backward pass) **Z** 32,000 sampled audio files. The attention models, which undoubtly provide the best performance in most of the cases, are extremely heavy in terms of actual time and memory consumption (even if they appear that the number of parameters is rather small). They also become prohibitively expenssive for longer sequencies.

## Model complexity and results

As we discuss in the paper, our main objective is to find efficient architectures not only in terms of one metric but in terms of all metrics which might become a bottleneck during training or inference. This will facilitate the needs of users that do not have in their disposal (or use case) the considerable requirements that many modern models exhibit. This will enable people with no GPU access, or users with interest in edge applications to also make use of this model and not be locked out of good performance.

We present here the results from our paper:

![FUSS-results](images/fuss_results.png "Results on FUSS")
Sudo -rm rf can score similarly to state-of-the-art models with much less parameters and computation time. Check the newest variation with group communication that brings down the number of parameters dramatically!

![ESC-50-results](images/Selection_061.png "ESC-50-results")
SI-SDRi non-speech sound separation performance on ESC50 vs computational resources with an input audio of 8000 samples for all models. (Top row) computational requirements for a single forward pass on CPU (Bottom) for a backward pass on GPU. All x-axis are shown in log-scale while the 3 connected blue stars correspond to the three SuDoRM-RF configurations that we proposed. Namely, SuDoRM-RF 1.0x , SuDoRM-RF 0.5x , SuDoRM-RF 0.25x consist of 16, 8 and 4  U-ConvBlocks, respectively.

![Table-results](images/Selection_062.png "Table-results")
SI-SDRi separation performance for all models on both separation tasks (speech and non-speech) alongside their computational requirements for performing inference on CPU (I) and a backward update step on GPU (B) for one second of input audio or equivalently 8000 samples. * We assign the maximum SI-SDRi performance obtained by our runs and the reported number on the corresponding paper.


## Sudo rm -rf architecture

This is the intiallly proposed architecture which is a mask-based convolutional architecture for audio source separation.

![Sudo rm -rf architecture](images/Selection_059.png "Sudo rm -rf architecture")

Specifically, the backbone structure of this convolutional network is the SUccessive DOwnsampling and Resampling of Multi-Resolution Features (SuDoRMRF) as well as their aggregation which is performed through simple one-dimensional convolutions. We call these blocks: U-ConvBlocks because of their structure:

![U-ConvBlock architecture](images/Selection_060.png "U-ConvBlock architecture")

By repeating those blocks we are able to increase the receptive field of our network without needing dilated convolutions or recurrent connections which is more costly in terms of computational time and memory requirements.

## (Short) How to run the best models

1. Try an example run with Libri2Mix and the latest version of sudo rm rf with group commucation that brings down the parameters by a lot!
```shell
python run_sudormrf_gc_v2.py --train LIBRI2MIX --val LIBRI2MIX --test LIBRI2MIX --train_val LIBRI2MIX \
--separation_task sep_clean --n_train 50800 --n_test 3000 --n_val 3000 --n_train_val 3000 \
--out_channels 512 --in_channels 256 --enc_kernel_size 21 --num_blocks 16 -cad 0 -bs 3 --divide_lr_by 3. --upsampling_depth 5 \
--patience 30 -fs 8000 -tags best_separation_model --project_name sudormrf_libri2mix --zero_pad --clip_grad_norm 5.0
```

2. Try to run the model on FUSS separation with up to 4 sources!
```shell
python run_fuss_separation.py --train FUSS  --n_channels 1 --n_train 20000 \
--enc_kernel_size 41 --enc_num_basis 512 --out_channels 256 --in_channels 512 --num_blocks 16 \
-cad 0 1 2 3 -bs 4 --divide_lr_by 3. -lr 0.001 --upsampling_depth 5 --divide_lr_by 3. \
--patience 10 -fs 16000 -tags sudormrf groupcommv2  bl16 N512 L41 --project_name fuss_sudo \
--zero_pad --clip_grad_norm 5.0 --model_type groupcomm_v2 --audio_timelength 10.
```


## (Extended) How to run previous versions

1. Setup your cometml credentials and paths for datasets in the config file.
```shell
vim __config__.py
```

2. Generate and preprocess the data:

```shell
cd sudo_rm_rf/utils
# Creates data for WHAM as well as WSJ0-2mix and ESC-50 preprocessed with different speakers or classes of sounds stored in separate folders.
bash generate_data.sh
```

3. Instead of creating complex scripts of mixing audio sources we introduce this very simple way of augmenting your mixtures datasets by keeping the same distribution of SNRs:  

```python
for data in tqdm(generators['train'], desc='Training'):
  opt.zero_grad()
  clean_wavs = data[-1].cuda()
  m1wavs = data[0].cuda()

  # Online mixing over samples of the batch. (This might cause to get
  # utterances from the same speaker but it's highly improbable).
  # Keep the exact same SNR distribution with the initial mixtures.

  energies = torch.sum(clean_wavs ** 2, dim=-1, keepdim=True)
  random_wavs = clean_wavs[:, torch.randperm(energies.shape[1])]
  new_s1 = random_wavs[torch.randperm(energies.shape[0]), 0, :]
  new_s2 = random_wavs[torch.randperm(energies.shape[0]), 1, :]
  new_s2 = new_s2 * torch.sqrt(energies[:, 1] /
                               (new_s2 ** 2).sum(-1, keepdims=True))
  new_s1 = new_s1 * torch.sqrt(energies[:, 0] /
                               (new_s1 ** 2).sum(-1, keepdims=True))
  m1wavs = normalize_tensor_wav(new_s1 + new_s2)
  clean_wavs[:, 0, :] = normalize_tensor_wav(new_s1)
  clean_wavs[:, 1, :] = normalize_tensor_wav(new_s2)

  rec_sources_wavs = model(m1wavs.unsqueeze(1))
  l = back_loss_tr_loss(rec_sources_wavs, clean_wavs)
  l.backward()
  opt.step()
```

4. Run the improved version of Sudo rm -rf models by replacing the final softmax with a ReLu activation, forcing the decoder to contain only one module and also replacing the Layernorm with the Global Layernorm. You can run this for WHAM clean separation experiment. 

```shell
cd sudo_rm_rf/dnn/experiments
python run_improved_sudormrf.py --train WHAM --val WHAM --test WHAM --train_val WHAM --separation_task sep_clean --n_train 20000 --n_test 3000 --n_val 3000 --n_train_val 3000 --out_channels 256 --num_blocks 16 -cad 0 1 -bs 4 --divide_lr_by 3. --upsampling_depth 5 --patience 49 -fs 8000 -tags sudo_rm_rf_16 --project_name sudormrf_wham --zero_pad --clip_grad_norm 5.0 --model_type relu
```
Or the following command for LIBRI2MIX clean speech separation experiment.
```shell
cd sudo_rm_rf/dnn/experiments
python run_improved_sudormrf.py --train LIBRI2MIX --val LIBRI2MIX --test LIBRI2MIX --train_val LIBRI2MIX --separation_task sep_clean --n_train 50800 --n_test 3000 --n_val 3000 --n_train_val 3000 --out_channels 512 --num_blocks 34 -cad 0 1 -bs 4 --divide_lr_by 3. --upsampling_depth 5 --patience 39 -fs 8000 -tags sudo_rm_rf_34 --project_name sudormrf_libri2mix --zero_pad --clip_grad_norm 5.0 --model_type relu
```

5. If you also want to take a look on some speech or environmental sound classification experiments by using our cool augmentation dataloader which is able to mix multiple datasets with specified prior probabilities:

```shell
cd sudo_rm_rf/dnn/experiments

# Speech separation experiments.

python run_sudormrf.py --n_epochs 50000 -lr 0.001 -X 5 -N 512 -B 256 -H 512 -L 21 -R 16 -bs 4 -tags environmental_sound_separation -cad 0 --train AUGMENTED_WSJMIX --val AUGMENTED_WSJMIX --train_val AUGMENTED_WSJMIX --datasets_priors 1. --max_abs_snr 5.  --n_val 3000 --n_train 20000 --project_name wsj-all --n_jobs 4 --divide_lr_by 3. --reduce_lr_every 49 --clip_grad_norm 5. --optimizer adam --selected_timelength 4. --log_audio -mlp /tmp/wsj_sudormrf -elp /tmp/wsj_sudormrf
```
```shell
# Environmental sound separation experiments.

python run_sudormrf.py --n_epochs 50000 -lr 0.001 -X 5 -N 512 -B 256 -H 512 -L 21 -R 16 -bs 4 -tags environmental_sound_separation -cad 0 --train AUGMENTED_ESC50 --val AUGMENTED_ESC50 --train_val AUGMENTED_ESC50 --datasets_priors 1. --max_abs_snr 5.  --n_val 3000 --n_train 20000 --project_name wsj-all --n_jobs 4 --divide_lr_by 3. --reduce_lr_every 49 --clip_grad_norm 5. --optimizer adam --selected_timelength 4. --log_audio -mlp /tmp/esc50_sudormrf -elp /tmp/esc50_sudormrf
```


## Copyright and license
University of Illinois Open Source License

Copyright © 2020, University of Illinois at Urbana Champaign. All rights reserved.

Developed by: Efthymios Tzinis 1, Zhepei Wang 1 and Paris Smaragdis 1,2

1: University of Illinois at Urbana-Champaign 

2: Adobe Research 

This work was supported by NSF grant 1453104. 

Paper link: https://arxiv.org/abs/2007.06833

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. Neither the names of Computational Audio Group, University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
