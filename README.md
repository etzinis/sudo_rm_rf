# Sudo rm -rf: Efficient Networks for Universal Sound Source Separation

TLDR; I think the main contribution of this paper is not to simply talk about another boring neural network architecture that works for audio source separation. I think of this paper as a way to express what actually is __"model complexity"__? In this paper we follow a more holistic approach on considering various aspects which are cumbersome for training or running neural models. Mainly, we care about: 
1. The number of floating point operations (FLOPs)
2. Actual memory requirements on the device (in bytes)
3. Time for completing a single forward or backward pass
4. Number of parameters

Moreover, we also propose a convolutional architecture which is capable of capturing long-term temporal audio structure by using successive downsampling and resampling operations which can be efficiently implemented. Our experiments on both speech and environmental sound separation datasets show that SuDoRM-RF performs comparably and even surpasses various state-of-the-art approaches with significantly higher computational resource requirements

You can find our paper here: https://arxiv.org/abs/2007.06833. Please wait for new updates on the presentation at MLSP 2020 https://ieeemlsp.cc/!


## Table of contents

- [Model complexity and results](#model-complexity-and-results)
- [Sudo rm -rf Architecture](#whats-included)
- [How to run](#how-to-run)
- [Copyright and license](#copyright-and-license)


## Model complexity and results

Wait for it...

## Sudo rm -rf architecture

Wait for it...

## How to run

1. Setup your cometml credentials and paths for datasets in the config file.
```
vim __config__.py
```

2. Generate and preprocess the data:

```
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

4. Run the improved version of Sudo rm -rf models by replacing the final softmax with a ReLu activation, forcing the decoder to contain only one module and also replacing the Layernorm with the Global Layernorm. You can run this 

```
cd sudo_rm_rf/dnn/experiments
python run_improved_sudormrf.py --train WHAM --val WHAM --test WHAM --train_val WHAM --separation_task sep_clean --n_train 20000 --n_test 3000 --n_val 3000 --n_train_val 3000 --out_channels 256 --num_blocks 16 -cad 0 1 -bs 4 --divide_lr_by 3. --upsampling_depth 5 --patience 49 -fs 8000 -tags source_separation_is_cool --project_name sudormrf_wham --zero_pad --clip_grad_norm 5.0 --model_type relu
```

5. If you also want to take a look on some speech or enviromental sound classification experiments by using our cool augmentation dataloader which is able to mix multiple datasets with specified prior probabilities:

```
cd sudo_rm_rf/dnn/experiments

# Speech separation experiments.

python run_sudormrf.py --n_epochs 50000 -lr 0.001 -X 5 -N 512 -B 256 -H 512 -L 21 -R 16 -bs 4 -tags environmental_sound_separation -cad 0 --train AUGMENTED_WSJMIX --val AUGMENTED_WSJMIX --train_val AUGMENTED_WSJMIX --datasets_priors 1. --max_abs_snr 5.  --n_val 3000 --n_train 20000 --project_name wsj-all --n_jobs 4 --divide_lr_by 3. --reduce_lr_every 49 --clip_grad_norm 5. --optimizer adam --selected_timelength 4. --log_audio -mlp /tmp/wsj_sudormrf -elp /tmp/wsj_sudormrf
```
```
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
