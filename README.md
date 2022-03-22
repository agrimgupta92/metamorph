# MetaMorph: Learning Universal Controllers with Transformers

This is the code for the paper

**<a href="https://openreview.net/forum?id=Opmqtk_GvYL">MetaMorph: Learning Universal Controllers with Transformers</a>**
<br>
<a href="http://web.stanford.edu/~agrim/">Agrim Gupta</a>,
<a href="http://jimfan.me">Linxi Fan</a>,
<a href="https://ganguli-gang.stanford.edu/surya.html">Surya Ganguli</a>,
<a href="http://vision.stanford.edu/feifeili/">Fei-Fei Li</a>
<br>

Multiple domains like vision, natural language, and audio are witnessing tremendous progress by leveraging Transformers for large scale pre-training followed by task specific fine tuning. In contrast, in robotics we primarily train a single robot for a single task. However, modular robot systems now allow for the flexible combination of general-purpose building blocks into task optimized morphologies. However, given the exponentially large number of possible robot morphologies, training a controller for each new design is impractical. In this work, we propose MetaMorph, a Transformer based approach to learn a universal controller over a modular robot design space. MetaMorph is based on the insight that robot morphology is just another modality on which we can condition the output of a Transformer. Through extensive experiments we demonstrate that large scale pre-training on a variety of robot morphologies results in policies with combinatorial generalization capabilities, including zero shot generalization to unseen robot morphologies. We further demonstrate that our pre-trained policy can be used for sample-efficient transfer to completely new robot morphologies and tasks.

<div align='center'>
<img src="images/teaser.gif"></img>
</div>

## Code Structure

The code consists of two main components:

1. [Metamorph](tools/train_ppo.py): Code for joint pre-training of different robots.
2. [Environments and evaluation tasks](metamorph/envs):  Three pre-training environments and two evaluation environments.  

## Benchmark

We also provide Unimal-100 benchmark. The benchmark consists of 100 train morphologies, 1600 morphologies with dynamics variations, 800 morphologies with kinematics variations, and 100 test morphologies. 

```bash
# Install gdown
pip install gdown
# Download data
gdown 1LyKYTCevnqWrDle1LTBMlBF58RmCjSzM
# Unzip
unzip unimals_100.zip
```

## Setup
We provide [Dockerfile](docker/Dockerfile) for easy installation and development. If you prefer to work without docker please take a look at Dockerfile and ensure that your local system has all the necessary dependencies installed. 

## Training
```bash
# Build docker container. Ensure that MuJoCo license is present: docker/mjkey.txt
./scripts/build_docker.sh
# Joint pre-training. Please change MOUNT_DIR location inside run_docker_gpu.sh
# Finally ensure that ENV.WALKER_DIR points to benchmark files and is accessible
# from docker.
./scripts/run_docker_gpu.sh python tools/train_ppo.py --cfg ./configs/ft.yaml
```

The default parameters assume that you are running the code on a machine with atlesat 1 GPU.

## Citation
If you find this code useful, please consider citing:

```bibtex
@inproceedings{
    gupta2022metamorph,
    title={MetaMorph: Learning Universal Controllers with Transformers},
    author={Agrim Gupta and Linxi Fan and Surya Ganguli and Li Fei-Fei},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=Opmqtk_GvYL}
}

```

## Credit

This codebase would not have been possible without the following amazing open source codebases:

1. [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
2. [hill-a/stable-baselines](https://github.com/hill-a/stable-baselines)
3. [deepmind/dm_control](https://github.com/deepmind/dm_control)
4. [openai/multi-agent-emergence-environments](https://github.com/openai/multi-agent-emergence-environments)
