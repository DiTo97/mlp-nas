# An overview of Neural architecture search (NAS)

## Joint project for AML exam (101804)

This written plan is intended to provide a detailed separation of roles and personal contributions to the final project of the *Advanced machine learning* exam (101804) at UniGe.

#### Involved students:
 + **Federico Minutoli** (4211286) 
 + **Massimiliano Ciranni** (4053864)

For each of the following content/activity presented here we set a tag [*Last name*], indicating which one of us is atomically contributing to the associated task. In the absence of a specific tag a 50% effort has to be inteded from the both of us.

**Please note:** The presence of a question mark (**?**) indicates that we are still in doubt about including the associated activity into the final submission-ready *Table of Contents* of the report.

## Abstract

Recent advances in Deep learning research have given birth to really powerful architectures, which have been used with great success in a wide range of fields. Despite designing neural architectures has become easier, a lot of human expertise is still required to find optimal solutions and often comes as the result of extended trial and error. This has led to a rapid increase in the complexity of architectures (e.g., ResNet with its many skip connections) that has left a large segment of the machine learning community wondering "*How can someone come up with such an architecture?*" (J. Langford). This is a hot-topic in machine learning right-now, which is being addressed by neural architecture search (NAS), a novel branch of automated machine learning that aims at automating architecture engineering. The core idea behind a NAS system is to feed it with a dataset and a task (classification, regression, etc…) as input, expecting it to find an architecture by navigating through a search space of all possible architectures trying to maximize a reward-based search strategy. In this paper we will provide an overview of the many questions that NAS tries to answer, also addressing some of the main limitations that plague it (i.e., GPU time, combinatorial explosion, domain shift, etc..). Then we will provide a brief outline of the main approaches followed by a more in depth analysis on two of them: NAS based on differentiable search spaces (DNAS) and on (efficient) reinforcement learning (ENAS). Afterwards, we will shortly discuss a state-of-the-art network optimization model, MorphNet by Google, that optimizes a neural network through a cycle of *shrinking and expanding* phases. Finally, if GPU time permits, we will present a comparison between our proposed naive implementation of a controller-based NAS against ResNet and its optimized version through MorphNet on a simple and controlled task, i.e., digits recognition on the *MNIST dataset*.

## Report

We expect to deliver a 5 to 10 pages joint LaTeX report with the following (tentative) *Table of Contents*.

### Table of Contents

1. **Introduction to NAS** (5W's of journalism) 

2. **Problem setting analysis**
	- Different approaches
		* Reinforcement learning (classic **NAS**, or efficient via *Parameter sharing*, **ENAS**) [*Minutoli*]
		* Progressive search (block-cells NAS or **PNAS**) [*Ciranni*] 
		* Differentiable continuous search (**DNAS**) [*Ciranni*] 
		* Evolutionary genetic search (**AmoebaNAS**) [*Minutoli*]
		* Multi-agent game search (**MANAS**) (*?*)
	- Limitations and advances
		* GPU time complexity [*Minutoli*]
		* Macro vs. micro search [*Ciranni*]
		* Domain shift robustness [*Ciranni*]
		* Skip connections (multi-branch NNs) [*Minutoli*]
		
3. **Algorithmic overview**
	- In-depth analysis **DARTS** [*Ciranni*]
	- In-depth analysis **ENAS** [*Minutoli*]
	
4. **Few words on MorphNet** (*?*)

5. **Custom implementation vs. MorphNet on a controlled domain**
	- Accuracy metrics
	- Domain search time (GPU hours)
	- Domain shift robustness (*?*)
	
### Useful links
	
- [Neural architecture search - The future of Deep learning](https://theaiacademy.blogspot.com/2020/05/neural-architecture-search-nas-future.html)

- [AutoML for large-scale Image classification and detection (**NASNet**)](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html)

- [Towards faster and smaller Neural networks (**MorphNet**)](https://ai.googleblog.com/2019/04/morphnet-towards-faster-and-smaller.html)

- [Bridge the gap between Micro- and Macro-search in NAS](http://metalearning.ml/2018/papers/metalearn2018_paper16.pdf)

- [One-shot NAS algorithms in Microsof's NNI](https://nni.readthedocs.io/en/latest/NAS/one_shot_nas.html)

- [Tesla's new NN model for Model compression](https://analyticsindiamag.com/why-tesla-invented-a-new-neural-network/)

- **Neural architecture search**: A series of articles on different aspects of NAS with bonus Python [source code](https://github.com/codeaway23/MLPNAS)
	* **Part 1** - [An overview](https://blog.paperspace.com/overview-of-neural-architecture-search/)
	* **Part 2** - [Search space, Architecture design and One-shot training](https://blog.paperspace.com/neural-architecture-search-one-shot-training/)
	* **Part 3** - [Controllers and Accuracy predictors](https://blog.paperspace.com/neural-architecture-search-controllers/)
	* **Part 4** - TBD

- **Microsoft Research talks**: A series of YouTube lectures by Dr. Debadeepta Dey
	* [Neural architecture search](https://www.youtube.com/watch?v=wL-p5cjDG64)
	* [Efficient forward architecture search](https://www.youtube.com/watch?v=sZMZ6nJFaJY&t=84s)

## Code

In the case of a feasible computational effort underlying the cost of GPUs, we would like to build a naive implementation of a *controller-based NAS* system (Reinforcement learning), along with a comparison against *ResNet* and its optimized version obtained through *MorphNet* in the context of a simple and controlled task (i.e. digits recognition on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)). Otherwise, if the computational demand on GPU is too high for our custom implementation we will stick to the latter, comparing a state-of-the-art human made network against one optimized through MorphNet.

Python source code and benchmarks will be made available for evaluation through a GitHub repository.

### Dataset

- [Handwritten digits MNIST dataset](http://yann.lecun.com/exdb/mnist/)
