# An overview of Neural architecture search (NAS)

## Joint project for AML exam (101804)

This written plan is intended to provide a detailed separation of roles and personal contributions to the final project of the *Advanced machine learning* exam (101804) at UniGe.

#### Involved students:
 + **Federico Minutoli** (4211286) 
 + **Massimiliano Ciranni** (4053864)

For each of the following content/activity presented here we set a tag [*last_name*], indicating which one of us is atomically contributing to the associated task. In the absence of a specific tag a 50% effort has to be inteded from the both of us.

**Note:** In the <code>code</code> folder you can find source code files for our Python implementation, with *run.py* the main entry point, whereas in the <code>report</code> you can find our produced joint report for AML's porject.

## Abstract

Recent advances in deep learning research have given birth to really powerfularchitectures, which have been used with great success in a wide range of fields.Despite designing neural architectures has become easier, a lot of human expertise isstill required to find optimal solutions and often comes as the result of extended trialand error cycles. This has led to a rapid increase in the complexity of architectures(e.g., ResNet with its many skip connections) that has left a large segment of themachine learning community wondering "How can someone come out with suchan architecture?" (J. Langford referring toDenseNet) [1]. This is a hot-topic inmachine learning right-now, which is being addressed by neural architecture search(NAS), a novel branch of automated machine learning that aims at automatingarchitecture engineering.  The core idea behind a NAS system is to feed it witha dataset and a task (classification, regression, etc.) as input, expecting it to findan architecture by navigating through a search space of all possible architecturestrying to maximize a reward-based search strategy. In this paper we will providean overview of the many questions that NAS tries to answer, also addressing someof the main limitations that plague it at the moment (i.e., GPU time, combinatorialexplosion, domain shift, etc.).  Then we will provide a brief outline of the mainapproaches followed by a more in depth analysis on two of them: differentiablesearch spaces (DNAS) and (efficient) reinforcement learning (ENAS). In the end,we will present a naive toy implementation of a controller-based NAS systemtailored at sampling multi-layer perceptrons (MLPs) on a handful of datasets,against Scikit-learn’s pre-made MLP classifier.

### Table of Contents

1. **Introduction to NAS** (5W's of journalism) 

2. **Problem setting analysis**
	* Search Space
		* Macro-vs-micro [*Ciranni*] 
	* Search Strategy
		* Reinforcement learning (classic **NAS**) [*Minutoli*]
		* Progressive search (**PNAS**) [*Ciranni*] 
		* Differentiable search (**DNAS**) [*Ciranni*]
		* Others [*Minutoli*]
	* Performance estimation [*Minutoli*]		
		
3. **Algorithmic overview**
	- In-depth analysis **DARTS** [*Ciranni*]
	- In-depth analysis **ENAS** [*Minutoli*]
	
4. **Few words on MorphNet**

5. **Custom implementation vs. MorphNet on a controlled domain**
	- Accuracy metrics
	- Performances of our implementation
