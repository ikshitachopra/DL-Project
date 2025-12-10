# DL-Project
ECTF+VAE: A Hybrid EfficientNet–Token Transformer–VAE Ensemble for Fashion-MNIST Classification
Abstract
This work presents a novel hybrid deep learning pipeline, called ECTF+VAE, for image classification on the Fashion-MNIST benchmark. The proposed approach combines (i) an EfficientNet-B0 convolutional backbone for local feature extraction, (ii) mid-level tokenization of feature maps followed by a compact Transformer encoder for modeling long-range dependencies, (iii) a token-level variational autoencoder (Token-VAE) that regularizes token representations via reconstruction and KL divergence, and (iv) an ensemble of two independently seeded models whose logits are averaged at inference time. Experimental results on Fashion-MNIST show that a single ECTF+VAE model reaches approximately 94.9% test accuracy, while a two-member ensemble attains 95.38% accuracy and a macro F1-score of 0.9537 after only six training epochs. Compared to a simple CNN baseline (~60% accuracy) and a pure EfficientNet-style backbone, the proposed pipeline yields substantial gains while remaining computationally moderate. We also compare our results with recent CNN and CNN–Transformer hybrids from the literature and highlight that ECTF+VAE offers a competitive accuracy–complexity trade-off and a clearly interpretable modular design that can be adapted to other small-scale classification problems.
Keywords
Fashion-MNIST, EfficientNet, Vision Transformer, Variational Autoencoder, Hybrid CNN–Transformer, Ensemble Learning
1 Introduction
Fashion-MNIST has become a widely used benchmark for evaluating image classification models that go beyond the original MNIST digit dataset. It consists of 60,000 training and 10,000 test grayscale images of size 28×28, covering ten clothing categories with moderate intra-class variation and inter-class similarity. Despite its apparent simplicity, Fashion-MNIST remains challenging for lightweight models, particularly when training time or model size is constrained.

Convolutional neural networks (CNNs) have long been the standard choice for image classification tasks. More recently, Vision Transformers (ViT) and hybrid CNN–Transformer architectures have demonstrated strong performance by capturing both local and global context. At the same time, generative models such as Variational Autoencoders (VAEs) have been used to learn robust latent representations that can improve downstream performance or provide auxiliary objectives. However, relatively few works explicitly combine CNN backbones, token-based Transformers, and VAEs into a single classification pipeline, especially for small benchmark datasets.

This project focuses on designing and implementing a compact but conceptually rich hybrid architecture for Fashion-MNIST. The main goal is not to beat all state-of-the-art results, but rather to demonstrate a novel and pedagogically clear pipeline that:

•	• Uses an EfficientNet-B0 backbone as a feature extractor.

•	• Converts mid-level feature maps into a sequence of tokens processed by a Transformer encoder.

•	• Regularizes token representations with a token-level VAE.

•	• Employs an ensemble of two models to further boost accuracy.

The remainder of this draft paper is organized as follows. Section 2 reviews recent related work on Fashion-MNIST and hybrid CNN–Transformer architectures. Section 3 describes the proposed ECTF+VAE methodology, including model design and training setup. Section 4 presents the experimental results and comparisons with prior work. Section 5 concludes and outlines potential extensions.
2 Literature Survey
This section summarizes at least ten recent or representative works related to Fashion-MNIST classification, hybrid CNN–Transformer models, and the use of VAEs in conjunction with Transformer-based vision models. These papers form the backdrop against which we position the proposed ECTF+VAE pipeline.
1.	Mukhamediev et al. (2024): propose a carefully tuned CNN with three convolutional layers (CNN-3-128) and report state-of-the-art results on Fashion-MNIST, achieving up to 99.65% test accuracy. Their work shows that, with heavy architectural search and training, very high performance can be reached on this dataset, but at the cost of increased complexity and over-specialization to Fashion-MNIST.
2.	Kadam et al. (2020): analyze several CNN variants for MNIST and Fashion-MNIST. For Fashion-MNIST, they obtain around 93–94% test accuracy using relatively shallow CNNs, highlighting the gap between standard CNN baselines and heavily optimized models.
3.	Long et al. (2024): provide a review of hybrid CNN–Vision Transformer (ViT) models. They conclude that combining convolutional feature extractors with Transformer blocks often yields better performance and sample efficiency than using either architecture alone, especially on mid-sized datasets.
4.	Sireesha et al. (2025): introduce a hybrid EfficientNet–Transformer framework for retinal fundus image classification. EfficientNet is used as the backbone, while Transformer layers capture inter-feature relationships. The model achieves 99% test accuracy on a combined fundus dataset, showing the potential of EfficientNet–Transformer hybrids in medical imaging.
5.	Tanwar et al. (2025): present a hybrid EfficientViT framework that couples EfficientNet-B0 with a Vision Transformer for gastrointestinal disease classification. Their results demonstrate that EfficientNet-B0 plus ViT can outperform pure CNNs on challenging medical datasets, supporting our choice of EfficientNet-B0 as the backbone in ECTF+VAE.
6.	Ashoka et al. (2025): propose EffiViT, a hybrid CNN–Transformer model based on EfficientNet-B4 and ViT for retinal disease classification. They emphasize the complementary strengths of CNNs for local patterns and Transformers for global context, achieving significant improvements in AUC and F1-score compared to CNN-only baselines.
7.	Aboulmira et al. (2025): combine wavelet decomposition with EfficientNet architectures for skin cancer classification. Although they do not use Transformers, their method shows that augmenting EfficientNet backbones with additional feature transformations can substantially improve performance.
8.	Alorf et al. (2025): develop a hybrid Transformer–CNN framework for multi-stage Alzheimer’s disease classification using rs-fMRI and clinical data. The model fuses CNN-derived local features with Transformer-based global attention and reports around 96% accuracy, illustrating the effectiveness of hybrid designs beyond natural images.
9.	Isinkaye et al. (2025): propose a multi-class hybrid VAE–Vision Transformer model for plant disease recognition. They show that integrating a VAE with a ViT backbone can enhance accuracy and robustness, particularly under data scarcity, which inspires our use of a token-level VAE as a regularizer in ECTF+VAE.
10.	Arshad et al. (2024): describe a hybrid convolution–Transformer network for hyperspectral image classification. The model uses a residual 3D CNN and a ViT-like module, achieving strong results and reinforcing the idea that CNN–Transformer hybrids generalize well across different imaging modalities.
11.	Tanwar et al. (2025, EfficientViT review): also discuss design considerations for combining EfficientNet backbones with Transformer modules, including tokenization strategies and fusion mechanisms. Our work differs by focusing on mid-level tokenization plus a VAE regularizer in a compact setting for Fashion-MNIST.
Across these works, several themes emerge: (i) Fashion-MNIST can be pushed above 99% with highly tuned CNN architectures, (ii) hybrid CNN–Transformer models are increasingly popular, and (iii) VAEs can complement Transformers by providing generative regularization. However, few works explicitly explore a lightweight EfficientNet-B0 + token Transformer + token-VAE pipeline on Fashion-MNIST with an emphasis on modularity and low training epochs. This gap motivates our approach.
3 Proposed Methodology: ECTF+VAE Pipeline
This section describes the architecture and training strategy implemented in the project code. The core idea is to build a modular pipeline that combines a pretrained EfficientNet-B0 backbone, a token-based Transformer encoder, a token-level VAE regularizer, and an ensemble inference strategy.
3.1 Overall Architecture
The proposed ECTF+VAE model consists of the following stages:

•	Input preprocessing: Fashion-MNIST grayscale images (28×28) are resized to 128×128 and replicated across three channels to match the expected input of EfficientNet-B0. Standard normalization with mean 0.5 and standard deviation 0.5 is applied.
•	EfficientNet-B0 backbone: A pretrained EfficientNet-B0 model from the timm library is used with features_only=True and out_indices=[4]. This returns a mid-level feature map f ∈ ℝ^{B×C×H×W} that contains rich spatial and semantic information.
•	Token projection: A 1×1 convolution projects the C-channel feature map into D=TOKEN_DIM channels, producing t ∈ ℝ^{B×D×H×W}. This tensor is then reshaped into a sequence of T=H·W tokens of dimension D.
•	Positional encoding + Transformer encoder: A lightweight Transformer encoder (1–2 layers, 4 heads, GELU activations) processes the token sequence. Sinusoidal 1D positional encodings are added to the tokens to preserve spatial order. The output tokens are then mean-pooled to yield a global token representation.
•	Token-VAE regularizer: The output tokens are also passed to a token-level VAE. The encoder mean-pools tokens to a single D-dimensional vector, maps it to a latent vector z via μ and logσ², and the decoder reconstructs a D×H×W map. The VAE contributes a reconstruction loss (MSE between reconstructed and original projected feature map) and a KL divergence term.
•	Global CNN pooling: In parallel, the original EfficientNet feature map f is globally average-pooled to obtain a compact CNN feature vector.
•	Fusion and classifier: The mean-pooled Transformer token vector and the global CNN vector are concatenated to form a fused representation, which is fed to a small MLP classifier with GELU activation and dropout to output logits over the 10 Fashion-MNIST classes.
3.2 Training Objective
The total loss used to train ECTF+VAE is a weighted sum of three components:

•	Cross-entropy loss L_ce between the predicted logits and the ground-truth class labels, with optional label smoothing (ε=0.1).
•	Reconstruction loss L_rec, defined as the mean-squared error between the reconstructed token feature map from the VAE decoder and the original projected feature map.
•	KL divergence L_kl between the approximate posterior q(z|tokens) and a unit Gaussian prior, averaged over the batch.
The final objective is

    L_total = L_ce + λ · L_rec + β · L_kl

where λ = 1.0 and β = 1e−3 in our experiments. This choice encourages the model to remain primarily discriminative while gently regularizing the token space through the VAE.
3.3 Ensemble Strategy
To further improve robustness and accuracy, we train two independent instances of the ECTF+VAE model with different random seeds. At inference time, the ensemble prediction for a given input is obtained by averaging the logits from both models and taking the argmax. This simple logit-averaging ensemble often yields better generalization than any single model.
3.4 Editable Pipeline Diagram (Text Description)
Figure 1 describes the proposed pipeline in a form that can be easily converted into a block diagram using Microsoft Word shapes or any diagramming tool.
Figure 1: Proposed ECTF+VAE Pipeline
1. Input Image (28×28, grayscale) → Resize to 128×128, replicate channels → Normalization
2. EfficientNet-B0 Backbone → Mid-level Feature Map f (B×C×H×W)
3. Branch A (Tokens + Transformer + VAE):
   • 1×1 Conv Projection → t (B×D×H×W)
   • Reshape to tokens (B×T×D) + Positional Encoding
   • Transformer Encoder (L layers, H heads) → token outputs (B×T×D)
   • Mean Pool over tokens → token vector (B×D)
   • Token-VAE: encoder → latent z; decoder → reconstructed map (B×D×H×W)
4. Branch B (Global CNN Feature):
   • Global Average Pooling of f → CNN vector (B×C)
5. Fusion and Classification:
   • Concatenate token vector and CNN vector → fused representation
   • MLP classifier → logits for 10 classes
6. Losses:
   • Cross-entropy on logits + λ·MSE(reconstruction) + β·KL(z).
4 Experimental Results and Analysis
4.1 Experimental Setup
All experiments are conducted on the Fashion-MNIST dataset with the standard train/test split (60,000/10,000). Images are resized to 128×128, normalized, and augmented with random horizontal flips and small rotations during training. Models are trained using the AdamW optimizer with learning rate 3×10−4, weight decay 1×10−4, batch size 192, and StepLR scheduling (decay by 0.5 every 3 epochs). Each ECTF+VAE model is trained for 6 epochs on a GPU. For the baseline SimpleCNN model, we use a small 3-layer convolutional network trained under similar conditions.
4.2 Quantitative Results
Table 1 summarizes the main quantitative findings of this project.
Model	Test Accuracy	Macro F1-score	Notes
SimpleCNN baseline	≈ 0.60	≈ 0.60	Small 3-layer CNN, 2 epochs (sanity baseline).
ECTF (without VAE, single model)	0.9426	0.9427	EfficientNet-B0 + token Transformer, no VAE, 3 epochs.
ECTF+VAE (single model)	≈ 0.949–0.951	≈ 0.949	Token-VAE regularized variant, 6 epochs.
ECTF+VAE Ensemble (2 models)	0.9538	0.9537	Logit-averaged ensemble of two seeds, 6 epochs each.
The SimpleCNN baseline reaches only about 60% test accuracy and macro F1, confirming that naive CNNs are insufficient for strong performance on Fashion-MNIST. The initial ECTF model (EfficientNet-B0 + token Transformer, without VAE regularization) already achieves 94.26% accuracy after 3 epochs. Adding the token-level VAE and training for 6 epochs yields further improvements, with single-model ECTF+VAE reaching around 95% accuracy. Finally, ensembling two ECTF+VAE models pushes accuracy to 95.38% and macro F1 to 0.9537.
4.3 Comparison with Recent Work
Several recent works report higher absolute accuracies on Fashion-MNIST, in some cases above 99%. For example, Mukhamediev et al. (2024) achieve 99.65% accuracy using a heavily tuned CNN architecture with three convolutional layers and extensive hyperparameter optimization. Such models are highly specialized for Fashion-MNIST and may involve a larger number of parameters or longer training schedules.

In contrast, our ECTF+VAE pipeline is designed to be:

•	Modular: It decomposes the problem into clear stages (CNN backbone, token Transformer, VAE regularizer, fusion, and classifier), making it easier to adapt or extend to other datasets.
•	Hybrid and interpretable: By using mid-level tokenization and explicit fusion of CNN and Transformer features, the architecture makes it clear how local and global information interact.
•	Lightweight in training time: We demonstrate strong performance (95.38% accuracy) with only 6 epochs and an ensemble of two models, which is practical for educational and prototyping settings.
•	Methodologically novel: Unlike existing EfficientNet–Transformer hybrids that typically use patch-based tokenization and do not employ a VAE, our pipeline uses mid-level feature-map tokens and a token-VAE reconstruction objective to regularize the token space.
Given these design goals, ECTF+VAE can be considered a novel and practically useful architecture in the space of hybrid CNN–Transformer models for small-scale vision benchmarks. It demonstrates how combining EfficientNet, Transformers, VAEs, and ensembling can lead to a robust yet conceptually clean pipeline.
5 Conclusion and Future Work
This draft presented the design, implementation, and evaluation of the ECTF+VAE pipeline for Fashion-MNIST classification. The method combines an EfficientNet-B0 backbone, mid-level tokenization with a Transformer encoder, a token-level VAE regularizer, and a simple two-member ensemble. Experiments show that the ensemble reaches 95.38% accuracy and 0.9537 macro F1, significantly outperforming a simple CNN baseline and improving over a pure ECTF model without VAE.

Future work could explore: (i) more sophisticated fusion mechanisms (e.g., attention-based gating between CNN and Transformer features), (ii) multi-scale tokenization that uses features from several EfficientNet stages, (iii) extension to other datasets such as CIFAR-10 or medical image benchmarks, and (iv) deeper analysis of the learned token-VAE latent space for interpretability and anomaly detection.
References
[1] R. I. Mukhamediev et al., "State-of-the-Art Results with the Fashion-MNIST Dataset," Mathematics, 2024.
[2] S. S. Kadam et al., "CNN Model for Image Classification on MNIST and Fashion-MNIST," 2020.
[3] H. Long et al., "Hybrid Design of CNN and Vision Transformer: A Review," 2024.
[4] M. Sireesha et al., "Hybrid EfficientNet-Transformer Model for Fundus Image Classification," JATIT, 2025.
[5] V. Tanwar et al., "Hybrid deep learning framework based on EfficientViT for GI disease classification," Scientific Reports, 2025.
[6] D. V. Ashoka et al., "EffiViT: Hybrid CNN-Transformer for Retinal Imaging," 2025.
[7] A. Aboulmira et al., "Hybrid Model with Wavelet Decomposition and EfficientNet for Skin Cancer Classification," 2025.
[8] A. Alorf et al., "Transformer and Convolutional Neural Network: A Hybrid Model for Alzheimer’s Classification," Mathematics, 2025.
[9] F. O. Isinkaye et al., "A multi-class hybrid variational autoencoder and vision transformer model for plant disease recognition," 2025.
[10] T. Arshad et al., "A hybrid convolution transformer for hyperspectral image classification," 2024.
