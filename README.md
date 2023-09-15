# Transformers Combined - BERT, GPT and ViT 
This repo contains multiple transformers combined together into one. The function call is easy and can be done as shown below
```python
!git clone https://github.com/mkthoma/bert_gpt_vit_transformers.git

from bert_gpt_vit_transformers.library.combined_transformer import *
TransformerCombined("GPT", 20)
```
- For BERT, change the value to `BERT` and for ViT change the value to `VIT`. 

- The number of iterations or epochs it runs for can also be controlled by making use of  the second parameter. By default it runs for 10 iterations.

## BERT
BERT, short for Bidirectional Encoder Representations from Transformers, is a revolutionary deep learning model in the field of natural language processing (NLP). Introduced by Google AI in 2018, BERT has had a profound impact on various NLP tasks, from text classification to language generation. This short note provides an overview of BERT, its key components, and its significance in the world of NLP.

### The BERT Architecture

At its core, BERT is built upon the Transformer architecture, a neural network framework initially designed for sequence-to-sequence tasks. BERT's innovation lies in its ability to learn context from both left and right directions, making it bidirectional in contrast to previous models that were unidirectional.

Here are the main components of the BERT model:

1. Transformer Encoder: BERT utilizes the Transformer's encoder stack, which consists of multiple layers of self-attention mechanisms and feedforward neural networks. These layers enable BERT to capture contextual information from the input text.

2. Bidirectional Training: BERT is pre-trained on massive amounts of text data using two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). MLM involves randomly masking words in a sentence and training the model to predict those masked words based on the context. NSP, on the other hand, trains the model to determine whether a randomly selected sentence follows another sentence in a given document.

Since its introduction, BERT has fundamentally transformed the NLP landscape. It has paved the way for a new era of NLP models and inspired various follow-up architectures, including GPT-3, T5, and more. BERT-based models have become the foundation for numerous NLP applications, from chatbots to content recommendation systems.

In the future, research in NLP is likely to continue building upon BERT's success. Ongoing work includes improving model efficiency, multilingual support, and fine-tuning techniques. BERT's influence on NLP is set to endure, shaping the way we interact with and understand human language in the digital age.

In conclusion, BERT represents a significant breakthrough in natural language understanding. Its bidirectional context-awareness and transfer learning capabilities have redefined the state of the art in NLP, making it a landmark model in the field.

## GPT
The term "GPT" stands for "Generative Pre-trained Transformer," a series of cutting-edge deep learning models developed by OpenAI. Beginning with GPT-1 in 2018 and progressing to GPT-3 in 2020, these models have had a profound impact on the field of natural language processing (NLP) and have set new benchmarks for language generation, understanding, and many NLP tasks. This short note provides an overview of GPT, its key characteristics, and its significance in the realm of NLP.

### The GPT Architecture

At the core of GPT lies the Transformer architecture, a neural network framework initially designed for sequence-to-sequence tasks. GPT models are designed to work with text data and possess several fundamental components:

1. Transformer Decoder: Unlike the original Transformer architecture, which employed an encoder-decoder structure, GPT uses only the decoder component. This decoder is composed of multiple layers of self-attention mechanisms and feedforward neural networks.

2. Pre-training: GPT models are pre-trained on massive amounts of text data from the internet. During pre-training, they learn to predict the next word in a sentence based on the context provided by the preceding words. This process enables GPT to acquire a broad understanding of language, grammar, and world knowledge.

3. Generative Abilities: One of the most distinctive features of GPT models is their generative capability. They can produce coherent, contextually relevant text, making them highly versatile for tasks like text completion, text generation, and even creative writing.

The introduction of GPT models has had a transformative impact on various industries, including healthcare, customer service, content generation, and more. These models have sparked significant interest in ethical AI development and responsible AI use due to their potential to generate realistic but potentially biased or harmful content.

Looking forward, ongoing research aims to improve the efficiency and safety of large-scale language models like GPT. Additionally, there is a growing focus on developing models that better understand and reason about the content they generate, addressing concerns related to misinformation and biased output.

In conclusion, GPT represents a remarkable advancement in natural language processing. Its generative and pre-training capabilities have redefined the capabilities of AI in understanding and generating human language. As these models continue to evolve, they will likely play an increasingly central role in how we interact with, understand, and leverage language in the digital age.

## ViT
The Vision Transformer, often abbreviated as ViT, is a groundbreaking deep learning architecture that has revolutionized computer vision tasks. Introduced in 2020, ViT represents a departure from the conventional Convolutional Neural Networks (CNNs) that had long been the dominant approach for image processing. This note provides an overview of ViT, its key components, advantages, and its impact on the field of computer vision.

Since its introduction, ViT has gained widespread attention and adoption in the computer vision community. It has pushed the boundaries of what is possible in terms of image understanding and has sparked further research into improving ViT-based architectures. Ongoing work includes exploring hybrid models that combine ViT with other architectures and addressing challenges like handling high-resolution images efficiently.

### The Vision Transformer Architecture

At its core, ViT leverages the Transformer architecture, which was initially designed for natural language processing tasks. Unlike CNNs, which rely on convolutional layers to extract features from images, ViT directly applies the Transformer architecture to process the entire image as a sequence of patches. Here are the main components of a ViT model:

1. Patch Embedding: The input image is divided into a grid of non-overlapping patches. Each patch is then linearly embedded into a lower-dimensional vector. These patch embeddings serve as the model's input.

2. Positional Embeddings: To encode spatial information, ViT introduces positional embeddings, which are added to the patch embeddings. These positional encodings enable the model to understand the relative locations of different patches within the image.

3. Transformer Encoder: The heart of the ViT architecture is the Transformer encoder, which processes the patch embeddings along with positional encodings. It consists of multiple layers of self-attention mechanisms and feedforward neural networks. The self-attention mechanism allows ViT to capture global context information and dependencies between patches.

4. Classification Head: After processing the image patches through the Transformer encoder, ViT typically uses a classification head, which consists of one or more fully connected layers, to make predictions. For tasks like image classification, the output of the classification head corresponds to class probabilities.

In conclusion, the Vision Transformer (ViT) is a significant milestone in the field of computer vision. Its departure from traditional CNN-based approaches has led to new possibilities and opened doors for innovation in various image-related tasks. As research in this area continues to evolve, ViT is likely to play a pivotal role in shaping the future of computer vision.

## Training Logs

### BERT
![image](https://github.com/mkthoma/bert_gpt_vit_transformers/assets/135134412/b96029ed-f299-487c-afe2-599301fc66eb)


### GPT
![image](https://github.com/mkthoma/bert_gpt_vit_transformers/assets/135134412/a00a15d7-d4b9-4764-bb11-e9332a7dd0ac)

### ViT

![image](https://github.com/mkthoma/bert_gpt_vit_transformers/assets/135134412/147072b1-2034-474b-8dba-7031cb1feca1)

As we can from the logs, the training loss for all three transformers keep decreasing as we increase the number of iterations.