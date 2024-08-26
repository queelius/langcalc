# Autoregressive Models: Inductive Biases and Projections

This paper investigates how inductive biases, particularly projections onto training data, can be utilized in autoregressive (AR) models to improve out-of-distribution (OOD) generalization.

Our approach consists of the following steps:

1. **Infini-gram Model Exploration**: We begin by considering the infinite-gram model, which uses suffix arrays to efficiently handle arbitrary input (context) lengths. Infinite-gram models are noteworthy due to their ability to scale to any Markov order. For reference, see the paper *Infini-gram: An Engine for n-gram / ∞-gram Language Modeling with Trillion-Token Corpora* by Jiacheng Liu et al. (https://huggingface.co/papers/2401.17377). A brute-force approach could involve storing the training data as-is and searching for the longest matching suffix in the input sequence.

2. **Projection-Based Inductive Biases**: We explore a subset of inductive biases where inputs are projected onto the training data. Additionally, we consider mapping output tokens from the training data back onto the context to maintain coherence. For example, if the input is "the king demanded" and "king" is substituted with "dictator," we could implement a token mapping to revert "dictator" to "king" in subsequent outputs. However, this approach introduces potential challenges in maintaining consistency.

3. **Exploring Variations and Extensions**: In the default implementation, the model finds the longest matching suffix, exhibiting a recency bias. However, there are many possible variations to consider. Given the rich history of n-gram models, revisiting these concepts in the context of Infini-gram models and modern LLMs is worthwhile. While we do not expect to match the OOD generalization capabilities of models like GPT, we anticipate developing more sophisticated projection functions that surpass the simple longest-suffix matching approach.

By framing OOD generalization as a projection problem, we propose strategies to optimize these projections using machine learning techniques. Additionally, we explore the integration of classical information retrieval methods and pre-trained language model embeddings to improve the semantic relevance of these projections.

We are also developing code for a Python library to experiment with various inductive biases. This library will enable running these simpler models alongside more powerful neural generative models like GPTs, where these models "learn" by simply accumulating more training data.
