# Notes https://motion-gpt.github.io/

*Sentences written like this are my own deduction, question...*

## Links :

- Paper : https://arxiv.org/pdf/2306.14795.pdf
- Code : https://github.com/OpenMotionLab/MotionGPT
- Demo : https://huggingface.co/spaces/OpenMotionLab/MotionGPT

## Quick summary :

- Different tasks :
  - Motion Complete : predict next movement from motion tokens
  - Text-to-Motion : text description to 3D movement (also called motion generation when using motion tokens)
  - Motion-to-Text : 3D movement to text description (also called motion translation when using motion tokens)
  - Text-to-Text

- Abstract :
  - Motion can be perceived as a form of body language
  - Motion-language model can handle multiple motion-relevant tasks
  - Discrete vector quantization for human motion --> transfer 3D motion into motion tokens
  - Need to build motion vocabulary
  - Must treat human motion as a specific language
  - Pre-train on a mixture of motion-language data --> Fine-tune on prompt-based Q&A tasks
  - This way they achieve SOTA performances on text-driven motion generation, motion captioning...

- Method :
  - Motion tokenizer : convert raw motion data into discrete motion tokens
  - Motion-aware language model : learn to understand the motion tokens from pre-trained LLM by corresponding textual descriptions
  - How does it work :
    - Input : input motion (optional) and text. The input motion goes into a motion encoder. This part is the motion tokenizer.
    - Vocabulary : then both inputs goes into a text codebook and a motion codebook to create text and motion tokens.
    - Model : Language encoder takes these mixed tokens and produce Kencdec and Vencdec. These vectors go into the language decoders and produce output tokens (last layer linear + softmax).

## Paper notes :

### Introduction

- Emergence of multimodal stuff with GPT, Bert, T5...
- Need a motion motion one
- Could benefit to gaming, robotics, virtual assistant, human behavior analysis
- Previous research :
  - MDM is a motion diffusion model with conditional text tokens from CLIP
  - MLD integrated motion latent space to improve motion diffusion process
  - MotionCLIP and TM2T models the coupled relationship between motion and text description
- Problem is :
  - They treat motion and language as separate modalities
  - This requires strictly paired motion and text data
  - Task-specific superversions so it hardly generalize
  - Lack of comprehensive understanding of the relationship between motion and language
  - *This look like when NLP was task-specific and now we have LLMs*
- Goal is to build a motion-language model + uniform multi-task framework the can generalize on new task
- Main idea is to :
  - Follow vision-language pre-training from BEiT-3 to treat human motion as a specific foreign (body) language
  - Encode language and motion within a single vocabulary
  - This enables textual instructions like prompts in InstructGPT
- To generate human-like motions :
  - Learn a motion-specific vector quantized variational autoencoder (VQ-VAE) model to construct the motion vocabulary (learn a representation of motion data akin to english vocabulary)
  - Then these tokens are processed by a pre-trained language model that learns the underlying grammar and syntax of the motion language + relationship with corresponding textual descriptions
- Training :
  - Pre-train the language model on the raw motion dataset to learn motion language (only motion dataset) *(They predict next motion ? Autoregressiv model ?)*
  - Fine-tune on an instruction dataset (textual description + motion dataset) for prompt tuning + learning correlation between text and motion *(Supervised learning ? Need labels ?)*

### Related Work

- Human motion synthesis :
  - Goal : generating diverse human-like motion using text, action, incomplete motion...
  - Models : MDM / MLD / T2M-GPT
  - Problem : single model for multiple tasks
  - Solution : treat human motion as a foreign language (so we can use the power of pre trained language models in generalization and zero-shot transfer abilities)
- Human motion captioning
  - Goal : describing human motion with natural languages and learn the mapping from motion to language relying on two statistical models
  - Models : RNN / TM2T
  - Problem : constrained to bi-directional translation between text and motion
  - *Solution : treat motion as a foreign language to avoid translation*
- Language models and multi-model
  - Goal : bring images, audio, videos... to LLMs
  - Models : BERT / T5 / FLAN
  - *Problem : nothing for human motion*
  - *Solution : incorporate motion as a foreign language*
- Motion language pre-training
  - Goal : generate motion from textual description
  - Models : MotionCLIP
  - Problem : limited in supporting instructions from users like InstructGPT
 
### Method

#### 1. Motion tokenizer

- Goal is to represent motion in discrete tokens
- Data is 3D human motion
- Model is Vector Quantized Variational Autoencoder (VQ-VAE)
- Encoder generates discrete motion tokens (
  - Encoder applies 1D convolutions along the time dimension to obtain latent vectors z
  - Then z is transformed into a collection of codebook entries through discrete quantization
  - The learnable codebook Z consists of K latent embeddings vectors of dimension d
  - The process of quantization replaces each row vector b with its nearest codebook entry b_k in Z
- Decoder reconstruct the motion tokens
  - The motion decoder project all z back to raw motion space with M frames
- The input and output are motion tokens sequences
- There is 3 distinct loss functions fo training :
  - Lr is the reconstruction loss (use of L1 smooth loss + velocity regularization + exponential moving average (EMA) + codebook reset techniques)
  - Le is the embedding loss
  - Lc is the commitment loss

#### 2. Motion-aware language model

- 

#### 3. Training strategy

-

#### 4. Motion-language pre-training stage

-

#### 5. Instruction tuning stage

-


















