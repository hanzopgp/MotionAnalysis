# Notes https://motion-gpt.github.io/

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

- Introduction :
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
    - **This look like when NLP was task-specific and now we have LLMs**
  - Goal is to build a motion-language model + uniform multi-task framework the can generalize on new task
  - 






















