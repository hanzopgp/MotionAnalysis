Notes https://motion-gpt.github.io/

Quick summary :

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
