# AI tools
This page contains a collection of opensource AI models and tools available for various use cases

---
## Vision
### Generative modeling
#### 1. Stable Diffusion (2022 Aug 10)
- **Summary**:
- **Resources**
    - [Stable Diffusion Launch Announcement](https://stability.ai/blog/stable-diffusion-announcement)
    - [Stable Diffusion Public Release](https://stability.ai/blog/stable-diffusion-public-release)
    - [stable-diffusion - Github](https://github.com/CompVis/stable-diffusion)
    - [stable-diffusion - Huggingface](https://huggingface.co/CompVis/stable-diffusion)
- **Projects**
    1. [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) - Port for Apple Silicon + CoreML
    2. [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) - fast-stable-diffusion, +25-50% speed increase + memory efficient + DreamBooth
    3. [Lsmith](https://github.com/ddPn08/Lsmith) - StableDiffusionWebUI accelerated using TensorRT
    4. [ControlNet](https://stable-diffusion-art.com/controlnet/) - copy compositions or human poses from a reference image
        - [ControlNet v1.1 - A Complete Guide](https://stable-diffusion-art.com/controlnet/)
    5. [imaginAIry - Github](https://github.com/brycedrennan/imaginAIry) - AI imagined images. Pythonic generation of stable diffusion images.
#### 2. Grounded-SAM ()
- **Summary**: Marrying Grounding DINO with Segment Anything & Stable Diffusion & BLIP - Automatically Detect , Segment and Generate Anything with Image and Text Inputs
- **Resources**:
    - [Grounded-Segment-Anything - Github](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- **Projects**
    - [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) - Segment and Recognize Anything at Any Granularity
#### 3. AnimateDiff
- **Summary**: Combine static images with motion dynamics
- **Resources**:
    - [AnimateDiff - Project Site](https://animatediff.github.io/)
#### 4. PhotoMaker (2024 Jan)
- **Summary**: Create photos/paintings/avatars of anyone in any style within seconds
- **Resources**:
    - [Photomaker - Project page](https://photo-maker.github.io/)
    - [PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding - Paper](https://arxiv.org/abs/2312.04461)
    - [Photomaker - Github](https://github.com/TencentARC/PhotoMaker)
#### 5. DragGAN (2023 May)
- **Summary**
- **Resources**:
    - [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold - Project page](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
    - [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold - Paper](https://arxiv.org/abs/2305.10973)
    - [DragGAN - Github](https://github.com/XingangPan/DragGAN)
### Image Inpainting
#### 1. lama-cleaner (2022 Nov)
- **Summary**: Image inpainting tool powered by SOTA AI Model. Remove any unwanted object, defect, people from your pictures or erase and replace(powered by stable diffusion) any thing on your pictures.
- **Resources**
    - [lama-cleaner - Github](https://github.com/Sanster/lama-cleaner)
### Object detection
#### 1. YOLOv8
- **Summary**: YOLOv8 in PyTorch > ONNX > CoreML > TFLite. Can do detection, segmentation and much more.
- **Resources**
    - [ultralytics - Github](https://github.com/ultralytics/ultralytics)
#### 2. Face Recognition
- **Summary**: 2D and 3D Face alignment library build using pytorch
- **Resources**
    - [1adrianb/face-alignment - Github](https://github.com/1adrianb/face-alignment)
### Image Segmentation
#### 1. SAM (2023 Apr 5) (License: Apache 2.0)
- **Summary**: high quality object masks from input prompts such as points or boxes
- **Resources**:
    - [Introducing Segment Anything - Blog](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)
    - [Website demo](https://segment-anything.com/)
    - [segment-anything - Github](https://github.com/facebookresearch/segment-anything)
- **Projects**
    1. [sam-hq](https://github.com/SysCV/sam-hq) - Segment Anything in High Quality
    2. [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM) - Fast Segment Anything
    3. [sam.cpp](https://github.com/YavorGIvanov/sam.cpp) - Inference of Meta's Segment Anything Model in pure C/C++
#### 2. Detic (2021 Jan)
- **Summary**: A **Det**ector with **i**mage **c**lasses that can use image-level labels to easily train detectors, detects any given class names
- **Resources**:
    - [Detecting Twenty-thousand Classes using Image-level Supervision - Paper](https://arxiv.org/abs/2201.02605)
    - [Replicate - Interactive demo](https://replicate.com/facebookresearch/detic)
    - [Detic - Github](https://github.com/facebookresearch/Detic)

### Image embeddings
#### 1. DINO
- **Summary**: high-performance visual features that can be directly employed with classifiers as simple as linear layers on a variety of computer vision tasks
- **Resources**:
    - [DINOv2: State-of-the-art computer vision models with self-supervised learning - Blogpost](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
    - [DINOv2: Learning Robust Visual Features without Supervision - Paper](https://arxiv.org/abs/2304.07193)
    - [dinov2 - Github](https://github.com/facebookresearch/dinov2)
### Video

### Object tracking
#### 1. TrackHQ (2023 Jul)
- **Summary**: Tracking Anything in High Quality
- **Resources**
    - [HQTrack - Github](https://github.com/jiawen-zhu/HQTrack)
    - [Technical Report](https://arxiv.org/abs/2307.13974)
### Feature matching
#### 1. LightGlue (2023 June 26)
- **Summary**: a lightweight feature matcher with high accuracy and blazing fast inference
- **Resources**: 
    - [Paper: LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643)
    - [LightGlue - Github](https://github.com/cvg/LightGlue)

---
## Speech
### Speech recognition
#### 1. OpenAI Whisper (2022 Sept 21)
- **Summary**: Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
- **Resources**
    - [Introducing Whisper - Blog](https://openai.com/blog/whisper/)
    - [Robust Speech Recognition via Large-Scale Weak Supervision - Paper](https://cdn.openai.com/papers/whisper.pdf)
        - [arxiv](https://arxiv.org/abs/2212.04356)
    - [whisper - Github](https://github.com/openai/whisper) (**License**: MIT)
- **Projects**:
    1. [distil-whisper](https://github.com/huggingface/distil-whisper) - 6x faster, 50% smaller, within 1% word error rate.
    2. [Talk to your multi-lingual AI assistant](https://huggingface.co/spaces/ysharma/Talk_to_Multilingual_AI_WhisperBloomCoqui) - Uses Whispher, GPT-3 and Coqui-TTS
    3. [Transcribe Youtube Video to text with OpenAI Whispher - YouTube](https://www.youtube.com/watch?v=xam2U8loUvA) - Using pytube and whispher
    4. [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Port in C/C++, runs in CPU including mobile and rpi.
    5. [Whisper](https://github.com/Const-me/Whisper) - High-performance GPGPU inference for Windows
    6. [whispherX](https://github.com/m-bain/whisperX) - Timestamp-Accurate Automatic Speech Recognition using Force Alignment
    7. [faster-whispher](https://github.com/guillaumekln/faster-whisper) - Faster Whisper transcription with CTranslate2
    8. [whispher-jax](https://github.com/sanchit-gandhi/whisper-jax) - optimised JAX code Whisper
---
## Text
### Text generation
#### 1. BLOOM (2022 July)
- **Summary**: BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text data using industrial-scale computational resources. As such, it is able to output coherent text in 46 languages and 13 programming languages that is hardly distinguishable from text written by humans. BLOOM can also be instructed to perform text tasks it hasn't been explicitly trained for, by casting them as text generation tasks.
- **Resources**
    - [Introducing The World’s Largest Open Multilingual Language Model: BLOOM - Blog](https://bigscience.huggingface.co/blog/bloom)
    - [BLOOM Model Card - Huggingface](https://huggingface.co/bigscience/bloom) (**License**: Responsible AI License)
    - [tr11-176B-ml - Github](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml)
- **Projects**
    1. [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp) - C++ implementation for BLOOM Inference
#### 2. GALACTICA (2022 Nov)
- **Summary**: A general-purpose scientific language model. It is trained on a large corpus of scientific text and data. It can perform scientific NLP tasks at a high level, as well as tasks such as citation prediction, mathematical reasoning, molecular property prediction and protein annotation.
- **Resources**
    - [Galactica online demo](https://galactica.org/)
    - [Galactica: A Large Language Model for Science - Paper](https://galactica.org/static/paper.pdf)
    - [galai - Github](https://github.com/paperswithcode/galai) (**License**: Code - Apache 2.0, Model - CCA-NC4.0-PIL)
#### 3. GPT-GJT (Dec 2022)
- **Summary**: a variant forked off GPT-J (6B), and performs exceptionally well on text classification and other tasks
- **Resources**
    - [GPT-JT-6B-v1 - HuggingFace](https://huggingface.co/togethercomputer/GPT-JT-6B-v1)
    - [Releasing v1 of GPT-JT powered by opensource AI - Blog](https://www.together.xyz/blog/releasing-v1-of-gpt-jt-powered-by-open-source-ai)
#### 4. PubMed GPT 2.7B (2022 Dec)
**Summary:** A language model trained on biomedical literature which delivers an improved state of the art for medical question answering.
- **Resources**
    - [PubMedGPT 2.7B - Official blog](https://crfm.stanford.edu/2022/12/15/pubmedgpt.html)
    - [pubmedgpt - HuggingFace](https://huggingface.co/stanford-crfm/pubmedgpt)
    - [pubmedgpt - Github](https://github.com/stanford-crfm/pubmedgpt)
#### 5. nanoGPT (2022 Dec)
- **Summary**: The simplest, fastest repository for training/finetuning medium-sized GPTs
- **Resources**
    - [nanoGPT - Github](https://github.com/karpathy/nanoGPT) (**License**: MIT)
#### 7. Petals (2022 Dec)
- **Summary**: Run 100B+ language models at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading
- **Resources**:
    - https://petals.ml/
    - https://github.com/bigscience-workshop/petals
#### 8. Chat-RWKV (Jan 2023)
- **Summary**: ChatRWKV is like ChatGPT but powered by the RWKV (100% RNN) language model, and open source.
- **Resources**:
    - https://github.com/BlinkDL/ChatRWKV
    - https://github.com/BlinkDL/RWKV-LM
    - https://huggingface.co/BlinkDL
#### 9. LLaMA (Feb 24, 2023)
- **Summary**: Large Language Model Meta AI
- **Resources**:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model - Blog](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [Paper](https://arxiv.org/abs/2302.13971)]
- **Projects**
    1. [open_llama](https://github.com/openlm-research/open_llama) - a permissively licensed open source reproduction
    2. [LLaMa - facebookresearch](https://github.com/facebookresearch/llama) - Minimal project for inference
    3. [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference with C/C++
    4. [dalai](https://github.com/cocktailpeanut/dalai) - The simplest way to run LLaMA on your local machineml
    5. [llama-rs](https://github.com/setzer22/llama-rs) - Run LLaMA inference on CPU, with Rust
    6. [alpaca-lora](https://github.com/tloen/alpaca-lora) - Instruct-tune LLaMA on consumer hardware
    7. [vicuna](https://vicuna.lmsys.org/) - an open-source chatbot trained by fine-tuning LLaMA
    8. [FastChat - Github](https://github.com/lm-sys/FastChat)
    9. [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)
    10. [lit-LLaMA](https://github.com/Lightning-AI/lit-llama) - Implementation of the LLaMA language model based on nanoGPT (**Commercial Use**)
    11. [Open-Llama](https://github.com/s-JoL/Open-Llama) - Train Llama model
    12. [open_llama](https://github.com/openlm-research/open_llama) - OpenLLaMA, a permissively licensed open source reproduction of Meta AI’s LLaMA 7B trained on the RedPajama dataset
    13. [llama2.c](https://github.com/karpathy/llama2.c) - Inference Llama 2 in one file of pure C
    14. [llama-dfdx](https://github.com/coreylowman/llama-dfdx) - LLaMa 7b with CUDA acceleration implemented in rust. Minimal GPU memory needed!
    15. [llama2.mojo](https://github.com/tairov/llama2.mojo) - Inference Llama 2 in one file of pure
#### 11. Falcon
- **Summary**: LLM for research and commercial purposes. Allows commercial use upto $1M revenue.
- **Resources**:
    - [Falcon LLM - Home](https://falconllm.tii.ae/)
    - [Huggingface models](https://huggingface.co/tiiuae)
#### 12. FinGPT (2023 Jun)
- **Summary**: Data-Centric FinGPT. Open-source for open finance!
- **Resources**
    - [FinGPT - Github](https://github.com/AI4Finance-Foundation/FinGPT)
    - [FinNLP - Website](https://ai4finance-foundation.github.io/FinNLP/)
#### 13. Llama2
- **Summary**: Open-source LLM free for research and commercial\
- **Resources**
    - [Meta and Microsoft Introduce the Next Generation of Llama - Blog post](https://ai.meta.com/blog/llama-2/)
    - [Llama 2 Portal](https://ai.meta.com/llama/)
    - [Llama 2 is here - get it on Hugging Face](https://huggingface.co/blog/llama2)
    - [Github issue discussing hardware requirements](https://github.com/facebookresearch/llama/issues/425)
- **Projects**
    1. [Llama2-Onnx](https://github.com/microsoft/Llama-2-Onnx) - an optimized version of the Llama 2 model
    2. [llama-recipes](https://github.com/facebookresearch/llama-recipes/) - Examples and recipes for Llama 2 model
#### 14. Mistral 7B
- **Summary**: 7B model with Apache license, commercial use
- **Resources**
    - [Announcing Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) - Blog post
    - [mistral-src](https://github.com/mistralai/mistral-src) - Inference code
    - [Huggingface](https://huggingface.co/mistralai) - Model in hub
### Embeddings
#### 1. **StarSpace (2017)**
- **Summary**: Learning embeddings for classification, retrieval and ranking.
- **Resources**: 
    1. [Paper](https://arxiv.org/abs/1709.03856)
    2. [Github](https://github.com/facebookresearch/StarSpace)
#### 2. Jina Embeddings-v2
- **Summary**: Model that can support upto 8K context length
- **Resources**
    - [HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
    - [Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models](https://arxiv.org/abs/2307.11224)
---
## Image - Language
#### 1. OpenCLIP ()
- **Summary**: An open source implementation of CLIP.
- **Resources**:
    - [Reproducible scaling laws for contrastive language-image learning - Paper](https://arxiv.org/abs/2212.07143)
    - https://github.com/mlfoundations/open_clip
#### 2. IF ()
- **Summary**: a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding
- **Resources**:
    - https://github.com/deep-floyd/IF
    - [Running IF with diffusers on a Free Tier Google Colab - Blog post](https://huggingface.co/blog/if)
#### 3. TinyGPT-V
- **Summary**: Efficient Multimodal Large Language Model via Small Backbones. Requires a 24G GPU for training and an 8G GPU or CPU for inference.
- **Resources**:
    - [TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones](https://arxiv.org/abs/2312.16862) - Research paper
    - [TinyGPT-V - Github](https://github.com/DLYuanGod/TinyGPT-V) - Code
#### 4. LLaVa (2023 Apr)
- **Summary**:
- **Resources**:
    - [LLaVA: Large Language and Vision Assistant - Project page](https://llava-vl.github.io/)
    - [Demo](https://llava.hliu.cc/)
    - Research papers
        - [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (NeurIPS 2023 Oral)
        - [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) (LLaVa 1.5)
    - [LLaVa - Github](https://github.com/haotian-liu/LLaVA)
#### 5. moondream (2024 Jan)
- **Summary**: a tiny (1.6B) vision language model that kicks ass and runs anywhere
- **Resources**:
    - [moondream - Github](https://github.com/vikhyat/moondream)
    - [moondream - Huggingface Spaces](https://huggingface.co/spaces/vikhyatk/moondream1)
---
## Speech - Language
### Text to Speech
#### 1. Coqui-TTS
- **Summary**: A deep learning toolkit for Text-to-Speech, battle-tested in research and production.
- **Resources**
    - [Double Decode Consistency - Blog](https://coqui.ai/blog/tts/solving-attention-problems-of-tts-models-with-double-decoder-consistency)
    - [Coqui-TTS Samples - Blog](https://erogol.com/ddc-samples/)
    - [TTS - Github](https://github.com/coqui-ai/TTS)
#### 2. TorToiSe
- **Summary**: A multi-voice TTS system trained with an emphasis on quality
- **Resources**
    - [tortise - Github](https://github.com/neonbjb/tortoise-tts)
#### 3. **AudioGPT**
- **Summary**: Understanding and Generating Speech, Music, Sound, and Talking Head
- **Resources**:
    - [AudioGPT - Github](https://github.com/AIGC-Audio/AudioGPT)
#### 4. suno-ai/bark
- **Summary**: Text-Prompted Generative Audio Model
- **Resources**
    - [bark - Github](https://github.com/suno-ai/bark)

#### 5. EmotiVoice
- **Summary**: a powerful and modern open-source text-to-speech engine. EmotiVoice speaks both English and Chinese, and with over 2000 different voices. The most prominent feature is emotional synthesis, allowing you to create speech with a wide range of emotions, including happy, excited, sad, angry and others.
- **Resources**
    - [EmotiVoice - Github](https://github.com/netease-youdao/EmotiVoice)

### Speech to Text
#### 1. Coqui - STT
- **Summary**: An open-source deep-learning toolkit for training and deploying speech-to-text models.
- **Resources**:
    - [Documentation](https://stt.readthedocs.io/en/latest/)
    - [STT - Github](https://github.com/coqui-ai/STT)
---
## Tabular data
### Transformers
#### 1. Tab Transformers
- **Summary**: Attention network for tabular data
- **Resources**
    - [Paper - TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)
    - [lucidrains/tab-transfor-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)
---
## 3D rendering
### NeRF
#### 1. NVIDIA Instant-NGP
- **Summary**: Instant neural graphics primitives: lightning fast NeRF and more
- **Resources**
    - [instant-ngp](https://github.com/NVlabs/instant-ngp) (**License**: NVIDIA Custom License)
    - [Getting started with NVIDIA Instant NeRFs](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/)
#### 2. Shap-E (2023, May 3)
- **Summary**: Generate 3D objects conditioned on text or images
- **Resources**
    - [shap-e - Github](https://github.com/openai/shap-e)
    - [Shap-E: Generating Conditional 3D Implicit Functions - Paper](https://arxiv.org/abs/2305.02463)
#### 3. Neuralangelo (Jun 2023)
- **Summary**: 
- **Resources**:
    - [Neuralangelo: High-Fidelity Neural Surface Reconstruction ](https://research.nvidia.com/labs/dir/neuralangelo/)

---
# AI Tools

## Language
#### 1. langchain
- **Summary**: Building applications with LLMs through composability
- **Resources**:
    - [langchain - Github](https://github.com/hwchase17/langchain)
    - [Getting started with LangChain - Towards Datascience](https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c)
- **Projects**
    1. [langflow](https://github.com/logspace-ai/langflow) - LangFlow is a UI for LangChain
    2. [flowise](https://github.com/FlowiseAI/Flowise) - Drag & drop UI to build your customized LLM flow using LangchainJS
    3. [awesome-langchain](https://github.com/kyrolabs/awesome-langchain) - Awesome list of tools and projects with the awesome LangChain framework
#### 2. xturing
- **Summary**: Build and control your own LLMs
- **Resources**:
    - [xturing - Github](https://github.com/stochasticai/xturing)
#### 3. LocalAI
- **Summary**: Self-hosted, community-driven simple local OpenAI-compatible API written in go
- **Resources**:
    - [LocalAI - Github](https://github.com/go-skynet/LocalAI)
#### 4. Lamini
- **Summary**: The LLM engine for rapidly customizing models. Allows commercial use!
- **Resources**:
    - [Introducing Lamini - Blog](https://lamini.ai/blog/introducing-lamini)
    - [lamini - Github](https://github.com/lamini-ai/lamini)
#### 5. CodeTF
- **Summary**: One-stop Transformer Library for State-of-the-art Code LLM
- **Resources**
    - [CodeTF](https://github.com/salesforce/CodeTF)
#### 6. MLC-LLM (2023 Mar)
- **Summary**: Enable everyone to develop, optimize and deploy AI models natively on everyone's devices.
- **Resources**
    - [MLC LLM - Blog](https://mlc.ai/mlc-llm/)
    - [mlc-llm - Github](https://github.com/mlc-ai/mlc-llm)
    - [Bringing Hardware Accelerated Language Models to Consumer Devices - Project page](https://mlc.ai/blog/2023/05/01/bringing-accelerated-llm-to-consumer-hardware)
#### 7. GPT4All
- **Summary**: Open-source large language models that run locally on your CPU and nearly any GPU
- **Resources**:
    - [Technical Report](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)
    - [gpt4all - Github](https://github.com/nomic-ai/gpt4all)
#### 8. OpenChatKit (2023 Mar 10)
- **Summary**: OpenChatKit provides a powerful, open-source base to create both specialized and general purpose models for various applications. The kit includes an instruction-tuned language models, a moderation model, and an extensible retrieval system for including up-to-date responses from custom repositories
- **Resources**:
    - [Announcing OpenChatKit](https://www.together.xyz/blog/openchatkit)
    - [OpenChatKit - Github](https://github.com/togethercomputer/OpenChatKit)
#### 9. FreedomGPT
- **Summary**: A React and Electron-based app that executes the FreedomGPT LLM locally (offline and private) on Mac and Windows using a chat-based interface (based on Alpaca Lora)
- **Resources**
    - https://github.com/ohmplatform/FreedomGPT
#### 10. Open-Assistant
- **Summary**: Open Assistant is a project meant to give everyone access to a great chat based large language model.
- **Resources**:
    - https://projects.laion.ai/Open-Assistant/
    - https://github.com/LAION-AI/Open-Assistant
#### 11. SuperAGI
- **Summary**: A dev-first open source autonomous AI agent framework. Enabling developers to build, manage & run useful autonomous agents quickly and reliably.
- **Resources**
    - [SuperAGI - Github](https://github.com/TransformerOptimus/SuperAGI)

#### 12. exllamav2
- **Summary**: A fast inference library for running LLMs locally on modern consumer-class GPUs
- **Resources**
    - [exllamav2 - Github](https://github.com/turboderp/exllamav2)

#### 13. QAnything
- **Summary**: a local knowledge base question-answering system designed to support a wide range of file formats and databases, allowing for offline installation and use
- **Resources**:
    - [QAnything - Github](https://github.com/netease-youdao/QAnything)

#### 14. llmware
- **Summary**:  Providing enterprise-grade LLM-based development framework, tools, and fine-tuned models. 
- **Resources**:
    - [llmware - Github](https://github.com/llmware-ai/llmware)
---
## Vision
#### 1. PixelLib
- **Summary**: a library for performing segmentation of objects in images and videos
- **Resources**
    - [Simplifying Object Segmentation with PixelLib Library - Paper](https://vixra.org/abs/2101.0122)
    - [PixelLib - Github](https://github.com/ayoolaolafenwa/PixelLib)
    - [Papers with Code](https://paperswithcode.com/paper/simplifying-object-segmentation-with-pixellib)
    - [Documentation](https://pixellib.readthedocs.io/en/latest/)
#### 2. StreamDiffusion
- **Summary**: A Pipeline-Level Solution for Real-Time Interactive Generation
- **Resources**
    - [StreamDiffusion - Github](https://github.com/cumulo-autumn/StreamDiffusion)
#### 3. Supervision
- **Summary**: We write your reusable computer vision tools.
- **Resources**:
    - [supervision - Github](https://github.com/roboflow/supervision)
---
## Video
#### 1. Roop (2023 Jun)
- **Summary**: one-click face swap
- **Resources**:
    - [roop - Github](https://github.com/s0md3v/roop)
#### 2. ShortGPT (2023 Jul)
- **Summary**: ShortGPT is a powerful framework for automating content creation. It simplifies video creation, footage sourcing, voiceover synthesis, and editing tasks.
- **Resources**:
    - [ShortGPT - Github](https://github.com/RayVentura/ShortGPT)
---
## Audio
#### 1. JARVIS
- **Summary**: a voice assistant made as an experiment using neural networks with Rust
- **Resources**:
    - [jarvis - Github](https://github.com/Priler/jarvis)
---
## Multi-modal
#### 1. TaskMatrix
- **Summary** - **TaskMatrix** connects ChatGPT and a series of Visual Foundation Models to enable **sending** and **receiving** images during chatting.
- **Resources**
    - [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models - Paper](https://arxiv.org/abs/2303.04671)
    - [TaskMatrix - Github](https://github.com/microsoft/TaskMatrix)
#### 2. Transformer Agents
- **Summary**: Multi modal AI agent
- **Resources**
    - [transformer-agents - HuggingFace](https://huggingface.co/docs/transformers/transformers_agents)
#### 3. LibreChat
- **Summary**: Enhanced ChatGPT Clone: Features OpenAI, GPT-4 Vision, Bing, Anthropic, OpenRouter, Google Gemini, AI model switching, message search, langchain, DALL-E-3, ChatGPT Plugins, OpenAI Functions, Secure Multi-User System, Presets, completely open-source for self-hosting. More features in development
- **Resources**:
    - [LibreChat - Github](https://github.com/danny-avila/LibreChat)
    - [Documentation](https://docs.librechat.ai/index.html)
---
## Dataset management
#### 1. fiftyone
- **Summary**: The open-source tool for building high-quality datasets and computer vision models
- **Resources**:
    - [fiftyone - Github](https://github.com/voxel51/fiftyone)
    - [docs](https://docs.voxel51.com/)
---
# AI Libraries
## General
1. [ColossalAI](https://github.com/hpcaitech/ColossalAI) - Making large AI models cheaper, faster and more accessible
## Vision
1. [monai](https://monai.io/) - medical imaging with deep learning
2. [supervision](https://github.com/roboflow/supervision) - We write your reusable computer vision tools
## Audio
1. [SpeechBrain](https://speechbrain.github.io/) - An Open-Source Conversational AI Toolkit
## Language
1. [OpenNMT](https://opennmt.net/) - An open source neural machine translation system
2. [outlines](https://github.com/normal-computing/outlines) - Neuro Symbolic Text Generation
3. [llm-foundry](https://github.com/mosaicml/llm-foundry) - LLM training code for MosaicML foundation models
4. [chainlit](https://github.com/Chainlit/chainlit) - Build Python LLM apps in minutes!
5. [languagemodels](https://github.com/jncraton/languagemodels) - Explore large language models on any computer with 512MB of RAM
6. [lit-gpt](https://github.com/Lightning-AI/lit-gpt) - Hackable implementation of state-of-the-art open-source LLMs based on nanoGPT. Supports flash attention, 4-bit and 8-bit quantization, LoRA and LLaMA-Adapter fine-tuning, pre-training. Apache 2.0-licensed.
## Multi-modal
1. [rasa](https://github.com/RasaHQ/rasa) - Open source machine learning framework to automate text- and voice-based conversations
    1. [RasaGPT](https://github.com/paulpierre/RasaGPT) - headless LLM chatbot platform
---
# Miscellaneous
## Model Zoo
1. [modelzoo.co](https://modelzoo.co/) - Discover open source deep learning code and pretrained models.
2. [OpenVINO Model Zoo](https://docs.openvino.ai/2021.4/model_zoo.html) - Model zoo from multiple sources
3. [replicate](https://replicate.com/explore) - easy to use setup for popular models
4. [modelscope](https://github.com/modelscope/modelscope) - bring the notion of Model-as-a-Service to life
5. https://civitai.com/
6. [open-llms](https://github.com/eugeneyan/open-llms) - A list of open LLMs available for commercial use.
## AI in the world
1. [AI Product Index](https://github.com/dair-ai/AI-Product-Index) - A curated index to track AI-powered products.
2. [awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) - A curated list of modern Generative Artificial Intelligence projects and services
3. [LinkedIn Post - Commercial use LLMs](https://www.linkedin.com/posts/sahar-mor_artificialintelligence-machinelearning-activity-7049789761728770049-QLsv?utm_source=share&utm_medium=member_desktop) -  List of commercially usable LLMs
4. [ai-collection](https://github.com/ai-collection/ai-collection) - A Collection of Awesome Generative AI Applications
5. [tuning-playbook](https://github.com/google-research/tuning_playbook) - A playbook for systematically maximizing the performance of deep learning models.
6. [ollama](https://ollama.ai/) - Get up and running with large language models, locally.
7. [inference](https://github.com/xorbitsai/inference) - Replace OpenAI GPT with another LLM in your app by changing a single line of code
8. [llama-embeddings-fastapi-service](https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service) - designed to facilitate and optimize the process of obtaining text embeddings using different LLMs
