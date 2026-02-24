# Use a standard CUDA runtime base image that we know works for ComfyUI
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Critical for RTX 4090 (Ada Lovelace) compilation
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV MAX_JOBS=4

# Install basic dependencies
RUN apt-get update --yes && \
    apt-get install -y wget git libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]" runpod websocket-client librosa

WORKDIR /

# Setup ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

# ComfyUI Nodes
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    cd ComfyUI-MelBandRoFormer && pip install -r requirements.txt

# The WanVideoWrapper node (which uses SageAttention)
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# Force reinstall SageAttention & FlashAttention building from source with TORCH_CUDA_ARCH_LIST="8.9"
RUN pip uninstall -y sageattention flash-attn && \
    pip install flash-attn --no-build-isolation && \
    pip install sageattention

# Download Models
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors && \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors && \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors -O /ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors && \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors -O /ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors && \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors && \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors -O /ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors && \
    wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors && \
    wget -q https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors -O /ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors

# Downgrade transformers to avoid strict torch>=2.6.0 checks for loading .bin files (CVE-2025-32434 patch bypassed securely)
RUN pip install "transformers<4.49.0"

COPY . .
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
