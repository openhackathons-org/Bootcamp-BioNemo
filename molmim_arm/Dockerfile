# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# NIM-like MolMIM service for aarch64 + Blackwell (GB200/GB300), built on a
# stock NGC PyTorch base (which provides arm64 + Blackwell CUDA). It runs a
# pure-PyTorch MolMIM (molmim_torch.py) that loads the official molmim_70m_24_3
# checkpoint -- no NeMo / Megatron / apex / TransformerEngine required.
#
# The MolMIM checkpoint (molmim_70m_24_3.nemo) is mounted at runtime into
# /models (it is not baked into the image).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}

# FastAPI service (like the BioNeMo NIMs). torch+numpy come from the base image.
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2,<3"

COPY molmim_torch.py engine.py server.py /opt/molmim-shim/
COPY entrypoint.sh /opt/molmim-shim/entrypoint.sh
RUN chmod +x /opt/molmim-shim/entrypoint.sh

ENV NIM_HTTP_API_PORT=8001
ENV MOLMIM_CHECKPOINT=/models/molmim_70m_24_3.nemo
EXPOSE 8001

ENTRYPOINT ["/opt/molmim-shim/entrypoint.sh"]
