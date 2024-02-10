# llmtopic
A library that leverages large language models (llms) for extracting topics from text documents.

## Install
To install for CPU run `pip install llmtopic`.

To install with GPU support run `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llmtopic`.

If you are under Ubuntu and encounter errors regarding a failed build consider upgrading your cuda version to >=12.1. Also, it could be you have to point to the new nvcc version but modifying the install command as follows: `PATH="/usr/local/cuda-12.3/bin:$PATH" CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llmtopic`. Note that you of course have to adjsut that to your specific CUDA version.