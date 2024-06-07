conda create -n llmcal python
conda activate llmcal
pip install -e .
mkdir ../affinecal
git clone https://github.com/LautaroEst/affinecal.git ../affinecal
pip install -e ../affinecal

if [ -z "${LIT_CHECKPOINTS}" ]; then
    echo "Select a directory to store the checkpoints:"
    read -p ">>> " LIT_CHECKPOINTS
    echo "" >> ~/.bashrc
    echo "export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}" >> ~/.bashrc
    export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}
fi