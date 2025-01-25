conda create -n geox python=3.10

conda activate geox


pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install flash-attn==2.5.9.post1 --no-build-isolation
