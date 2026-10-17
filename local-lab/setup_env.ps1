$ErrorActionPreference='Continue'
$L='D:\knowleen\local-lab'
$env:PIP_DISABLE_PIP_VERSION_CHECK='1'
$env:PIP_CACHE_DIR="$L\.pipcache"
if (-not (Test-Path "$L\.venv\Scripts\python.exe")) { py -3.12 -m venv "$L\.venv" }
$P="$L\.venv\Scripts\python.exe"
& $P -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"=== TORCH cu126 ==="
& $P -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120
& $P -c "import torch;print('TORCH',torch.__version__,torch.version.cuda,torch.cuda.is_available())"
"=== OTHERS ==="
& $P -m pip install "transformers==5.12.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
if ($LASTEXITCODE -ne 0) { "5.12.1 unavailable, installing latest"; & $P -m pip install -U "transformers>=5.8" -i https://pypi.tuna.tsinghua.edu.cn/simple }
& $P -m pip install accelerate modelscope safetensors numpy scikit-learn matplotlib pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
"=== FLA (optional) ==="
& $P -m pip install flash-linear-attention -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60
"FLA exit $LASTEXITCODE"
"=== FREEZE ==="
& $P -m pip freeze
"INSTALL_DONE"