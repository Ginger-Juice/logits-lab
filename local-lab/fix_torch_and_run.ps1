$ErrorActionPreference='Continue'
$L='D:\knowleen\local-lab'; $P="$L\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "$L\wheels" | Out-Null
$w="$L\wheels\torch-2.7.1+cu118-cp312-cp312-win_amd64.whl"
"download start $(Get-Date)"
curl.exe -L -sS --retry 5 -C - -o $w "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torch-2.7.1+cu118-cp312-cp312-win_amd64.whl"
"download exit $LASTEXITCODE size $((Get-Item $w).Length) $(Get-Date)"
& $P -m pip install $w -i https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
"pip exit $LASTEXITCODE"
& $P -c "import torch;print('TORCH',torch.__version__,torch.version.cuda);x=torch.zeros(4,device='cuda');print('ALLOC_OK',x.sum().item(),torch.cuda.mem_get_info())"
if ($LASTEXITCODE -ne 0) { "CUDA_STILL_FAILS"; exit 1 }
& $P -m pip freeze > "$L\logs\pip_freeze.txt"
"STARTING_RUN $(Get-Date)"
Set-Location "$L\exp_min"
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'
& $P run_min.py *> "$L\logs\run_min.log"
"RUN_EXIT $LASTEXITCODE $(Get-Date)"