# 先杀掉之前可能残留的假死进程
pkill -f "sglang serve"
pkill -f "webui.py"

# 使用 nohup 和 bash -c 激活环境并启动，彻底解决重定向失效问题
nohup bash -c "source /root/miniconda3/bin/activate qwen38 && bash /root/start.sh" > /root/qwen38_start.log 2>&1 &