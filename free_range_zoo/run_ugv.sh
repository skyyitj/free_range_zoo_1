export all_proxy="socks5://127.0.0.1:7890"
export ALL_PROXY="socks5://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"

export PYTHONPATH=$PYTHONPATH:/home/liuchi/yitianjiao/aamas2025/free-range-zoo


python main_ams.py env=wildfire model=gpt-4-0613 attempt_times=1 sample=9