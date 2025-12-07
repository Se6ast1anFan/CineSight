@echo off
:: 1. 自动激活 Conda 环境 'db'
:: 如果你不知道 conda 的路径，尝试直接写 call conda activate db
:: 这种写法是最稳妥的：
call conda activate db

:: 2. 启动刚才写好的 GUI
python gui_launcher.py

pause