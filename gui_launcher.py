import customtkinter as ctk
import os
import subprocess
import threading
import sys
from tkinter import filedialog
import platform

# --- 配置界面外观 ---
ctk.set_appearance_mode("Dark")  # 模式: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # 主题: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 窗口基础设置
        self.title("[心视CineSight]视障辅助解说器")
        self.geometry("900x700")
        
        # 布局网格权重
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # 让日志框自动伸缩

        # --- 1. 顶部：文件选择区 ---
        self.top_frame = ctk.CTkFrame(self, corner_radius=10)
        self.top_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.lbl_title = ctk.CTkLabel(self.top_frame, text="请先输入视频文件", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=(15, 5))
        
        self.btn_select = ctk.CTkButton(self.top_frame, text="点击选择 MP4 文件", height=50, width=300, 
                                        font=ctk.CTkFont(size=16), command=self.select_file)
        self.btn_select.pack(pady=15)

        self.lbl_file_path = ctk.CTkLabel(self.top_frame, text="未选择文件", text_color="gray")
        self.lbl_file_path.pack(pady=(0, 15))

        # --- 2. 中部：日志输出区 ---
        self.log_textbox = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=12), activate_scrollbars=True)
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.log_textbox.insert("0.0", "--- 等待任务开始 ---\n系统就绪。\n")
        self.log_textbox.configure(state="disabled") # 初始设为只读

        # --- 3. 底部：操作区 ---
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        
        self.btn_run = ctk.CTkButton(self.bottom_frame, text="开始生成解说音频", height=45, fg_color="green", 
                                     hover_color="darkgreen", font=ctk.CTkFont(size=16, weight="bold"), 
                                     state="disabled", command=self.start_process_thread)
        self.btn_run.pack(side="left", expand=True, fill="x", padx=(0, 10))

        self.btn_open_folder = ctk.CTkButton(self.bottom_frame, text="打开输出文件夹", height=45, 
                                             font=ctk.CTkFont(size=16), state="disabled", command=self.open_output_folder)
        self.btn_open_folder.pack(side="right", expand=True, fill="x", padx=(10, 0))

        # 变量存储
        self.selected_file_path = ""
        self.output_wav_path = ""
        self.work_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本所在目录

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")])
        if file_path:
            self.selected_file_path = file_path
            self.lbl_file_path.configure(text=f"已选: {os.path.basename(file_path)}", text_color="#4CC2FF")
            self.btn_run.configure(state="normal")
            self.log_message(f"已加载文件: {file_path}")

    def log_message(self, message):
        """向文本框追加日志，并滚动到底部"""
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def start_process_thread(self):
        """使用线程运行，防止界面卡死"""
        self.btn_run.configure(state="disabled", text="正在处理中...")
        self.btn_select.configure(state="disabled")
        
        # 开启新线程运行任务
        threading.Thread(target=self.run_yolo_process, daemon=True).start()

    def run_yolo_process(self):
        try:
            # 1. 解析文件名构建输出路径
            # 输入: C:/Users/xxx/demo.mp4
            # 输出: C:/Users/xxx/demo_out.wav
            dir_name = os.path.dirname(self.selected_file_path)
            base_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
            self.output_wav_path = os.path.join(dir_name, f"{base_name}_out.wav")

            # 2. 准备环境变量 (相当于 set KMP_DUPLICATE_LIB_OK=TRUE)
            current_env = os.environ.copy()
            current_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            # 3. 构建命令 (相当于 python yolo_slowfast.py ...)
            # 注意：我们直接调用 python，因为假设用户是在conda环境下启动的这个GUI
            command = [
                sys.executable,  # 获取当前运行此GUI的python解释器路径
                "yolo_slowfast.py",
                "--input", self.selected_file_path,
                "--output", self.output_wav_path,
                "--device", "cuda"
            ]

            self.log_message("-" * 40)
            self.log_message(f"执行命令: {' '.join(command)}")
            self.log_message("-" * 40)

            # 4. 执行子进程并实时获取输出
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # 将错误也重定向到标准输出
                text=True,
                cwd=self.work_dir, # 确保在当前目录下运行
                env=current_env,
                bufsize=1,
                universal_newlines=True
            )

            # 实时逐行读取输出
            for line in process.stdout:
                self.log_message(line.strip())

            process.wait() # 等待结束

            if process.returncode == 0:
                self.log_message("\n>>> 处理成功完成！ <<<")
                self.on_process_finish(success=True)
            else:
                self.log_message(f"\n>>> 错误：进程退出，代码 {process.returncode} <<<")
                self.on_process_finish(success=False)

        except Exception as e:
            self.log_message(f"\n>>> 发生严重错误: {str(e)} <<<")
            self.on_process_finish(success=False)

    def on_process_finish(self, success):
        """任务结束后的回调，需在主线程更新UI"""
        # 使用 after 方法确保在主线程更新 UI
        self.after(0, lambda: self._update_ui_after_finish(success))

    def _update_ui_after_finish(self, success):
        self.btn_run.configure(text="重新开始生成", state="normal")
        self.btn_select.configure(state="normal")
        if success:
            self.btn_open_folder.configure(state="normal", fg_color="#1f538d")
            # 弹窗提示
            # ctk.CTkMessagebox(title="Success", message="音频生成完毕！") # 需要额外库，这里省略

    def open_output_folder(self):
        """打开输出文件所在的文件夹，并选中文件"""
        if not self.output_wav_path or not os.path.exists(self.output_wav_path):
            return
        
        folder_path = os.path.dirname(self.output_wav_path)
        
        # 根据系统自动选择打开方式
        if platform.system() == "Windows":
            subprocess.run(["explorer", "/select,", self.output_wav_path])
        elif platform.system() == "Darwin": # macOS
            subprocess.run(["open", "-R", self.output_wav_path])
        else: # Linux
            subprocess.run(["xdg-open", folder_path])

if __name__ == "__main__":
    app = App()
    app.mainloop()