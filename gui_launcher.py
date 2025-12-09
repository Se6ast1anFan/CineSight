import customtkinter as ctk
import os
import subprocess
import threading
import sys
from tkinter import filedialog
import platform

# --- 配置界面外观 ---
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 1. 微调：窗口标题 ---
        self.title("[心视CineSight]视障辅助解说器")
        self.geometry("900x700")
        
        # 布局网格权重
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) 

        # --- 顶部区域 ---
        self.top_frame = ctk.CTkFrame(self, corner_radius=10)
        self.top_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        # --- 2. 微调：标签文本 ---
        self.lbl_title = ctk.CTkLabel(self.top_frame, text="请先输入视频文件", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=(15, 5))
        
        self.btn_select = ctk.CTkButton(self.top_frame, text="点击选择 MP4 文件", height=50, width=300, 
                                        font=ctk.CTkFont(size=16), command=self.select_file)
        self.btn_select.pack(pady=15)

        self.lbl_file_path = ctk.CTkLabel(self.top_frame, text="未选择文件", text_color="gray")
        self.lbl_file_path.pack(pady=(0, 15))

        # --- 中部日志区 ---
        self.log_textbox = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=12), activate_scrollbars=True)
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.log_textbox.insert("0.0", "--- 系统就绪 ---\n")
        self.log_textbox.configure(state="disabled") 

        # --- 底部按钮区 ---
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
        self.work_dir = os.path.dirname(os.path.abspath(__file__)) 

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")])
        if file_path:
            # 获取绝对路径，防止路径错乱
            self.selected_file_path = os.path.abspath(file_path)
            
            # --- 3. 微调：文件名处理逻辑 ---
            # 逻辑：输入 demo.mp4 -> 输出 demo_out.wav
            dir_name = os.path.dirname(self.selected_file_path)
            base_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
            
            # 组合输出路径
            self.output_wav_path = os.path.join(dir_name, f"{base_name}_out.wav")
            
            self.lbl_file_path.configure(text=f"已选: {os.path.basename(file_path)}", text_color="#4CC2FF")
            self.btn_run.configure(state="normal")
            
            self.log_message("-" * 30)
            self.log_message(f"输入视频: {self.selected_file_path}")
            self.log_message(f"预定输出: {self.output_wav_path}")

    def log_message(self, message):
        """向文本框追加日志，并滚动到底部"""
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def start_process_thread(self):
        self.btn_run.configure(state="disabled", text="正在处理中...")
        self.btn_select.configure(state="disabled")
        self.btn_open_folder.configure(state="disabled") # 运行时禁用打开文件夹
        threading.Thread(target=self.run_yolo_process, daemon=True).start()

    def run_yolo_process(self):
        try:
            # 设置环境变量
            current_env = os.environ.copy()
            current_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            # 构建命令
            command = [
                sys.executable,
                "-u",  
                "yolo_slowfast.py",
                "--input", self.selected_file_path,
                "--output", self.output_wav_path, # 使用我们在 select_file 中计算好的路径
                "--device", "cuda"
            ]

            self.log_message(f"正在启动处理引擎...")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                cwd=self.work_dir,
                env=current_env,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                self.log_message(line.strip())

            process.wait()

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
        self.after(0, lambda: self._update_ui_after_finish(success))

    def _update_ui_after_finish(self, success):
        self.btn_run.configure(text="重新开始生成", state="normal")
        self.btn_select.configure(state="normal")
        if success:
            self.btn_open_folder.configure(state="normal", fg_color="#1f538d")
            # 可以在这里加一个弹窗提示
            # from tkinter import messagebox
            # messagebox.showinfo("完成", "音频生成已完成！")

    def open_output_folder(self):
        """修复后的打开文件夹逻辑"""
        if not self.output_wav_path:
            return
        
        # 1. 确保路径是绝对路径
        abs_path = os.path.abspath(self.output_wav_path)
        
        # 2. 核心修复：将路径标准化为 Windows 格式 (使用反斜杠 \)
        # 如果路径里有 '/'，Windows Explorer 的 /select 参数有时会失效并跳转到桌面
        abs_path = os.path.normpath(abs_path)
        
        if not os.path.exists(abs_path):
            self.log_message(f"错误：找不到文件 {abs_path}")
            return

        # 3. 根据系统调用
        if platform.system() == "Windows":
            # 注意：/select, 后面紧跟路径，不要加空格，且路径最好用引号包起来（虽然 subprocess 会处理，但 normpath 很关键）
            subprocess.run(['explorer', '/select,', abs_path])
        elif platform.system() == "Darwin": # macOS
            subprocess.run(["open", "-R", abs_path])
        else: # Linux
            folder_path = os.path.dirname(abs_path)
            subprocess.run(["xdg-open", folder_path])

if __name__ == "__main__":
    app = App()
    app.mainloop()