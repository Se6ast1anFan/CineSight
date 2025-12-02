import torch
import numpy as np
import os, cv2, time, torch, random, pytorchvideo, warnings, argparse, math
from collections import Counter # 用于统计最高频的动作

# --- 新增：豆包 API 和 TTS 库 ---
import pyttsx3
from volcenginesdkarkruntime import Ark
import dotenv

warnings.filterwarnings("ignore", category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort

# --- 新增：加载环境变量 ---
# 请确保 .env 文件路径正确，这里假设它在上一级目录的 doubao 文件夹中，或者你可以改成绝对路径
# 建议：将 .env 复制到 yolo_slowfast 文件夹下，然后写 load_dotenv(".env")
if os.path.exists(".env"):
    dotenv.load_dotenv(".env")
    print("成功加载 .env 配置文件")
else:
    print("Warning: 当前目录下找不到 .env 文件，请检查文件位置！")

# --- 新增：初始化豆包客户端 ---
client = Ark()
model_id = os.getenv("ENDPOINT_ID")

# --- 新增：豆包描述生成函数 ---
def generate_description(actions_counter, start_sec, end_sec):
    """
    输入：一段时间内的动作统计（例如 {'walk': 10, 'stand': 2}）
    输出：豆包生成的自然语言描述
    """
    if not actions_counter:
        return ""
    
    # 将统计字典转换为字符串提示
    actions_str = ", ".join([f"{k}" for k, v in actions_counter.most_common(3)]) # 只取前3个高频动作
    
    prompt = f"在视频的第{start_sec}秒到第{end_sec}秒，检测到的主要动作标签有：{actions_str}。请用一句简短、流畅的中文描述这段画面中发生的事情（例如：'画面中有人正在行走'）。不要包含任何Markdown格式或额外废话，直接输出描述。"
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个视频内容解说员。"},
                {"role": "user", "content": prompt},
            ],
        )
        description = completion.choices[0].message.content
        print(f"Time {start_sec}-{end_sec}s: {description}")
        return description
    except Exception as e:
        print(f"API Error: {e}")
        return ""

class MyVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        
    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img
    
    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)
        
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip
    
    def release(self):
        self.cap.release()
        
def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, 
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

# --- 修改：删除了 save_yolopreds_tovideo，因为我们不需要输出视频了 ---

def main(config):
    device = config.device
    imsize = config.imsize
    
    print("Loading models...")
    model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5l6').to(device)
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 100
    if config.classes:
        model.classes = config.classes
    
    video_model = slowfast_r50_detection(True).eval().to(device)
    
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    
    # --- 修改：不再创建 cv2.VideoWriter ---
    # outputvideo = cv2.VideoWriter(...) 
    
    cap = MyVideoCapture(config.input)
    
    print("Processing video frames...")
    
    # --- 新增变量：用于存储一段时间内的动作 ---
    interval_actions = [] 
    full_narrative_text = "" # 最终的总文本
    segment_duration = 5 # 每 5 秒总结一次
    last_summary_time = 0
    fps = 25 # 假设视频帧率为 25，如果不是请读取 video.get(cv2.CAP_PROP_FPS)

    a = time.time()
    
    while not cap.end:
        ret, img = cap.read()
        if not ret:
            continue
            
        current_time_sec = cap.idx // fps
        
        yolo_preds = model([img], size=imsize)
        yolo_preds.files = ["img.jpg"]
        
        deepsort_outputs = []
        for j in range(len(yolo_preds.pred)):
            temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:,0:4].cpu(), yolo_preds.imgs[j])
            if len(temp) == 0:
                temp = np.ones((0,8))
            deepsort_outputs.append(temp.astype(np.float32))
            
        yolo_preds.pred = deepsort_outputs
        
        # 每处理 1 秒的片段 (25帧) 进行一次 SlowFast 识别
        if len(cap.stack) == 25:
            # print(f"Processing {current_time_sec}th second clips")
            clip = cap.get_video_clip()
            if yolo_preds.pred[0].shape[0]:
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:,0:4], crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                
                # 获取当前的标签
                for tid, avalabel in zip(yolo_preds.pred[0][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
                    label_name = ava_labelnames[avalabel+1]
                    interval_actions.append(label_name)

            # --- 修改：每隔 segment_duration (5秒) 调用一次豆包 ---
            if current_time_sec > 0 and current_time_sec % segment_duration == 0 and current_time_sec != last_summary_time:
                # 统计这5秒内最常见的动作
                action_counts = Counter(interval_actions)
                # 调用豆包生成描述
                description = generate_description(action_counts, last_summary_time, current_time_sec)
                full_narrative_text += description + " "
                
                # 重置缓存
                interval_actions = []
                last_summary_time = current_time_sec

    # --- 处理剩余的尾部片段 ---
    if interval_actions:
        action_counts = Counter(interval_actions)
        description = generate_description(action_counts, last_summary_time, cap.idx // fps)
        full_narrative_text += description
        
    print(f"\nProcessing finished. Total time cost: {time.time()-a:.3f} s")
    print("-" * 30)
    print("Final Video Description Script:")
    print(full_narrative_text)
    print("-" * 30)

    # --- 新增：使用 TTS 生成音频文件 ---
    if full_narrative_text.strip():
        print("Generating audio file...")
        engine = pyttsx3.init()
        # 设置语速
        engine.setProperty('rate', 150)
        
        # 你的输出文件名
        audio_output_path = config.output.replace(".mp4", ".mp3")
        if not audio_output_path.endswith('.mp3'):
             audio_output_path = "output_audio.mp3"

        # 保存音频
        engine.save_to_file(full_narrative_text, audio_output_path)
        engine.runAndWait()
        print(f"Audio saved to: {audio_output_path}")
    else:
        print("No actions detected, no audio generated.")
    
    cap.release()
    # outputvideo.release() # 已移除

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/wufan/images/video/vad.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="output.mp4", help='will be used to name the mp3 file')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--show', action='store_true', help='show img')
    config = parser.parse_args()
    
    if config.input.isdigit():
        print("using local camera.")
        config.input = int(config.input)
        
    print(config)
    main(config)