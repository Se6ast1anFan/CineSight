import torch
import numpy as np
import os, cv2, time, torch, random, pytorchvideo, warnings, argparse, math
from collections import Counter

# --- 语音合成与音频处理 ---
import pyttsx3
from volcenginesdkarkruntime import Ark
import dotenv
from pydub import AudioSegment

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

# --- 1. 加载环境变量 ---
if os.path.exists(".env"):
    dotenv.load_dotenv(".env")
    print("成功加载 .env 配置文件")
else:
    print("Warning: 当前目录下找不到 .env 文件，请检查文件位置！")

# --- 2. 初始化豆包客户端 ---
client = Ark()
model_id = os.getenv("ENDPOINT_ID")

# --- 3. 豆包生成函数 ---
def generate_description(actions_counter, start_sec, end_sec, last_narrative):
    if not actions_counter:
        return ""
    
    # 提取前4个高频标签（动作或物体）
    top_actions = [k for k, v in actions_counter.most_common(4)]
    actions_str = ", ".join(top_actions)
    
    prompt = f"""
    任务：根据检测到的【视觉标签】（包含动作和物体），生成一句极简的中文描述。
    
    【上一段描述】：{last_narrative}
    【当前检测标签】：{actions_str}
    
    【严格约束】：
    1. **拒绝幻觉**：只直译标签。如果标签有 "car"，就说"有车"；如果只有 "walk"，就说"有人在走"。
    2. **去重机制**：如果核心含义与【上一段描述】基本一致（例如车一直在开），输出 "SKIP"。
    3. **字数限制**：严格控制在 **20个汉字以内**。
    4. **纯中文**：将英文标签翻译成中文。
    
    【输出格式】：
    - 有新内容：输出中文描述。
    - 无变化：输出 SKIP。
    """
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个严谨的视觉标签翻译员。"},
                {"role": "user", "content": prompt},
            ],
        )
        description = completion.choices[0].message.content.strip()
        description = description.replace('"', '').replace("'", "")
        
        if "SKIP" in description or description == last_narrative:
            print(f"Time {start_sec}-{end_sec}s: [AI SKIP] (内容重复)")
            return "SKIP"
            
        print(f"Time {start_sec}-{end_sec}s: {description}")
        return description
    except Exception as e:
        print(f"API Error: {e}")
        return "SKIP" 

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
            # 强制缩小到 720P
            height, width = img.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_height = int(height * scale)
                img = cv2.resize(img, (1280, new_height), interpolation=cv2.INTER_AREA)
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

# 辅助函数保持不变
def ava_inference_transform(clip, boxes, num_frames=32, crop_size=640, data_mean=[0.45, 0.45, 0.45], data_std=[0.225, 0.225, 0.225], slow_fast_alpha=4):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)
    clip = normalize(clip, np.array(data_mean, dtype=np.float32), np.array(data_std, dtype=np.float32)) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip, 1, torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    return clip, torch.from_numpy(boxes), roi_boxes

def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:,4:5], pred[:,5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs

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
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    
    cap = MyVideoCapture(config.input)
    print("Processing video frames...")
    
    interval_actions = [] 
    last_segment_narrative = "" 
    
    segment_duration = 5 
    last_summary_time = 0
    fps = 25 
    if hasattr(cap.cap, 'get'):
        f_val = cap.cap.get(cv2.CAP_PROP_FPS)
        if f_val > 0: fps = f_val

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    final_audio_track = AudioSegment.empty()
    
    a = time.time()
    
    while not cap.end:
        ret, img = cap.read()
        if not ret: continue
            
        current_time_sec = int(cap.idx // fps)
        
        # YOLO & DeepSort
        yolo_preds = model([img], size=imsize)
        yolo_preds.files = ["img.jpg"]
        deepsort_outputs = []
        for j in range(len(yolo_preds.pred)):
            temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:,0:4].cpu(), yolo_preds.imgs[j])
            if len(temp) == 0: temp = np.ones((0,8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred = deepsort_outputs
        
        # --- 核心逻辑修改：行为识别 + 物体识别 ---
        if len(cap.stack) == 25:
            clip = cap.get_video_clip()
            if yolo_preds.pred[0].shape[0]:
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:,0:4], crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                
                if isinstance(inputs, list): inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else: inputs = inputs.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                
                # 获取 SlowFast 预测结果
                slowfast_labels = np.argmax(slowfaster_preds, axis=1).tolist()
                
                # --- 修改开始：混合使用 YOLO 和 SlowFast 的结果 ---
                # 遍历每一个检测框
                num_detections = yolo_preds.pred[0].shape[0]
                preds = yolo_preds.pred[0]
                
                for i in range(num_detections):
                    # 获取该框的类别 ID (YOLO 的第4列是 class id)
                    # DeepSort output: [x1, y1, x2, y2, class_id, track_id, vx, vy]
                    cls_id = int(preds[i, 4])
                    
                    if cls_id == 0: 
                        # 如果是人 (Class 0)，记录 SlowFast 的行为标签
                        avalabel = slowfast_labels[i]
                        if avalabel + 1 < len(ava_labelnames):
                            interval_actions.append(ava_labelnames[avalabel+1])
                    else:
                        # 如果不是人 (例如 Car, Dog)，记录 YOLO 的物体名称
                        # model.names 包含所有类别名称 {0: 'person', 1: 'bicycle', 2: 'car'...}
                        if hasattr(model, 'names'):
                            obj_name = model.names[cls_id]
                            interval_actions.append(obj_name)
                # --- 修改结束 ---

        # 分段处理逻辑 (每5秒触发一次)
        if current_time_sec > 0 and current_time_sec % segment_duration == 0 and current_time_sec != last_summary_time:
            action_counts = Counter(interval_actions)
            print(f"\n[DEBUG] {last_summary_time}-{current_time_sec}s 原始检测数据: {dict(action_counts)}")
            
            text_result = generate_description(action_counts, last_summary_time, current_time_sec, last_segment_narrative)
            
            temp_wav = "temp_segment.wav"
            segment_audio = None
            
            if text_result and text_result != "SKIP":
                engine.save_to_file(text_result, temp_wav)
                engine.runAndWait()
                if os.path.exists(temp_wav):
                    segment_audio = AudioSegment.from_wav(temp_wav)
                    last_segment_narrative = text_result
                else:
                    segment_audio = AudioSegment.silent(duration=0)
            else:
                segment_audio = AudioSegment.silent(duration=0)
            
            # 音频同步逻辑
            target_len_ms = segment_duration * 1000
            current_len_ms = len(segment_audio)
            
            if current_len_ms < target_len_ms:
                silence_gap = AudioSegment.silent(duration=target_len_ms - current_len_ms)
                final_segment = segment_audio + silence_gap
            else:
                final_segment = segment_audio[:target_len_ms] # 截断
                print(f"Warning: 语音过长，已截断。")

            final_audio_track += final_segment
            interval_actions = []
            last_summary_time = current_time_sec
            if os.path.exists(temp_wav): os.remove(temp_wav)

    # 处理尾部
    if interval_actions:
        action_counts = Counter(interval_actions)
        print(f"\n[DEBUG] {last_summary_time}s-End 原始检测数据: {dict(action_counts)}")
        text_result = generate_description(action_counts, last_summary_time, int(cap.idx // fps), last_segment_narrative)
        temp_wav = "temp_last.wav"
        if text_result and text_result != "SKIP":
            engine.save_to_file(text_result, temp_wav)
            engine.runAndWait()
            if os.path.exists(temp_wav):
                seg = AudioSegment.from_wav(temp_wav)
                final_audio_track += seg
                os.remove(temp_wav)
    
    print(f"\nProcessing finished. Total time cost: {time.time()-a:.3f} s")
    
    output_path = config.output
    if output_path.endswith(".mp3") or output_path.endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".wav"
    
    print(f"Saving synchronized audio to: {output_path}")
    final_audio_track.export(output_path, format="wav")
    print("Done!")
    
    cap.release()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="demo2.mp4", help='video path')
    parser.add_argument('--output', type=str, default="output.wav", help='output audio path')
    parser.add_argument('--imsize', type=int, default=640, help='inference size')
    parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold')
    parser.add_argument('--device', default='cuda', help='cuda device')
    parser.add_argument('--classes', nargs='+', type=int, help='filter classes')
    parser.add_argument('--show', action='store_true', help='show img')
    config = parser.parse_args()
    
    print(config)
    main(config)