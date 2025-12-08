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
def generate_description(interval_data, start_sec, end_sec, last_narrative):
    if not interval_data:
        return ""
    
    # --- 步骤 A: 数据预处理 ---
    
    # 1. 宏观统计 (物体/动作统计)
    all_labels = [item['val'] for item in interval_data]
    macro_counts = Counter(all_labels)
    macro_str = ", ".join([f"{k}({v})" for k, v in macro_counts.most_common(6)])
    
    # 2. 微观追踪 (ID 统计)
    person_actions = {} 
    
    for item in interval_data:
        if item['type'] == 'person':
            tid = item['id']
            if tid not in person_actions:
                person_actions[tid] = []
            person_actions[tid].append(item['val'])
            
    # 人数硬逻辑拦截
    unique_ids = list(person_actions.keys())
    if len(unique_ids) > 3:
        micro_str = f"检测到 {len(unique_ids)} 个不同人物(人群/Crowd)"
    elif len(unique_ids) == 0:
        micro_str = "无特定人物"
    else:
        track_summary = []
        for tid in unique_ids:
            acts = person_actions[tid]
            if not acts: continue
            most_common_act = Counter(acts).most_common(1)[0][0]
            # Prompt 里也带上 ID，方便豆包区分
            track_summary.append(f"Person_{tid}({most_common_act})")
        micro_str = ", ".join(track_summary)

    # --- 步骤 B: 构建 Prompt ---
    prompt = f"""
    【角色设定】
    你是指向视障人士的专业影视音频解说员，遵循 Netflix 标准。
    
    【输入数据】：
    1. **物体/动作统计**: {macro_str}
    2. **人物追踪信息**: {micro_str}
    
    【上一时段解说】：{last_narrative}

    【解说生成规则 - 优先级由高到低】：
    1. **去噪逻辑**：
       - 如果物体(car)频次极高(>20)而动作(run)频次极低(<5)，忽略动作。
    2. **共存逻辑 (核心)**：
       - **如果 人 和 车(或其他大物体) 同时高频出现，必须同时提及。**
       - 示例：输入 "Person_1(stand), car(15)" -> 输出 "一人站在车旁"。
    3. **数量归纳**：
       - 如果【人物追踪信息】显示“人群”或“>3个不同人物”，直接说“多人[动作]”，不要试图数数。
       - 如果是具体ID（如 Person_1, Person_2），则精确描述人数（如“两人奔跑”）。
    4. **去重静默**：
       - 如果画面含义与【上一时段】基本一致，输出 "SKIP"。

    【输出格式】：
    - 15字以内。
    - 纯中文，无标点符号。
    - 若无新信息输出 SKIP。
    """
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个逻辑严密的盲人辅助解说员。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1, 
        )
        description = completion.choices[0].message.content.strip()
        description = description.replace('"', '').replace("'", "").replace("\n", "")
        
        if "SKIP" in description or description == last_narrative:
            print(f"Time {start_sec}-{end_sec}s: [AI SKIP] (Silence)")
            return "SKIP"
        
        if len(description) > 25:
             print(f"Time {start_sec}-{end_sec}s: [Ignored] Text too long.")
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

# 辅助函数
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
    
    interval_data = [] 
    last_segment_narrative = "" 
    
    segment_duration = 5 
    last_summary_time = 0
    fps = 25 
    if hasattr(cap.cap, 'get'):
        f_val = cap.cap.get(cv2.CAP_PROP_FPS)
        if f_val > 0: fps = f_val

    engine = pyttsx3.init()
    engine.setProperty('rate', 180) 
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
        
        # SlowFast Inference
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
                
                slowfast_labels = np.argmax(slowfaster_preds, axis=1).tolist()
                
                num_detections = yolo_preds.pred[0].shape[0]
                preds = yolo_preds.pred[0]
                
                for i in range(num_detections):
                    cls_id = int(preds[i, 4])
                    track_id = int(preds[i, 5])
                    
                    if cls_id == 0: 
                        # 人：记录 ID
                        avalabel = slowfast_labels[i]
                        if avalabel + 1 < len(ava_labelnames):
                            label = ava_labelnames[avalabel+1]
                            interval_data.append({'id': track_id, 'type': 'person', 'val': label})
                    else:
                        # 物体
                        if hasattr(model, 'names'):
                            obj_name = model.names[cls_id]
                            interval_data.append({'id': track_id, 'type': 'object', 'val': obj_name})

        # 分段处理
        if current_time_sec > 0 and current_time_sec % segment_duration == 0 and current_time_sec != last_summary_time:
            
            # --- 修复：[DEBUG] 显示 ID ---
            # 如果是人，显示 ID:val，否则显示 val
            debug_list = []
            for item in interval_data:
                if item['type'] == 'person':
                    debug_list.append(f"ID{item['id']}:{item['val']}")
                else:
                    debug_list.append(item['val'])
            
            debug_counts = Counter(debug_list)
            print(f"\n[DEBUG] {last_summary_time}-{current_time_sec}s 原始标签: {dict(debug_counts)}")
            # ---------------------------

            text_result = generate_description(interval_data, last_summary_time, current_time_sec, last_segment_narrative)
            
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
            
            target_len_ms = segment_duration * 1000
            current_len_ms = len(segment_audio)
            
            if current_len_ms < target_len_ms:
                silence_gap = AudioSegment.silent(duration=target_len_ms - current_len_ms)
                final_segment = segment_audio + silence_gap
            else:
                final_segment = segment_audio[:target_len_ms] 
                print(f"Warning: 语音过长，已截断。")

            final_audio_track += final_segment
            interval_data = [] 
            last_summary_time = current_time_sec
            if os.path.exists(temp_wav): os.remove(temp_wav)

    # 处理尾部
    if interval_data:
        # 尾部 Debug
        debug_list = [f"ID{x['id']}:{x['val']}" if x['type'] == 'person' else x['val'] for x in interval_data]
        debug_counts = Counter(debug_list)
        print(f"\n[DEBUG] {last_summary_time}-End 原始标签: {dict(debug_counts)}")
        
        text_result = generate_description(interval_data, last_summary_time, int(cap.idx // fps), last_segment_narrative)
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