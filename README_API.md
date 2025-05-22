# CosyVoice API 使用文档

## 简介

CosyVoice API 是基于 CosyVoice 语音合成模型的 RESTful API 服务，支持多种语音合成模式，包括预训练音色、3s极速复刻、跨语种复刻和自然语言控制。

## 安装依赖

确保已安装所有必要的依赖：

```bash
pip install fastapi uvicorn pydantic numpy torch torchaudio librosa
```

## 启动服务

```bash
python api.py --port 8000 --model_dir pretrained_models/CosyVoice2-0.5B
```

参数说明：
- `--port`：服务端口号，默认为 8000
- `--model_dir`：模型目录，可以是本地路径或 modelscope 仓库 ID

## API 端点

### 1. 获取可用音色列表

```
GET /available_spks
```

**响应示例：**

```json
{
  "speakers": [
    {
      "spk_id": "zhongwen_nv",
      "spk_name": "中文女"
    },
    {
      "spk_id": "your_custom_speaker_id",
      "spk_name": "我的自定义音色"
    }
  ]
}
```

### 2. 预训练音色合成

```
POST /tts/sft
```

**请求参数：**

```json
{
  "tts_text": "需要合成的文本",
  "spk_id": "中文女",
  "stream": false,
  "speed": 1.0,
  "seed": 0
}
```

- `tts_text`：需要合成的文本
- `spk_id`：音色 ID，可通过 `/available_spks` 获取（返回结果中的 `spk_id` 字段）
- `stream`：是否使用流式推理，默认为 false
- `speed`：语速调节，范围 0.5-2.0，默认为 1.0
- `seed`：随机种子，0 表示随机生成

**响应：**

返回 WAV 格式的音频流

### 3. 3s极速复刻

```
POST /tts/zero_shot
```

**请求参数：**

```json
{
  "tts_text": "需要合成的文本",
  "prompt_text": "提示文本，需与提示音频内容一致",
  "prompt_audio_base64": "BASE64编码的WAV音频",
  "stream": false,
  "speed": 1.0,
  "seed": 0
}
```

- `tts_text`：需要合成的文本
- `prompt_text`：提示文本，需与提示音频内容一致
- `prompt_audio_base64`：BASE64 编码的 WAV 音频
- `stream`：是否使用流式推理，默认为 false
- `speed`：语速调节，范围 0.5-2.0，默认为 1.0
- `seed`：随机种子，0 表示随机生成

**响应：**

返回 WAV 格式的音频流

### 4. 跨语种复刻

```
POST /tts/cross_lingual
```

**请求参数：**

```json
{
  "tts_text": "需要合成的文本",
  "prompt_audio_base64": "BASE64编码的WAV音频",
  "stream": false,
  "speed": 1.0,
  "seed": 0
}
```

- `tts_text`：需要合成的文本
- `prompt_audio_base64`：BASE64 编码的 WAV 音频
- `stream`：是否使用流式推理，默认为 false
- `speed`：语速调节，范围 0.5-2.0，默认为 1.0
- `seed`：随机种子，0 表示随机生成

**响应：**

返回 WAV 格式的音频流

### 5. 自然语言控制

```
POST /tts/instruct
```

**请求参数：**

```json
{
  "tts_text": "需要合成的文本",
  "spk_id": "中文女",
  "instruct_text": "控制文本，如：激动地说",
  "stream": false,
  "speed": 1.0,
  "seed": 0
}
```

- `tts_text`：需要合成的文本
- `spk_id`：预训练音色 ID
- `instruct_text`：控制文本，描述语音风格
- `stream`：是否使用流式推理，默认为 false
- `speed`：语速调节，范围 0.5-2.0，默认为 1.0
- `seed`：随机种子，0 表示随机生成

**响应：**

返回 WAV 格式的音频流

### 6. 上传音频文件

```
POST /upload_audio
```

**请求参数：**

使用 `multipart/form-data` 格式上传文件，参数名为 `file`

**响应示例：**

```json
{
  "filename": "uploaded_audio.wav",
  "audio_base64": "BASE64编码的音频内容"
}
```

### 7. 保存音色

```
POST /speaker_prompt/save
```

**请求参数：**

使用 `multipart/form-data` 格式上传文件和表单数据：
- `prompt_audio`： (file) WAV格式的音频文件，用于提取音色。
- `prompt_text`： (form data) 提示文本，需与提示音频内容一致。
- `spk_id`： (form data) 为该音色指定的唯一标识符。
- `spk_name`： (form data, 可选) 为该音色指定的用户友好名称。如果未提供，将使用 `spk_id` 作为名称。

**请求示例 (cURL):**

```bash
curl -X POST "http://localhost:8000/speaker_prompt/save" \
     -F "prompt_audio=@/path/to/your/audio.wav" \
     -F "prompt_text=这是提示文本" \
     -F "spk_id=your_custom_speaker_id" \
     -F "spk_name=我的自定义音色"
```

- `prompt_audio`：WAV格式的音频文件，用于提取音色。
- `prompt_text`：提示文本，需与提示音频内容一致。
- `spk_id`：为该音色指定的唯一标识符。
- `spk_name`：(可选) 为该音色指定的用户友好名称。

**响应示例：**

```json
{
  "status": "success",
  "message": "音色 '我的自定义音色' (ID: your_custom_speaker_id) 保存成功",
  "spk_id": "your_custom_speaker_id",
  "spk_name": "我的自定义音色"
}
```

## 使用示例

### Python 示例

```python
import requests
import base64
import json

# 服务地址
API_URL = "http://localhost:8000"

# 获取可用音色列表
def get_available_spks():
    response = requests.get(f"{API_URL}/available_spks")
    # 返回的数据结构现在是 {'speakers': [{'spk_id': 'id', 'spk_name': 'name'}, ...]}
    return response.json()

# 上传音频文件
def upload_audio(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/upload_audio", files=files)
    return response.json()

# 预训练音色合成
def tts_sft(text, spk_id, stream=False, speed=1.0, seed=0):
    data = {
        "tts_text": text,
        "spk_id": spk_id,
        "stream": stream,
        "speed": speed,
        "seed": seed
    }
    response = requests.post(f"{API_URL}/tts/sft", json=data)
    
    # 保存音频文件
    with open("output.wav", "wb") as f:
        f.write(response.content)
    
    return "output.wav"

# 3s极速复刻
def tts_zero_shot(text, prompt_text, prompt_audio_base64, stream=False, speed=1.0, seed=0):
    data = {
        "tts_text": text,
        "prompt_text": prompt_text,
        "prompt_audio_base64": prompt_audio_base64,
        "stream": stream,
        "speed": speed,
        "seed": seed
    }
    response = requests.post(f"{API_URL}/tts/zero_shot", json=data)
    
    # 保存音频文件
    with open("output.wav", "wb") as f:
        f.write(response.content)
    
    return "output.wav"

# 使用示例
if __name__ == "__main__":
    # 获取可用音色
    spks_data = get_available_spks()
    if spks_data and 'speakers' in spks_data and len(spks_data['speakers']) > 0:
        print(f"可用音色详情: {spks_data['speakers']}")
        first_speaker_id = spks_data['speakers'][0]['spk_id']
        print(f"将使用第一个音色进行SFT合成: ID='{first_speaker_id}', 名称='{spks_data['speakers'][0]['spk_name']}'")
        
        # 使用预训练音色合成
        output_file = tts_sft(
            "这是一段使用预训练音色合成的语音。", 
            first_speaker_id
        )
        print(f"音频已保存至: {output_file}")
    else:
        print("未能获取到可用音色列表，或列表为空。")
    
    # 上传音频文件并使用3s极速复刻
    upload_result = upload_audio("prompt.wav")
    output_file = tts_zero_shot(
        "这是一段使用3s极速复刻合成的语音。",
        "这是提示文本。",
        upload_result["audio_base64"]
    )
    print(f"音频已保存至: {output_file}")
```

## 注意事项

1. 不同的合成模式需要使用对应的模型，请确保使用正确的模型：
   - 自然语言控制模式需要使用 CosyVoice-300M-Instruct 模型
   - 跨语种复刻模式需要使用 CosyVoice-300M 模型

2. 音频采样率要求：
   - prompt 音频采样率不得低于 16kHz

3. 流式推理与速度调节：
   - 速度调节仅在非流式推理模式下有效