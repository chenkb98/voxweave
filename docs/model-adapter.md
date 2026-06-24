# 可选 Qwen2-Audio 适配器

`voxweave.adapters.qwen2_audio_response` 接受调用者提供的 Processor
和已设为 eval 模式的 Qwen2AudioForConditionalGeneration 模型。
它把本地消息转换为聊天模板，并传入已解码的音频数组；embedded:// 标记
只用于模板，不会发起下载请求。

```python
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from voxweave.adapters import qwen2_audio_response
from voxweave.backend import model_request
from voxweave.conversation import audio_message

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
).eval()
request = model_request([audio_message(samples, processor.feature_extractor.sampling_rate)])
response = qwen2_audio_response(request, processor, model)
```

`samples` 由调用者提供。音频必须预先显式重采样到 Processor 要求的采样率。
适配器只支持 greedy decoding，temperature 必须为零；该模式不使用随机种子。
处理器调用采用当前官方文档的 `audio=` 参数。需自行安装 torch/transformers
及适当设备依赖，并遵守模型许可。

默认测试只使用严格协议替身，未下载或执行真实预训练权重。
[官方接口文档](https://huggingface.co/docs/transformers/model_doc/qwen2_audio)
