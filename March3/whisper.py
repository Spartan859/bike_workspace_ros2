import mindspore as ms
import time

# ms.set_context(device_target="Ascend")

from mindnlp.transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id = "openai/whisper-small"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, ms_dtype=ms.float16, low_cpu_mem_usage=True, use_safetensors=True,
)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
    ms_dtype=ms.float16,
)

sample = "./sample.wav"

start_time = time.time()

prompt = "以下是普通话的句子。"

result = pipe(sample)

end_time = time.time()
execution_time = end_time - start_time

print(result)
print(f"推理执行时间: {execution_time}秒")