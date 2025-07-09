import math
from typing import Any, Union

import numpy as np
import torch
import torch.nn.attention
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline,
)

from ModelInterfaces import IASRModel


class WhisperASRModel(IASRModel):
    def __init__(self, language:str, model_name="openai/whisper-base"):
        # self.asr = pipeline("automatic-speech-recognition", model=model_name, return_timestamps="word",model_kwargs=dict(language="de"))
        
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000

        

        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task="transcribe")

        self._forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            return_timestamps="word"
            # chunk_length_s=30,
            # stride_length_s=(4, 2)
        )

    def processAudio(self, audio:Union[np.ndarray, torch.Tensor]):
        # 'audio' can be a path to a file or a numpy array of audio samples.
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        result : Any = self.asr(audio[0],generate_kwargs={"forced_decoder_ids": self._forced_decoder_ids})
        self._transcript = result["text"]

        l=[]
        for word_info in result['chunks']:
            ts,te=word_info["timestamp"]
            if ts==te:
                continue
            if te is None:
                te=math.inf
            d={"word":word_info["text"], "start_ts":ts*self.sample_rate,
                                 "end_ts":te*self.sample_rate}
            l.append(d)

        self._word_locations = l

    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        
        return self._word_locations