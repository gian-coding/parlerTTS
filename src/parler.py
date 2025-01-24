import os
import base64
import soundfile as sf
import logging
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


class ParlerTTS:
    def __init__(self, input_payload, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1")
        self.default_description = "Richard's voice is monotone, yet slightly fast in delivery, with a very close recording that almost has no background noise."
        self.prompt = input_payload.text
        self.language = input_payload.language
        self.description = input_payload.description

    def execute(self):
        try:
            # Preprocess and generate audio
            logging.info("Inizio generazione audio")
            audio = self.generate_audio(self.prompt, self.description)
            json_output = {"output": ""}
            self.postprocessing(audio, json_output)
            #output_path = self.save_output(json_output, "../../tts_outputs/output.json")
            logging.info(f"Elaborazione completata.")
            return json_output
        except Exception as e:
            logging.error(f"Errore durante l'elaborazione: {e}", exc_info=True)
            raise

    def generate_audio(self, prompt, description):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt",return_attention_mask=True).input_ids.to(self.device)
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        if len(audio_arr) == 0:
            raise ValueError("Audio vuoto generato dal modello")
        return audio_arr

    def postprocessing(self, audio, json_output):
        temp_wav_path = "temp_output.wav"
        sf.write(temp_wav_path, audio, self.model.config.sampling_rate)
        with open(temp_wav_path, "rb") as wav_file:
            encoded_audio = base64.b64encode(wav_file.read()).decode('utf-8')
        os.remove(temp_wav_path)
        json_output["output"] = encoded_audio

'''    def save_output(self, output_json, output_path):
        try:
            if not isinstance(output_json, dict):
                raise ValueError("output_json deve essere un dizionario compatibile con JSON")
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=4)'''

'''            logging.info(f"File salvato con successo in: {output_path}")
        except Exception as e:
            logging.error(f"Errore durante il salvataggio del file: {e}", exc_info=True)
            raise
        return output_path'''
