"""
ChatMusician で 1 文キャプションを生成するユーティリティ
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Captioner:
    def __init__(self, model_dir: str, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto"
        ).eval()

    @torch.no_grad()
    def caption(self, score_str: str, max_len: int = 32) -> str:
        """
        楽譜 (文字列) → 1 文キャプション
        """
        prompt = (
            "Describe the emotion of the following lead sheet in one sentence:\n\n"
            f"{score_str}\n\nEmotion:"
        )
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inp, max_length=inp["input_ids"].shape[1] + max_len,
            do_sample=True, temperature=0.8, top_p=0.95
        )
        decoded = self.tok.decode(out[0], skip_special_tokens=True)
        # 最後の行を返す
        return decoded.split("Emotion:")[-1].strip()
