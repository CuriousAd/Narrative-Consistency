from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class NLIPrediction:
    label: str
    p_entail: float
    p_neutral: float
    p_contra: float


class NLIVerifier:
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: str = "cuda",
        max_length: int = 384,
        batch_size: int = 8,
        fp16: bool = True,
    ):
        self.model_name = model_name
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length
        self.batch_size = batch_size
        self.fp16 = fp16 and (self.device == "cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        dtype = torch.float16 if self.fp16 else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # Label mapping from config
        id2label = self.model.config.id2label
        self.label_ids = {}
        for i, lab in id2label.items():
            lab_u = str(lab).upper()
            if "ENTAIL" in lab_u:
                self.label_ids["ENTAILMENT"] = int(i)
            elif "CONTR" in lab_u:
                self.label_ids["CONTRADICTION"] = int(i)
            elif "NEUT" in lab_u:
                self.label_ids["NEUTRAL"] = int(i)

        # Fallback for common MNLI ordering
        if len(self.label_ids) != 3:
            self.label_ids = {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}

    @torch.inference_mode()
    def predict_batch(self, premises: List[str], hypotheses: List[str]) -> List[NLIPrediction]:
        assert len(premises) == len(hypotheses)

        preds: List[NLIPrediction] = []
        c_id = self.label_ids["CONTRADICTION"]
        n_id = self.label_ids["NEUTRAL"]
        e_id = self.label_ids["ENTAILMENT"]

        for i in range(0, len(premises), self.batch_size):
            p_batch = premises[i : i + self.batch_size]
            h_batch = hypotheses[i : i + self.batch_size]

            enc = self.tokenizer(
                p_batch,
                h_batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            if self.device == "cuda" and self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits

            probs = torch.softmax(logits, dim=-1).detach().cpu()

            for row in probs:
                p_contra = float(row[c_id])
                p_neutral = float(row[n_id])
                p_entail = float(row[e_id])

                m = max(p_contra, p_neutral, p_entail)
                if m == p_contra:
                    lab = "CONTRADICTION"
                elif m == p_entail:
                    lab = "ENTAILMENT"
                else:
                    lab = "NEUTRAL"

                preds.append(NLIPrediction(lab, p_entail, p_neutral, p_contra))

        return preds
