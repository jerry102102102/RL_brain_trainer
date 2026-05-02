"""Text-only Qwen2.5-VL runner used by the V5 L1 client.

The main Qwen workspace originally provided an image+prompt script.  This small
runner keeps the same model/runtime but allows the L1 bridge to ask Qwen for a
structured tool-call JSON object from text and structured scene context.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL in text-only mode")
    parser.add_argument("--prompt", required=True, help="Text prompt to send to Qwen")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
