# Chart Captioning VLM

**Ultra-lightweight fine-tuned BLIP model for generating accurate chart captions**

![Model Architecture](docs/architecture.png)  

Visually impaired and blind individuals increasingly rely on image captions and alt text to interpret visual content online. Recent advances in vision–language models (VLMs) offer promising tools for automatic caption generation, but generic, pre-trained models often produce flawed or misleading descriptions—especially for data-heavy visuals like charts. These inaccuracies aren’t just cosmetic; they distort meaning and undermine accessibility. Moreover, state-of-the-art captioning models tend to be large and computationally intensive, limiting their practicality for small-scale or edge use.

This project explores a minimalist fine-tuning strategy for adapting a pre-trained VLM built on top of Salesforce’s [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) to generate more accurate, contextually relevant chart captions—using only a small dataset and modest consumer hardware. By aggregating (query, answer) annotations into a single dense caption per chart, we supervise the model to emphasize salient relationships and trends. The result is a compact model that delivers noticeably improved captions with just a few training epochs on a laptop, demonstrating how even lightweight tuning can meaningfully enhance accessibility in real-world settings.

---


## Features

-  Fine-tunes BLIP for chart-specific language
-  Fast: trains in ~3 epochs with small batches
-  Inference-ready caption generator
-  Exportable & reloadable via Hugging Face API

## Setup

```bash
git clone https://github.com/your-org/chart-captioning-vlm.git
cd chart-captioning-vlm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Fine-tune BLIP using Hugging Face’s Trainer:

```bash
python train.py \\
  --train_data data/raw/train \\
  --val_data data/raw/val \\
  --output_dir ./chart_caption_model \\
  --epochs 3 \\
  --batch_size 8 \\
  --fp16
```

Captions are formed by concatenating all Q&A pairs for each chart.

## Inference

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("./chart_caption_model")
model = BlipForConditionalGeneration.from_pretrained("./chart_caption_model").to(device)

def make_caption(path, prompt="", max_new_tokens=64):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True)
```

## Output Example

```text
Input: /charts/line-chart.png
Output: "The line chart shows a steady increase in sales from Jan to July."
