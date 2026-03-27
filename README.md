# Inference-Time Structural Reasoning for Compositional Vision-Language Understanding

A comprehensive research framework for evaluating and analyzing vision-language models (VLMs) on compositional reasoning tasks, with focus on the **Winoground benchmark**.

## Overview

This project evaluates multiple state-of-the-art vision-language models:
- **CLIP** (CLIP model with contrastive learning)
- **BLIP** (Bootstrap Language-Image Pre-training)
- **LLaVA** (Large Language and Vision Assistant)
- **Qwen3-VL-8B-Embedding** (Contrastive embedding variant)
- **Qwen3-VL-8B-Thinking** (Generative reasoning variant)

### Key Features

✨ **Scene Graph Augmentation**: Optional integration of compositional structure through scene graphs
✨ **Ablation Studies**: Caption-level ablations to understand model reasoning
✨ **Multi-turn Dialogue**: Generation with conversational context
✨ **Interpretability Analysis**: Attention and patching-based analysis for model understanding
✨ **Unified Evaluation Pipeline**: Streamlined evaluation across diverse models

## Project Structure

```
.
├── main.py                          # LLaVA evaluation with scene graphs
├── main_embeddings.py               # Qwen3-VL embedding-based evaluation
├── final.py                         # Unified multi-model evaluation
├── qwen_3_gen.py                    # Qwen3-VL generative evaluation
├── qwen_3_ablation.py               # Caption ablation experiments
├── qwen_3_gen_multi_turn.py         # Multi-turn dialogue generation
├── qwen_3_gen_multi_turn_json.py    # Multi-turn dialogue (JSON output)
├── qwen_sg_generation.py            # Scene graph generation from Qwen3
├── ablation_all.py                  # Comprehensive ablation study
├── qwen3_interpret.py               # Interpretability analysis
├── qwen3_probe.py                   # Attention probing & patching
├── llava_with_text_graph.py         # LLaVA with scene graph
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 30+ GB GPU VRAM (recommended for running all models)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository>
   cd llava_eval
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model** (for NLP/SG parsing)
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Authenticate with Hugging Face** (for model access)
   ```bash
   huggingface-cli login
   ```

## Usage

### Baseline Evaluations

#### 1. LLaVA Evaluation
```bash
# Single model evaluation
python main.py --max_samples 100

# With scene graph augmentation
python main.py --max_samples 100 --use_sg
```

#### 2. Qwen3-VL Embedding Evaluation
```bash
python main_embeddings.py --max_samples 100
```

#### 3. Unified Multi-Model Evaluation
```bash
# Evaluate all models
python final.py --methods all --max_samples 100

# Evaluate specific models
python final.py --methods clip blip llava qwen3 --max_samples 100

# With scene graphs
python final.py --methods llava llava_sg qwen3_gen qwen3_gen_sg --max_samples 50
```

### Generative Baselines

#### 4. Qwen3-VL Generative Evaluation
```bash
python qwen_3_gen.py --max_samples 100
```

#### 5. Multi-Turn Dialogue Generation
```bash
python qwen_3_gen_multi_turn.py --max_samples 50

# Output JSON format
python qwen_3_gen_multi_turn_json.py --max_samples 50
```

#### 6. Scene Graph Generation
```bash
python qwen_sg_generation.py --max_samples 50
```

### Ablation & Interpretability Studies

#### 7. Caption Ablation (Caption-level modifications)
```bash
# Tests which components of the caption the model relies on
python qwen_3_ablation.py --max_samples 100

# Supported ablations:
# - plain              (original caption)
# - mask_subj          (subject → [MASK])
# - mask_obj           (object → [MASK])
# - mask_rel           (relation → [MASK])
# - swap_subj_obj      (subject ↔ object swap)
# - shuffle_entities   (random entity reassignment)
# - replace_subj_rand  (random noun replacement)
```

#### 8. Complete Ablation Study
```bash
python ablation_all.py --max_samples 100
```

#### 9. Interpretability Analysis
```bash
python qwen3_interpret.py --max_samples 50
```

#### 10. Attention & Patching Analysis
```bash
python qwen3_probe.py --max_samples 20
```

## Arguments & Configuration

Common arguments across scripts:

| Argument | Description | Default |
|----------|-------------|---------|
| `--max_samples` | Number of samples to evaluate | 1000 |
| `--batch_size` | Batch size for processing | 1 |
| `--device` | Device to use ('cuda', 'cpu') | cuda |
| `--seed` | Random seed for reproducibility | 42 |
| `--output_dir` | Directory for results | ./results |
| `--use_sg` | Enable scene graph augmentation | False |
| `--methods` | Models to evaluate (space-separated) | - |
| `--max_new_tokens` | Max tokens for generation | 10 |

## Scripts Overview

### Evaluation Scripts
- **final.py**: Main entry point for unified multi-model evaluation with GPU management
- **main.py**: LLaVA-specific evaluation with optional scene graph augmentation
- **main_embeddings.py**: Qwen3-VL embedding-based scoring

### Generative Scripts
- **qwen_3_gen.py**: Qwen3-VL generative yes/no predictions
- **qwen_3_gen_multi_turn.py**: Multi-turn dialogue with spaCy tagging
- **qwen_3_gen_multi_turn_json.py**: Multi-turn dialogue with JSON serialization
- **qwen_sg_generation.py**: Scene graph generation for Winoground items

### Ablation & Analysis
- **qwen_3_ablation.py**: Caption-level ablation studies
- **ablation_all.py**: Comprehensive ablations (masking, swapping, shuffling)
- **qwen3_interpret.py**: Extract interpretability metrics
- **qwen3_probe.py**: Attention probing and patching experiments
- **llava_with_text_graph.py**: LLaVA with text-based scene graphs

## Output Format

Results are saved as JSON files with per-example and summary statistics:

```json
{
  "model": "llava",
  "method": "llava_sg",
  "total_samples": 100,
  "accuracy": 0.68,
  "per_example": [
    {
      "id": "0",
      "image_0": "path/to/image0.png",
      "caption_0": "...",
      "image_1": "path/to/image1.png",
      "caption_1": "...",
      "score": 0.95,
      "prediction": "correct"
    }
  ]
}
```

## Key Research Questions

1. **Do VLMs understand compositional structure?**
   → Tested via caption swapping and entity masking

2. **What role do scene graphs play?**
   → Compared base models vs. scene graph augmentation

3. **Embedding vs. Generative: Which is better for compositional reasoning?**
   → Compared Qwen3-Embedding vs. Qwen3-Generative

4. **Model-specific interpretability**
   → Attention analysis and patching experiments

## Dependencies

See `requirements.txt` for complete dependency list. Key packages:

- `torch` >= 2.0.0 - Deep learning framework
- `transformers` >= 4.35.0 - HuggingFace models
- `datasets` >= 2.14.0 - Dataset loading (Winoground)
- `PIL/Pillow` - Image processing
- `spacy` >= 3.7.0 - NLP utilities
- `tqdm` - Progress bars
- `numpy` - Numerical computing

## GPU Memory Requirements

| Model | Approximate VRAM |
|-------|------------------|
| CLIP | 2-3 GB |
| BLIP | 4-5 GB |
| LLaVA | 16-20 GB |
| Qwen3-VL (any variant) | 16-20 GB |

**Note**: Models are sequentially loaded/unloaded to prevent OOM errors.

## Citation

If you use this framework, please cite:

```bibtex
@software{llava_eval,
  title={LLaVA and Qwen3-VL Evaluation Framework},
  author={[Your Name]},
  year={2024},
  url={https://github.com/user/llava_eval}
}
```

## License

[Add appropriate license - MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### CUDA Out of Memory (OOM)
- Reduce `--batch_size` or `--max_samples`
- Ensure only one model is loaded at a time (use `--methods` to specify models)
- Check for background processes using GPU: `nvidia-smi`

### Model Download Timeout
- Increase timeout: `export HF_HUB_READ_TIMEOUT=600` (in seconds)
- Check internet connection
- Manually download from Hugging Face Hub

### spaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

### Authentication Required
```bash
huggingface-cli login
```

## Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: March 2024
