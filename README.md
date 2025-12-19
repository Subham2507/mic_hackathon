# Unsupervised Microstructure Image Analysis and Question Answering Using CLIP and VLMs

A hackathon project for automated microscopy technique classification and material categorization using CLIP embeddings and Vision-Language Models (VLMs).

## 🔬 Project Overview

This project implements zero-shot and few-shot learning approaches to:  
- **Classify microscopy techniques** (SEM, TEM, AFM, Optical/Reflected Light Microscopy)
- **Categorize material types** (Metal/Alloy, Ceramic, Polymer, Composite, Fracture)
- **Perform semantic search** on microscopy images using natural language queries
- **Visualize embedding spaces** using UMAP dimensionality reduction

## 🎯 Key Features

- **CLIP-based Image Embedding**:  Utilizes OpenAI's CLIP (ViT-B/32) for feature extraction
- **Zero-shot Classification**: Achieves 53.56% accuracy on technique classification without training
- **Vision-Language Model Evaluation**: Tests LLaVA-v1.6-Mistral-7B and Qwen3-VL-8B-Instruct on multi-choice questions
- **UMAP Visualization**: Projects 512-dim embeddings to 2D for clustering analysis
- **Text-to-Image Search**: Natural language queries to retrieve relevant micrographs

## 📊 Results

### CLIP Zero-shot Performance
| Task | Metric | Score |
|------|--------|-------|
| Technique Classification | Top-1 Accuracy | 53.56% |
| Technique Classification | Top-2 Accuracy | 79.81% |
| Category Classification | Top-1 Accuracy | 34.56% |
| Category Classification | Top-2 Accuracy | 52.26% |

### LLaVA-v1.6-Mistral-7B Performance
| Task | Accuracy | Balanced Accuracy | Macro F1 |
|------|----------|-------------------|----------|
| Technique MCQ | 26.1% | 25.0% | 0.103 |
| Category MCQ | 55.6% | 25.0% | 0.179 |

### Qwen3-VL-8B-Instruct Performance ⭐
| Task | Accuracy | Balanced Accuracy | Macro F1 |
|------|----------|-------------------|----------|
| Technique MCQ | **91.3%** | **86.2%** | **0.888** |
| Category MCQ | 37.8% | 38.2% | 0.329 |

**Processing Time**: 3220. 61 seconds total (~63. 15 seconds per image)

## 🗂️ Repository Structure

```
mic_hackathon/
├── Untitled.ipynb            # CLIP embedding generation
├── Untitled1.ipynb           # UMAP visualization & zero-shot evaluation
├── mistral. ipynb             # LLaVA VLM MCQ evaluation
├── qwen. ipynb                # Qwen3-VL-8B MCQ evaluation
├── clip_image_embeddings.npy # Pre-computed CLIP features (867×512)
├── clip_metadata_clean.csv   # Image metadata (paths, categories, techniques)
├── micrographs_metadata.csv  # Full dataset metadata
├── qa. csv                    # Evaluation question-answer pairs
├── umap. png                  # UMAP projection visualization
├── miclib_output/            # Downloaded microscopy images
│   └── images/
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision transformers
pip install scikit-learn pandas numpy
pip install umap-learn matplotlib pillow tqdm
```

### 1. Generate CLIP Embeddings

Run `Untitled.ipynb` to:
- Load CLIP ViT-B/32 model
- Process microscopy images from `miclib_output/images/`
- Generate 512-dimensional embeddings
- Save to `clip_image_embeddings.npy`

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed images
embeddings = []
for image_path in image_paths:
    image = Image. open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    features = model. get_image_features(**inputs)
    embeddings.append(features.numpy())
```

### 2. Visualize & Evaluate

Run `Untitled1.ipynb` to:
- Load pre-computed embeddings
- Apply UMAP for 2D projection
- Perform zero-shot technique classification
- Run text-to-image retrieval

**Example Query**:
```python
query = "a scanning electron microscopy image of a ceramic"
# Returns top-5 most similar images
```

### 3. VLM Evaluation

#### LLaVA Evaluation (Optional)
Run `mistral.ipynb` to:
- Evaluate LLaVA-v1.6-Mistral-7B on MCQ tasks

#### Qwen3-VL Evaluation
Run `qwen.ipynb` to:
- Evaluate Qwen3-VL-8B-Instruct on technique and category MCQ tasks
- Achieves state-of-the-art 91.3% accuracy on technique classification

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch. bfloat16,
    device_map="auto"
)
processor = AutoProcessor. from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

## 📈 UMAP Visualization
The UMAP plot shows clustering of different microscopy techniques in the CLIP embedding space, demonstrating the model's ability to group similar imaging modalities.

## 🧪 Dataset

- **Source**: Materials microscopy image library (miclib)
- **Size**: 867 labeled micrographs
- **Techniques**: SEM (134), Optical (651), TEM (36), AFM (21)
- **Categories**: Metal/Alloy, Ceramic, Polymer, Composite, Fracture, etc.

### Data Files
- `micrographs_metadata.csv`: Full dataset with categories and techniques
- `clip_metadata_clean.csv`: Filtered dataset used for evaluation
- `qa.csv`: 51 test samples for VLM evaluation

## 🔍 Methodology

### CLIP Zero-shot Classification
1. Embed all images using CLIP vision encoder
2. Create text prompts:  `"a [technique] micrograph"`
3. Compute cosine similarity between image and text embeddings
4. Predict class with highest similarity

### UMAP Dimensionality Reduction
```python
from umap import UMAP

reducer = UMAP(
    n_neighbors=20,
    min_dist=0.15,
    metric="cosine"
)
X_2d = reducer.fit_transform(X_img)
```

### Vision-Language Model MCQ Evaluation
**Qwen3-VL approach** (best performing):
1. Load Qwen3-VL-8B-Instruct model with bfloat16 precision
2. Format image + text prompt as chat messages
3. Generate answer with max 5 new tokens (single letter:  A/B/C/D/E)
4. Parse predicted letter and compare with ground truth
5. Calculate accuracy, balanced accuracy, and macro F1-score

## 📝 Key Findings

1. **Qwen3-VL significantly outperforms other models** on technique classification (91.3% vs 53.56% CLIP, 26.1% LLaVA)
2. **Balanced accuracy of 86.2%** shows robust performance across all microscopy technique classes
3. **Category classification remains challenging** across all models (37.8% best accuracy)
4. **CLIP performs well** on technique classification without any fine-tuning (53.56% top-1, 79.81% top-2)
5. **LLaVA struggles** with domain-specific microscopy knowledge
6. **UMAP reveals clustering** of similar imaging techniques
7. **Text-to-image search works** for semantic material queries
8. **Qwen3-VL demonstrates strong vision understanding** for scientific image analysis

## 🏆 Performance Comparison

| Model | Task | Accuracy | Macro F1 |
|-------|------|----------|----------|
| **Qwen3-VL-8B** | Technique | **91.3%** | **0.888** |
| CLIP (zero-shot) | Technique | 53.56% | - |
| LLaVA-v1.6-Mistral-7B | Technique | 26.1% | 0.103 |
| **LLaVA-v1.6-Mistral-7B** | Category | **55.6%** | **0.179** |
| Qwen3-VL-8B | Category | 37.8% | 0.329 |
| CLIP (zero-shot) | Category | 34.56% | - |

## 🤝 Contributing

This is a hackathon project.  Feel free to fork and improve!  

## 📧 Contact

For questions or collaborations, reach out via GitHub issues.  

## 📄 License

MIT License - see LICENSE file for details

---

**Note**: This project was developed during the MIC Hackathon to explore zero-shot learning for materials science image analysis. 
