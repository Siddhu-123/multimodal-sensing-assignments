# Assignment 1 - FER2013 Facial-Expression Classification

## Submit this file

`FER2013_Assignment_1_Report.pdf`

The report is under the ten-page limit and has separate sections for Tasks 1-4. The executed notebook and recorded results are included in this folder.

## Dataset check

The supplied archive is the original **FER2013** dataset, not the FER+ relabelled dataset named in the brief. It contains 35,887 grayscale 48 x 48 images and the standard `Training`, `PublicTest`, and `PrivateTest` partitions. The report names the dataset accordingly.

Source: [Kaggle FER2013 repository](https://www.kaggle.com/datasets/deadskull7/fer2013), downloaded 25 August 2026. The final executed notebook records Python 3.10.16; the course specification lists Python 3.11.8.

## Main deliverables

- `fer2013_analysis.ipynb` - executed PyTorch notebook.
- `FER2013_Assignment_1_Report.tex` - LaTeX source.
- `fer2013_references.bib` - references.
- `FER2013_Assignment_1_Report.pdf` - submission PDF.
- `figures/` - plots used by the notebook and report, including the class-wise mean-image EDA figure.
- `metrics.csv` and `results.json` - recorded experiment results.
- `requirements.txt` - Python dependencies.

## Results

| Model | Accuracy | Macro-F1 | Weighted-F1 | Selected epoch |
|---|---:|---:|---:|---:|
| MLP | 0.377 | 0.247 | 0.312 | 14 |
| CNN | **0.544** | **0.430** | **0.519** | 10 |

The CNN is the stronger model in this experiment. It is not suitable for classroom decisions. Both models used the same split, optimiser, 15-epoch limit, and stochastic training-only augmentation pipeline. The MLP has 295,943 trainable parameters and the CNN has 730,151, so this is not a parameter-controlled ablation. Epoch 14 was selected for the MLP and epoch 10 for the CNN using validation macro-F1.

## Reproduce

Install the Python packages:

```bash
python -m pip install -r requirements.txt
```

If the CSV is absent, extract it without replacing an existing copy:

```bash
unzip -n fer2013.csv.zip fer2013.csv
```

Execute the notebook in place:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 fer2013_analysis.ipynb
```

The notebook selects CUDA first, then Apple MPS, then CPU. The recorded experiment ran on Apple MPS. The pinned PyTorch 2.9.0 and torchvision 0.24.0 versions are a supported pair.

Compile the report:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf FER2013_Assignment_1_Report.tex
```
