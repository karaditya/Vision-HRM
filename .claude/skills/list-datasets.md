# List Available Datasets

Show all available datasets for training.

## Purpose
- See what datasets are available
- Get dataset information
- Choose dataset for training

## Usage

```bash
/list-datasets
```

## Commands

```bash
echo "=================================="
echo "Available Training Datasets"
echo "=================================="
echo ""

python train_language_model.py --list_datasets

echo ""
echo "=================================="
echo "Usage Examples"
echo "=================================="
echo ""
echo "Train on WikiText (testing):"
echo "  python train_language_model.py --dataset wikitext"
echo ""
echo "Train on OpenWebText (production):"
echo "  python train_language_model.py --dataset openwebtext --use_streaming"
echo ""
echo "Train on C4 (large-scale):"
echo "  python train_language_model.py --dataset c4 --use_streaming"
echo ""
echo "Or use the skill:"
echo "  export DATASET=wikitext"
echo "  export MODEL_SIZE=medium"
echo "  /train-production"
echo ""
```

## Quick Reference

| Dataset | Size | Type | Best For |
|---------|------|------|----------|
| tiny_shakespeare | 1MB | Text | Quick testing |
| wikitext | 500MB | Wikipedia | Small models |
| openwebtext | 40GB | Web text | GPT-2 style |
| c4 | 750GB | Common Crawl | Large-scale |
| the_pile | 825GB | Mixed | General purpose |
| wikipedia | 20GB | Encyclopedia | Factual knowledge |
| bookcorpus | 5GB | Books | Long-form text |
| the_stack | 3TB | Source code | Code models |

## Next Steps

After choosing a dataset:

1. Install datasets library:
```bash
pip install datasets
```

2. Train:
```bash
python train_language_model.py \
    --dataset DATASET_NAME \
    --use_streaming
```

3. Or use skill:
```bash
export DATASET=wikitext
/train-production
```
