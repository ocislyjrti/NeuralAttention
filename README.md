
# Neural Attention

Source code and datasets for the 'Neural Attention: Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks' research paper, exploring enhanced QKV computation within the self-attention mechanism using neural networks.

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/ocislyjrti/NeuralAttention.git
   ```

2. Navigate to the `transformers` directory:
   ```
   cd NeuralAttention/transformers
   ```

3. Install the necessary dependencies:
   ```
   pip install -e .
   ```

## Experiments

### Experiment 1

For the first experiment involving the Roberta model and the Wikitext-103 dataset, navigate to the language modeling directory and install additional requirements:

```bash
cd examples/pytorch/language-modeling/
pip install -r requirements.txt
```

Then, execute the following command:

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --num_train_epochs 5 \
    --save_steps 1000000 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --logging_steps 50 \
    --output_dir /tmp/test-mlm
```

### Experiment 2

For the second experiment involving the Marian model and the IWSLT 2017 dataset, navigate to the translation directory and install additional requirements:

```bash
cd NeuralAttention/transformers-main/examples/pytorch/translation
pip install -r requirements.txt
```

Then, execute the following command:

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-de-en \
    --do_train \
    --do_eval \
    --source_lang de \
    --target_lang en \
    --dataset_name iwslt2017 \
    --dataset_config_name iwslt2017-de-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --num_train_epochs 6 \
    --save_steps 100000000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 200 \
    --predict_with_generate
```

## Citing

If you find this code or the associated paper useful in your research, please consider citing:

```
@article{zhang2023neural,
  title={Neural Attention: Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks},
  author={Zhang, Muhan},
  journal={arXiv preprint arXiv:2310.11398},
  year={2023}
}
```

## Acknowledgements

Special thanks to the Hugging Face team for their [Transformers](https://github.com/huggingface/transformers) library, which provided a foundation for this work.

