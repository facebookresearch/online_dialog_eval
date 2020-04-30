## Experiments

Uses [Pytorch Lightning](https://github.com/williamFalcon/pytorch-lightning) as the base framework.

## LM Finetuning

Clone pytorch-transformers first.

1. Load Prepare data for finetuning (`--only_data`)
2. Navigate to `pytorch-transformers/examples/lm_finetuning`
3. Pregenerate the data: 

  ```
  python pregenerate_training_data.py \
    --train_corpus ~/mlp/latentDialogAnalysis/fine_tune_convai2.txt \ --bert_model bert-base-uncased \ 
    --do_lower_case \
    --output_dir ~/mlp/latentDialogAnalysis/fine_tune_convai2_ep_10/ \ --epochs_to_generate 10 \
    --max_seq_len 256
  ```

4. Run finetuning script

  ```
  python finetune_on_pregenerated.py --pregenerated_data ~/mlp/latentDialogAnalysis/fine_tune_convai2_ep_10/ --bert_model bert-base-uncased --do_lower_case --output_dir ~/mlp/latentDialogAnalysis/fine_tune_convai2_ep_10_lm/ --epochs 10
  ```


## Extract model responses

```
python data.py
```

## Baselines

### InferSent

```
python codes/trainer.py --mode train --bidirectional --downsample --id infersent_word_order --load_model_responses --corrupt_type word_order --corrupt_model_names seq2seq --train_mode ref_score --train_baseline infersent --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id infersent_word_drop --load_model_responses --corrupt_type word_drop --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline infersent --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id infersent_model_true --load_model_responses --corrupt_type seq2seq --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline infersent --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id infersent_model_false --load_model_responses --corrupt_type model_false --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline infersent --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id infersent_rand_utt --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline infersent --use_cluster
```

### Ruber - MSE

```
python codes/trainer.py --mode train --bidirectional --downsample --id rubermse_word_order --load_model_responses --corrupt_type word_order --corrupt_model_names seq2seq --train_mode ref_score --train_baseline ruber --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id rubermse_word_drop --load_model_responses --corrupt_type word_drop --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline ruber --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id rubermse_model_true --load_model_responses --corrupt_type seq2seq --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline ruber --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id rubermse_model_false --load_model_responses --corrupt_type model_false --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline ruber --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id rubermse_rand_utt --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --train_baseline ruber --use_cluster
```

### BERT NLI


```
python codes/trainer.py --mode train --bidirectional --downsample --id bertnli_word_order --load_model_responses --corrupt_type word_order --corrupt_model_names seq2seq --train_mode ref_score --train_baseline bertnli --batch_size 8 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id bertnli_word_drop --load_model_responses --corrupt_type word_drop --corrupt_model_names seq2seq --train_mode ref_score --batch_size 8 --train_baseline bertnli --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id bertnli_model_true --load_model_responses --corrupt_type seq2seq --corrupt_model_names seq2seq --train_mode ref_score --batch_size 8 --train_baseline bertnli --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id bertnli_model_false --load_model_responses --corrupt_type model_false --corrupt_model_names seq2seq --train_mode ref_score --batch_size 8 --train_baseline bertnli --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id bertnli_rand_utt --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 8 --train_baseline bertnli --use_cluster
```

## Model

local GPU

```
python codes/trainer.py --mode train --bidirectional --downsample --id bert_rand --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --gpus 0 --use_dp
```

### slurm

```
python codes/trainer.py --mode train --bidirectional --downsample --id model_rand --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id model_rand --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```

###  Running TransitionFn

```
python codes/trainer.py --mode train --bidirectional --downsample --id stage_word_order --load_model_responses --corrupt_type word_order --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id stage_word_drop --load_model_responses --corrupt_type word_drop --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id stage_model_true --load_model_responses --corrupt_type seq2seq --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id stage_model_false --load_model_responses --corrupt_type model_false --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```

```
python codes/trainer.py --mode train --bidirectional --downsample --id stage_rand_utt --load_model_responses --corrupt_type rand_utt --corrupt_model_names seq2seq --train_mode ref_score --batch_size 64 --use_cluster
```


