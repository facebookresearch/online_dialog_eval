# MaUde

**M**etric for **a**utomatic **U**nreferenced **d**ialog **e**valuation.

Contains code of the paper titled _"Learning an Unreferenced Metric for Online Dialogue Evaluation"_ to appear at **ACL 2020**.

## Getting the data

To get the trained models, [download the data from here](https://drive.google.com/file/d/1Ysso9hdzSenK13LjOFombyXYqA_kv-Vy/view?usp=sharing).

## Computing Backtranslation

We use [FairSeq](https://github.com/pytorch/fairseq) to compute back-translations. Our modified scripts are present in `scripts` folder, to run cd into that folder and run `./run_trans.sh`.  

## Computing Corruption Files

In the data dump we already provide the corruption files used for training. To generate new corruption files on the dataset, use `scripts/compute_corrupt.py`.

## Training Script

Uses [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) as the boilerplate for reproducibility.

```
python codes/trainer.py --mode train \
    --corrupt_type all \ 
    --batch_size 64 \
    --model_save_dir /checkpoint/koustuvs/dialog_metric/all_dailydialog_10_runs \
    --learn_down True --downsample True --down_dim 300 \
    --optim adam,lr=0.001 --dropout 0.2 --decoder_hidden 200 \ 
    --data_name convai2 \ 
    --data_loc /checkpoint/koustuvs/dialog_metric/convai2_data/ \
    --use_cluster
```

For baselines, add the appropriate flag:

```
--train_baseline [infersent/ruber/bertnli]
```

## Inference Script

```
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py \ 
    --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR \
    --model_version $VERSION --train_mode nce \ 
    --corrupt_pre $DATA_LOC --test_suffix true_response \ 
    --test_column true_response --results_file "results.jsonl"
```

Outputs the results in a `jsonl` file. To measure human correaltion with [See et al 2019](https://parl.ai/projects/controllable_dialogue/), specify `--human_eval` flag and `--human_eval_file` location.

## Acknowledgements

- Pytorch Lightning - https://github.com/williamFalcon/pytorch-lightning
- HuggingFace Transformers - https://github.com/huggingface/transformers
- FairSeq - https://github.com/pytorch/fairseq
- Liming Vie's RUBER implementation - https://github.com/liming-vie/RUBER
- Pytorch 1.2.0 - https://pytorch.org/
- ParlAI - https://parl.ai/
- See et al 2019 data - https://parl.ai/projects/controllable_dialogue/

## Citation

If our work is useful for your research, consider citing it using the following bibtex:

```
@article{sinha2020maude,
  Author = {Koustuv Sinha and Prasanna Parthasarathy and Jasmine Wang and Ryan Lowe and William L. Hamilton and Joelle Pineau},
  Title = {Learning an Unreferenced Metric for Online Dialogue Evaluation},
  Year = {2020},
  journal = {ACL},
  arxiv = {}
}
```

## License

This work is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) licensed, as found in the LICENSE file.
