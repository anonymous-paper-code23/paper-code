# paper-code
cli.py is the main script for the experiment.

Here is an instance for running it:

```python
python cli.py --method pet --pattern_ids 0 1 2 3 --data_dir ./data_path/ --model_type roberta_adapter_1 --model_name_or_path roberta-large --task_name snopes --output_dir /output_dir/output.file --overwrite_output_dir --do_train --do_eval --eval_set test --pet_per_gpu_train_batch_size 16 --pet_per_gpu_eval_batch_size 64 --learning_rate 7e-4 --train_examples -1 --test_examples -1 --no_distillation --pet_repetitions 10
```

