# <center>RULER: Discriminative and Iterative Adversarial Training for Deep Neural Network Fairness</center>

## 1.Ruler

​		Our technique  will  mitigate unfairness during the process of training, which leads to a more concise deployment.

### 1.1.Train

  		Come into the `Ruler` directory, and run below commands. You will obtain the result shown in our paper.

#### Adult(census)

```bash
python -u main.py --dataset adult --adv_ratio 0.3 --adv_epoch 70 --accuracy_threshold 83.5 --protected_attribs 0 6 7 --gpu_id 0 --save_path results_adult
```

#### Bank

```bash
python -u main.py --dataset bank --adv_ratio 0.3 --adv_epochs 70 --accuracy_threshold 89.5 --protected_attribs 0   --save_path result_bank  
```



#### German

```bash
python -u main.py --dataset german --adv_ratio 0.3 --protected_attribs 6 9 --accuracy_threshold 77.0 --save_path results_german --lr 0.003
```





#### COMPAS

```python
python -u main.py --dataset compas --accuracy_threshold 75.5 --adv_ratio 0.3 --protected_attribs 2 --save_path results_compas --adv_epochs 70
```



### 1.2.Metrics

​		Unfairness and accuracy, we take compas dataset as an instance.

​		Attention.  The parameters `model_start` and `model_end` depends on the your own choice.

​		The parameters `sample_round` and  `num_gen` determine the test scale.  You can take  (`sample_round`=10, `num_gen`=100) for a sample test.

```bash
python -u evaluators/evaluate_main.py --dataset compas --mode unfairness --model_start XX --model_end XX --model_path results_compas/train  --sample_round 100 --num_gen 10000

python -u evaluators/evaluate_main.py --dataset compas --mode test-acc --model_start XX --model_end XX --model_path results_compas/train  
```



