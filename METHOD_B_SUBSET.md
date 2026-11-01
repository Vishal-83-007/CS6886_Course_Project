# Method B: random-subset forgetting (`sub6_9_p10`)

Forget a fixed random **10% of the training images of classes 6 and 9** while the
other 90% of those classes stays: MNIST 1,185 images (591 + 594), Fashion-MNIST
1,200 (600 + 600). The exact IDs are in `forget_set_mnist_sub6_9_p10.csv` and
`forget_set_fashion_sub6_9_p10.csv` (training-set index, label); the same set is
used by the Method C subset runner. Selection rule (`src/fashion_data.py`): for
class 6 then 9, shuffle that class's training indices with
`numpy.random.default_rng(46)` and keep the first 10%.

The matched Gold is retrained without these images (`REQUEST=sub6_9_p10`;
58,815 / 58,800 training images). A Gold still recognises the forgotten images
through the rest of their class, so **good unlearning matches Gold, not 0%.**

## Run

```bash
# importance over the forgotten images only (minutes)
python run_frozen.py importance --dataset mnist --weights /path/to/mnist_baseline/baseline_weights.npz \
  --forget sub6_9_p10 --importance-set forget --poisson-seed 1 --out-dir results/mnist/importance/df

python run_method_b.py --dataset mnist --baseline-dir /path/to/mnist_baseline --request sub6_9_p10 \
  --importance results/mnist/importance/seed1/importance.npz \
  --forget-importance results/mnist/importance/df/importance.npz \
  --alpha 10 --lam 1 --gold-dir /path/to/mnist_sub6_9_p10_gold --out-dir results/mnist_method_b_subset
```

Aqua: `qsub -v DATASET=mnist,MODEL=$MB,FORGET=sub6_9_p10 jobs/importance_cpp.cmd`, then
`jobs/subset_sweep.sh` (α = 2, 5, 10, 25; λ = 1) and, for Gold,
`qsub -v "DATASET=mnist,WEIGHTS=$GM/baseline_weights.npz,FORGET=sub6_9_p10,FORGET_SET=1,FIXED_LABELS=$GM/neuron_labels.npy,EVAL_SPLIT=all,NAME=sub_gold" jobs/evaluate_cpp.cmd`.

## Evaluation (`--forget-set`)

On the full test set, plus: accuracy on D_f, on an equal-size random sample of
*kept* 6/9 training images, on test 6/9 and on the other classes; and a
membership AUC (feature: true-class score minus the best other class; members =
those training images, non-members = test 6/9; 0.5 = no membership signal).
Labelling never uses D_f.

## Findings

- The baseline holds no image-specific memory: it classifies D_f like unseen test
  images (MNIST 87.5% vs 88.1%; Fashion 60.0% vs 59.8%), membership AUC 0.48–0.50.
- I(D_f) correlates 0.97 (MNIST) / 0.94 (Fashion) with the importance of the whole
  classes. α = 2 wipes out 6/9 entirely (0%); α = 5 damages the whole classes
  (MNIST −30, Fashion −9 points on D_f, kept and test images alike); α ≥ 10 changes
  nothing. No α removes the images without removing their class.
- Retraining noise is larger than any forgetting effect: Gold's test 6/9 accuracy
  differs from the baseline's by −1.9 (MNIST) / +5.8 (Fashion) points.
- Conclusion: in this prototype-based network the right action for random-subset
  forgetting is no edit; importance-based dampening can only over-forget.

Full tables: [RESULTS_B_D.md](RESULTS_B_D.md).
