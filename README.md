# One-to-many GAN

Style-conditioned shoeprint-to-shoemark translation: one shoeprint maps to many
plausible shoemarks, the style vector selecting which.

## Configuration

Options resolve as **defaults < `config.toml` < command line**. Every option is
declared in `src/one_to_many_gan/data/config.py`; an unknown key in the TOML or
on the command line is an error rather than a silent no-op. Command-line
overrides name the option by its dotted path and take TOML literals:

```sh
uv run python src/train.py                       # config.toml as it stands
uv run python src/train.py my_run.toml           # a different config file
uv run python src/train.py --training.batch_size 8 --optimisation.true_kl_loss true
```

## Runs reported in the thesis

Each is `config.toml` plus overrides, so the differences between arms are
visible in the command rather than spread across near-duplicate files. All use
200,000 batch-agnostic steps (50,000 steps at batch 4) and skip validation
during training, because the reported figures come from the seeded post-hoc
sweep in `src/rescore_best.py`.

Shared by every arm below:

```sh
COMMON="--training.batch_agnostic_steps 200_000 \
  --evaluation.checkpoint_interval 5_000 \
  --evaluation.validate_during_training false \
  --evaluation.use_training_data true"
```

**Style-only ablation** — every loss but the adversarial and style-cycle terms
disabled, testing what style conditioning alone can carry. Seeds 4242/422/423:

```sh
uv run python src/train.py $COMMON \
  --training.training_run only_style_1 --training.random_seed 4242 \
  --optimisation.identity_loss_lambda 0.0 \
  --optimisation.reconstruction_loss_lambda 0.0 \
  --optimisation.kl_loss_lambda 0.0 \
  --optimisation.path_loss_lambda 0.0
```

**SANTA KL substitution** — the Xie et al. KL formulation (mean squared
latents) in place of moment matching, which requires latent noise. Note seed
421 for the first run, 422 and 423 after it:

```sh
uv run python src/train.py $COMMON \
  --training.training_run santa_kl_1 --training.random_seed 421 \
  --optimisation.true_kl_loss true \
  --architecture.add_latent_noise true
```

The configs as they were actually run are preserved at the `pre-restructure`
tag, before they became overrides.
