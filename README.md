# One-to-many GAN

Style-conditioned translation of shoeprints to shoemarks. A single shoeprint
maps to many plausible shoemarks, selected by a style vector.

## Configuration

Options resolve as defaults, then `config.toml`, then command-line arguments.
All options are declared in `src/one_to_many_gan/data/config.py`. An unknown key
in the configuration file or on the command line is an error. Overrides name an
option by its dotted path and accept TOML literals.

```sh
uv run python src/train.py
uv run python src/train.py my_run.toml
uv run python src/train.py --training.batch_size 8 --optimisation.true_kl_loss true
```

## Reported runs

Each arm is `config.toml` with overrides. All use 200,000 batch-agnostic steps
(50,000 steps at batch 4) and disable validation during training, as the
reported figures come from the seeded sweep in `src/rescore_best.py`.

Settings common to both arms:

```sh
COMMON="--training.batch_agnostic_steps 200_000 \
  --evaluation.checkpoint_interval 5_000 \
  --evaluation.validate_during_training false \
  --evaluation.use_training_data true"
```

Style-only ablation, disabling every loss except the adversarial and
style-cycle terms. Seeds 4242, 422, 423:

```sh
uv run python src/train.py $COMMON \
  --training.training_run only_style_1 --training.random_seed 4242 \
  --optimisation.identity_loss_lambda 0.0 \
  --optimisation.reconstruction_loss_lambda 0.0 \
  --optimisation.kl_loss_lambda 0.0 \
  --optimisation.path_loss_lambda 0.0
```

SANTA KL substitution, replacing moment matching with the formulation of Xie et
al., which requires latent noise. Seeds 421, 422, 423:

```sh
uv run python src/train.py $COMMON \
  --training.training_run santa_kl_1 --training.random_seed 421 \
  --optimisation.true_kl_loss true \
  --architecture.add_latent_noise true
```

Configurations as originally run are preserved at the `pre-restructure` tag.
