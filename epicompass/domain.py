"""The domain interface: what the epistemic state needs to know about one benchmark. The core never sees the ground
truth; a domain computes every relation from what the agent itself observed and wrote.

Construction
  key(artifact) -> hashable           hypothesis identity (same key = same hypothesis; a new version replaces the old)
  describe_hyp(artifact) -> [str]     optional: the lines that say what a hypothesis is (default: the law's code core)

Update (hypothesis check and belief)
  errors(artifact, observations, refit) -> [float]   error of the hypothesis on each observation (inf = no finite
                                      prediction; None = unknown). Parameters the benchmark itself would fit are fitted
                                      here; refit=False may reuse parameters fitted on fewer observations
  n_params(artifact) -> int           free parameters (the belief prompt mentions them)
  misfit_threshold: float | None      error above which an experiment conflicts with a hypothesis, for benchmarks with
                                      exact observations; None for noisy observations (a relative rule is used)

Feedback (wording)
  error_unit, describe_obs(obs), compare(artifact, obs)
  discriminate(a, b, observations) -> [{"input_text", "what", "pred_a", "pred_b"}]   optional: untried experiments on
                                      which two hypotheses disagree most
"""
