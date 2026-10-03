#!/usr/bin/env Rscript
# Attentional Thief — fERP/FRP single-fixation mixed models with image random effects (2026-05-21)
#
# Input: one row per kept fixation epoch and window (data/revision/), created from the
# epoched EEG in the full project repository.
#
# Model per latency window:
#   amplitude_uv ~ condition + (1 | subject) + (1 | image_id)
# with contrasts from emmeans, unadjusted because each contrast is reported as a
# planned model coefficient/contrast for the same prespecified window table.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

# Run from the open_data/ root:  Rscript code/revision/frp_single_fixation_models.R
# Requires readr, dplyr, lme4, lmerTest, emmeans.
out_dir <- file.path("output", "revision", "frp_single_fixation_models")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
in_path <- file.path("data", "revision", "frp_single_fixation_epoch_amplitudes.csv.gz")
qc_path <- file.path("data", "revision", "frp_epoch_image_mapping_qc.csv")
model_path <- file.path(out_dir, "frp_lmer_model_qc_20260521.csv")
contrast_path <- file.path(out_dir, "frp_mixedlm_condition_image_re_20260521.csv")
summary_path <- file.path(out_dir, "frp_image_random_effect_summary_20260521.md")

fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < .001) return("< .001")
  sprintf("= %.3f", p)
}

amp <- read_csv(in_path, show_col_types = FALSE) %>%
  mutate(
    condition = factor(condition, levels = c("passive", "active", "constant")),
    subject = factor(subject),
    image_id = factor(image_id)
  ) %>%
  filter(!is.na(amplitude_uv), !is.na(condition), !is.na(subject), !is.na(image_id))

windows <- unique(amp$window)
model_rows <- list()
contrast_rows <- list()

for (win in windows) {
  d <- amp %>% filter(window == win) %>% droplevels()
  means <- d %>% group_by(condition) %>% summarise(mean_uv = mean(amplitude_uv), n = n(), .groups = "drop")
  mean_active <- means$mean_uv[means$condition == "active"]
  mean_passive <- means$mean_uv[means$condition == "passive"]
  mean_constant <- means$mean_uv[means$condition == "constant"]
  n_active <- means$n[means$condition == "active"]
  n_passive <- means$n[means$condition == "passive"]
  n_constant <- means$n[means$condition == "constant"]

  fit <- tryCatch(
    lmer(
      amplitude_uv ~ condition + (1 | subject) + (1 | image_id),
      data = d,
      REML = FALSE,
      control = lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 200000),
        check.conv.singular = "ignore"
      )
    ),
    error = function(e) e
  )

  base <- data.frame(
    window = win,
    n_epochs = nrow(d),
    n_subjects = n_distinct(d$subject),
    n_images = n_distinct(d$image_id),
    mean_active_uv = ifelse(length(mean_active) == 0, NA_real_, mean_active),
    mean_passive_uv = ifelse(length(mean_passive) == 0, NA_real_, mean_passive),
    mean_constant_uv = ifelse(length(mean_constant) == 0, NA_real_, mean_constant),
    n_active = ifelse(length(n_active) == 0, 0L, n_active),
    n_passive = ifelse(length(n_passive) == 0, 0L, n_passive),
    n_constant = ifelse(length(n_constant) == 0, 0L, n_constant)
  )

  if (inherits(fit, "error")) {
    model_rows[[length(model_rows) + 1]] <- cbind(base, data.frame(
      model = "lmer amplitude_uv ~ condition + (1|subject) + (1|image_id)",
      converged = FALSE,
      singular = NA,
      logLik = NA_real_,
      AIC = NA_real_,
      subject_sd = NA_real_,
      image_sd = NA_real_,
      residual_sd = NA_real_,
      error = conditionMessage(fit)
    ))
    next
  }

  vc <- as.data.frame(VarCorr(fit))
  get_sd <- function(grp) {
    z <- vc$sdcor[vc$grp == grp]
    ifelse(length(z) == 0, NA_real_, z[1])
  }
  model_rows[[length(model_rows) + 1]] <- cbind(base, data.frame(
    model = "lmer amplitude_uv ~ condition + (1|subject) + (1|image_id)",
    converged = is.null(fit@optinfo$conv$lme4$messages),
    singular = isSingular(fit, tol = 1e-4),
    logLik = as.numeric(logLik(fit)),
    AIC = AIC(fit),
    subject_sd = get_sd("subject"),
    image_sd = get_sd("image_id"),
    residual_sd = sigma(fit),
    error = ""
  ))

  emm <- emmeans(fit, ~ condition)
  cont <- contrast(
    emm,
    method = list(
      active_minus_passive = c(-1, 1, 0),
      constant_minus_passive = c(-1, 0, 1),
      active_minus_constant = c(0, 1, -1)
    ),
    adjust = "none"
  )
  cs <- as.data.frame(summary(cont, infer = TRUE))
  if ("t.ratio" %in% names(cs)) names(cs)[names(cs) == "t.ratio"] <- "t"
  if ("z.ratio" %in% names(cs)) names(cs)[names(cs) == "z.ratio"] <- "t"
  if ("p.value" %in% names(cs)) names(cs)[names(cs) == "p.value"] <- "p"
  if ("asymp.LCL" %in% names(cs)) names(cs)[names(cs) == "asymp.LCL"] <- "lower.CL"
  if ("asymp.UCL" %in% names(cs)) names(cs)[names(cs) == "asymp.UCL"] <- "upper.CL"
  if (!("df" %in% names(cs))) cs$df <- NA_real_
  if (!("lower.CL" %in% names(cs))) cs$lower.CL <- NA_real_
  if (!("upper.CL" %in% names(cs))) cs$upper.CL <- NA_real_
  contrast_rows[[length(contrast_rows) + 1]] <- cbind(base[rep(1, nrow(cs)), ], cs)
}

models <- bind_rows(model_rows)
contrasts <- bind_rows(contrast_rows) %>%
  rename(estimate_uv = estimate, se = SE, ci_low = lower.CL, ci_high = upper.CL) %>%
  select(window, contrast, estimate_uv, se, df, ci_low, ci_high, t, p,
         n_epochs, n_subjects, n_images,
         mean_active_uv, mean_passive_uv, mean_constant_uv,
         n_active, n_passive, n_constant)

write_csv(models, model_path)
write_csv(contrasts, contrast_path)

qc <- read_csv(qc_path, show_col_types = FALSE) %>% filter(status == "ok")
lines <- c(
  "# fERP/FRP image-random-effect mixed model — 2026-05-21",
  "",
  "## Mapping QC",
  sprintf("- Subjects with usable FRP epochs: %d.", sum(qc$n_epochs_saved > 0, na.rm = TRUE)),
  sprintf("- Retained FRP epochs checked: %s.", format(sum(qc$n_epochs_selection_retained_checked, na.rm = TRUE), big.mark = ",")),
  sprintf("- Retained FRP epochs with image ID: %s.", format(sum(qc$n_retained_with_image_id, na.rm = TRUE), big.mark = ",")),
  sprintf("- Retained FRP epochs missing image ID: %s.", format(sum(qc$n_retained_missing_image_id, na.rm = TRUE), big.mark = ",")),
  sprintf("- Sample mismatches between saved epochs and reconstructed fixation events: %d.", sum(qc$n_sample_mismatch, na.rm = TRUE)),
  sprintf("- Condition mismatches between saved epochs and reconstructed fixation events: %d.", sum(qc$n_condition_mismatch, na.rm = TRUE)),
  "- Linkage is reconstructed from ET fixation intervals + sync pairs + behavioral image order, then verified against saved FRP epoch samples via `epochs.selection`.",
  "",
  "## Mixed-model contrasts",
  "Model: `amplitude_uv ~ condition + (1 | subject) + (1 | image_id)`; posterior-occipital ROI; lmer/lmerTest; unadjusted planned contrasts.",
  ""
)

for (i in seq_len(nrow(contrasts))) {
  r <- contrasts[i, ]
  lines <- c(lines, sprintf(
    "- %s / %s: β=%.3f µV, SE=%.3f, t(%s)=%.2f, p %s, Nepoch=%s, Nsub=%d, Nimage=%d.",
    r$window,
    r$contrast,
    r$estimate_uv,
    r$se,
    ifelse(is.na(r$df), "NA", sprintf("%.1f", r$df)),
    r$t,
    fmt_p(r$p),
    format(r$n_epochs, big.mark = ","),
    r$n_subjects,
    r$n_images
  ))
}

lines <- c(lines,
  "",
  "## Model QC",
  sprintf("- Singular fits: %s.", paste(models$window[models$singular %in% TRUE], collapse = ", ")),
  "",
  "## Output files",
  sprintf("- `%s`", in_path),
  sprintf("- `%s`", qc_path),
  sprintf("- `%s`", model_path),
  sprintf("- `%s`", contrast_path)
)
writeLines(lines, summary_path)
cat(paste(lines, collapse = "\n"), "\n")
