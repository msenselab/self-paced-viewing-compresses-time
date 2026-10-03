# Saccade-amplitude covariate on the single-fixation occipital lambda (Supplement S6, Figure S6.2B).
# Run from the open_data/ root:  Rscript code/revision/fit_lambda_saccade_covariate.R
# Requires lme4 and lmerTest. Reported output: data/revision/lambda_saccade_covariate_lmer_output.txt
suppressMessages({library(lme4); library(lmerTest)})
d <- read.csv("data/revision/lambda_scrolling_watching_saccade.csv")
d$condition <- factor(d$condition, levels=c("Watching","Scrolling"))
d$subject <- factor(d$subject); d$image_id <- factor(d$image_id)
d$sacc_c <- scale(d$sacc_amp_deg, center=TRUE, scale=FALSE)  # center saccade amplitude
m0 <- lmer(amplitude_uv ~ condition + (1|subject) + (1|image_id), data=d, REML=FALSE)
m1 <- lmer(amplitude_uv ~ condition + sacc_c + (1|subject) + (1|image_id), data=d, REML=FALSE)
cat("==== m0: amplitude ~ condition ====\n"); print(summary(m0)$coefficients)
cat("\n==== m1: amplitude ~ condition + saccade_amplitude ====\n"); print(summary(m1)$coefficients)
cat("\nScrolling-vs-Watching coefficient (uV):\n")
cat(sprintf("  without saccade covariate: b = %.3f\n", fixef(m0)["conditionScrolling"]))
cat(sprintf("  with    saccade covariate: b = %.3f\n", fixef(m1)["conditionScrolling"]))
cat(sprintf("  saccade-amplitude slope:   b = %.4f uV/deg\n", fixef(m1)["sacc_c"]))
cat(sprintf("  attenuation of condition effect: %.0f%%\n",
    100*(1 - fixef(m1)["conditionScrolling"]/fixef(m0)["conditionScrolling"])))
cat("\nLRT condition in m1 (does condition still matter after saccade?):\n")
m1n <- lmer(amplitude_uv ~ sacc_c + (1|subject) + (1|image_id), data=d, REML=FALSE)
print(anova(m1n, m1))
