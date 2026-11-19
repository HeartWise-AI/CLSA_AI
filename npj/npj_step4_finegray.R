# npj Step 4: Fine-Gray (survival::finegray) and Aalen-Johansen 6-y cumulative incidence by predicted-risk decile, MHI.
suppressMessages(library(survival))
out <- commandArgs(TRUE)[1]; H <- 2190
res <- list(); cal <- list()
for (f in c("last_comp", "admin_comp")) {
  d <- read.csv(file.path(out, paste0("step4_tte_", f, ".csv")))
  d$ev <- factor(d$status, 0:2, c("censor", "mace", "death"))
  d$zA <- (d$lA - mean(d$lA)) / sd(d$lA); d$zS <- (d$lS - mean(d$lS)) / sd(d$lS)
  fg <- finegray(Surv(T, ev) ~ ., data = d[, c("T", "ev", "zA", "zS", "lA")], etype = "mace")
  for (z in c("zA", "zS")) {
    m <- coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", z)), weight = fgwt, data = fg)
    ci <- exp(confint(m)); res[[length(res) + 1]] <- data.frame(cohort = f, predictor = z, sHR_perSD = exp(coef(m)), lo = ci[1], hi = ci[2],
      cause_specific_HR = exp(coef(coxph(as.formula(paste("Surv(T, status == 1) ~", z)), data = d))))
  }
  d$dec <- cut(d$lA, quantile(d$lA, 0:10 / 10), include.lowest = TRUE, labels = 1:10)
  aj <- survfit(Surv(T, ev) ~ dec, data = d)
  s <- summary(aj, times = H, extend = TRUE)
  k <- which(s$states == "mace")
  cal[[f]] <- data.frame(cohort = f, decile = 1:10, n = as.vector(table(d$dec)), mean_pred_transported = tapply(plogis(d$lA), d$dec, mean),
                         observed_cif_6y = s$pstate[, k], lo = s$lower[, k], hi = s$upper[, k])
}
write.csv(do.call(rbind, res), file.path(out, "step4_mhi_finegray.csv"), row.names = FALSE)
write.csv(do.call(rbind, cal), file.path(out, "step4_mhi_aj_calibration.csv"), row.names = FALSE)
print(do.call(rbind, res)); print(do.call(rbind, cal))
