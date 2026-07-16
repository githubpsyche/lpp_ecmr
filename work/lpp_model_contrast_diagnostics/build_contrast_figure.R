#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)])
package_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
summary_path <- file.path(package_dir, "contrast_summary.csv")

source_ids <- c(
  "Observed",
  "CategoryOnly_eCMR_LPP_EmotionalOnly",
  "EEM_eCMR",
  "EEM_eCMR_LPP_General",
  "EEM_eCMR_LPP_EmotionalOnly"
)
model_labels <- c(
  Observed = "Observed data",
  CategoryOnly_eCMR_LPP_EmotionalOnly = "Emotion-dependent-LPP\nmodel",
  EEM_eCMR = "Categorical-enhancement\nmodel",
  EEM_eCMR_LPP_General = "Categorical +\nGeneral-LPP model",
  EEM_eCMR_LPP_EmotionalOnly = "Categorical +\nEmotion-dependent-LPP\nmodel"
)
model_positions <- c(
  Observed = 5,
  CategoryOnly_eCMR_LPP_EmotionalOnly = 4,
  EEM_eCMR = 3,
  EEM_eCMR_LPP_General = 2,
  EEM_eCMR_LPP_EmotionalOnly = 1
)
diagnostic_ids <- c("recall_gap", "lpp_negative", "lpp_neutral")
diagnostic_labels <- c(
  recall_gap = "Recall rate\nNegative - Neutral",
  lpp_negative = "Early LPP\nRecalled - Unrecalled\nNegative",
  lpp_neutral = "Early LPP\nRecalled - Unrecalled\nNeutral"
)

plot_data <- read.csv(summary_path, stringsAsFactors = FALSE, check.names = FALSE)
if (nrow(plot_data) != length(source_ids) * length(diagnostic_ids)) {
  stop(sprintf("Unexpected contrast-summary row count: %d", nrow(plot_data)))
}
plot_data$model_y <- unname(model_positions[plot_data$source_id])
plot_data$diagnostic <- factor(
  plot_data$diagnostic_id,
  levels = diagnostic_ids,
  labels = unname(diagnostic_labels[diagnostic_ids])
)

anchor_data <- data.frame(
  model_y = rep(unname(model_positions["Observed"]), 6),
  diagnostic = factor(
    rep(unname(diagnostic_labels[diagnostic_ids]), each = 2),
    levels = levels(plot_data$diagnostic)
  ),
  estimate = c(-0.02, 0.16, -0.10, 0.60, -0.10, 0.60)
)

contrast_figure <- ggplot(
  plot_data,
  aes(x = estimate, y = model_y, fill = diagnostic)
) +
  geom_blank(
    data = anchor_data,
    aes(x = estimate, y = model_y),
    inherit.aes = FALSE
  ) +
  geom_vline(xintercept = 0, colour = "#333333", linewidth = 0.35) +
  geom_col(
    width = 0.62,
    colour = "#333333",
    linewidth = 0.35,
    orientation = "y"
  ) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    width = 0.16,
    colour = "#1F1F1F",
    linewidth = 0.35,
    orientation = "y",
    show.legend = FALSE
  ) +
  facet_grid(
    cols = vars(diagnostic),
    scales = "free_x",
    space = "fixed"
  ) +
  scale_x_continuous(
    breaks = pretty_breaks(n = 4),
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0.04, 0.06))
  ) +
  scale_y_continuous(
    breaks = unname(model_positions[source_ids]),
    labels = unname(model_labels[source_ids]),
    limits = c(0.5, 5.5),
    expand = expansion(add = 0)
  ) +
  scale_fill_manual(
    values = c(
      "Recall rate\nNegative - Neutral" = "#D95F5F",
      "Early LPP\nRecalled - Unrecalled\nNegative" = "#D95F5F",
      "Early LPP\nRecalled - Unrecalled\nNeutral" = "#666666"
    ),
    guide = "none"
  ) +
  labs(x = NULL, y = NULL) +
  theme_classic(base_family = "Helvetica", base_size = 9) +
  theme(
    axis.line = element_line(colour = "#333333", linewidth = 0.35),
    axis.text = element_text(colour = "#1F1F1F", size = 7.5),
    axis.text.y = element_text(hjust = 1, margin = margin(0, 4, 0, 0)),
    axis.ticks = element_line(colour = "#333333", linewidth = 0.3),
    axis.ticks.length = unit(2.5, "pt"),
    panel.spacing.x = unit(14, "pt"),
    strip.background = element_blank(),
    strip.text.x = element_text(
      colour = "#1F1F1F",
      face = "bold",
      size = 9,
      hjust = 0.5,
      lineheight = 0.95,
      margin = margin(0, 0, 6, 0)
    ),
    plot.margin = margin(6, 6, 4, 6)
  )

output_stem <- file.path(package_dir, "contrast_diagnostic_figure")
ggsave(
  paste0(output_stem, ".pdf"),
  contrast_figure,
  device = pdf,
  width = 7.25,
  height = 4.25,
  units = "in",
  family = "Helvetica",
  useDingbats = FALSE
)
png(
  paste0(output_stem, ".png"),
  width = 7.25,
  height = 4.25,
  units = "in",
  res = 300,
  type = "quartz",
  family = "Helvetica",
  bg = "white"
)
print(contrast_figure)
dev.off()

svg_status <- system2(
  "pdftocairo",
  c(
    "-svg",
    paste0(output_stem, ".pdf"),
    paste0(output_stem, ".svg")
  )
)
if (!identical(svg_status, 0L)) {
  stop("pdftocairo failed to convert the ggplot PDF to SVG")
}

message(
  sprintf(
    "Generated one 5-source x 3-contrast ggplot from %d long-form rows",
    nrow(plot_data)
  )
)
