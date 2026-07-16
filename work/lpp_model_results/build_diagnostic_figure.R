#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)])
package_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
summary_path <- file.path(package_dir, "diagnostic_summary.csv")

figure_ids <- c(
  "Observed",
  "CategoryOnly_eCMR_LPP_EmotionalOnly",
  "EEM_eCMR",
  "EEM_eCMR_LPP_General",
  "EEM_eCMR_LPP_EmotionalOnly"
)

model_labels <- c(
  Observed = "Observed data",
  CategoryOnly_eCMR_LPP_EmotionalOnly = "Emotion-dependent-LPP\nmodel",
  EEM_eCMR = "Preferential-learning\nmodel",
  EEM_eCMR_LPP_General = "Preferential learning +\nGeneral-LPP model",
  EEM_eCMR_LPP_EmotionalOnly = "Preferential learning +\nEmotion-dependent-LPP\nmodel"
)

diagnostics <- read.csv(summary_path, stringsAsFactors = FALSE, check.names = FALSE)
plot_data <- diagnostics[diagnostics$source_id %in% figure_ids, ]

if (length(unique(plot_data$source_id)) != length(figure_ids)) {
  stop("The diagnostic summary does not contain every selected figure source")
}

plot_data$model <- factor(
  unname(model_labels[plot_data$source_id]),
  levels = unname(model_labels[figure_ids])
)
plot_data$metric <- factor(
  ifelse(
    plot_data$metric == "recall_rate",
    "Recall rate",
    "Early LPP"
  ),
  levels = c("Recall rate", "Early LPP")
)
plot_data$category <- factor(
  plot_data$category,
  levels = c("Negative", "Neutral")
)
plot_data$memory_status <- ifelse(
  plot_data$metric == "Recall rate",
  "Overall",
  plot_data$recall_status
)
plot_data$memory_status <- factor(
  plot_data$memory_status,
  levels = c("Overall", "Remembered", "Forgotten")
)

expected_rows <- length(figure_ids) * (2 + 4)
if (nrow(plot_data) != expected_rows) {
  stop(sprintf("Expected %d plotting rows, found %d", expected_rows, nrow(plot_data)))
}

status_dodge <- position_dodge(width = 0.66)

diagnostic_figure <- ggplot(
  plot_data,
  aes(
    x = mean,
    y = category,
    fill = category,
    alpha = memory_status,
    linetype = memory_status,
    group = interaction(category, memory_status)
  )
) +
  geom_vline(xintercept = 0, colour = "#333333", linewidth = 0.35) +
  geom_col(
    position = status_dodge,
    width = 0.62,
    colour = "#333333",
    linewidth = 0.35,
    orientation = "y"
  ) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    position = status_dodge,
    width = 0.16,
    colour = "#1F1F1F",
    linewidth = 0.35,
    alpha = 1,
    linetype = "solid",
    orientation = "y",
    show.legend = FALSE
  ) +
  facet_grid(
    rows = vars(model),
    cols = vars(metric),
    scales = "free_x",
    switch = "y",
    axes = "all_x",
    axis.labels = "all_x"
  ) +
  scale_x_continuous(
    breaks = seq(-0.2, 0.6, by = 0.2),
    labels = label_number(accuracy = 0.1),
    expand = expansion(mult = c(0.04, 0.06))
  ) +
  scale_fill_manual(
    values = c(Negative = "#D95F5F", Neutral = "#666666"),
    guide = "none"
  ) +
  scale_alpha_manual(
    values = c(Overall = 1, Remembered = 1, Forgotten = 0.38),
    breaks = c("Remembered", "Forgotten"),
    name = NULL
  ) +
  scale_linetype_manual(
    values = c(Overall = "solid", Remembered = "solid", Forgotten = "dashed"),
    guide = "none"
  ) +
  guides(
    alpha = guide_legend(
      override.aes = list(
        fill = "#777777",
        colour = "#333333",
        linetype = c("solid", "dashed"),
        linewidth = 0.35
      )
    )
  ) +
  labs(x = NULL, y = NULL) +
  theme_classic(base_family = "Helvetica", base_size = 9) +
  theme(
    axis.line = element_line(colour = "#333333", linewidth = 0.35),
    axis.text = element_text(colour = "#1F1F1F", size = 7.5),
    axis.ticks = element_line(colour = "#333333", linewidth = 0.3),
    axis.ticks.length = unit(2.5, "pt"),
    legend.position = "top",
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.key.width = unit(18, "pt"),
    legend.key.height = unit(8, "pt"),
    legend.spacing.x = unit(4, "pt"),
    legend.margin = margin(0, 0, 2, 0),
    panel.spacing.x = unit(14, "pt"),
    panel.spacing.y = unit(8, "pt"),
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text.x = element_text(
      colour = "#1F1F1F",
      face = "bold",
      size = 9.5,
      hjust = 0.5,
      margin = margin(0, 0, 5, 0)
    ),
    strip.text.y.left = element_text(
      colour = "#1F1F1F",
      angle = 0,
      size = 8,
      hjust = 1,
      margin = margin(0, 5, 0, 0)
    ),
    plot.margin = margin(4, 6, 4, 6)
  )

output_stem <- file.path(package_dir, "diagnostic_figure")
ggsave(
  paste0(output_stem, ".pdf"),
  diagnostic_figure,
  device = pdf,
  width = 7.25,
  height = 7.1,
  units = "in",
  family = "Helvetica",
  useDingbats = FALSE
)
png(
  paste0(output_stem, ".png"),
  width = 7.25,
  height = 7.1,
  units = "in",
  res = 300,
  type = "quartz",
  family = "Helvetica",
  bg = "white"
)
print(diagnostic_figure)
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
    "Generated one %d x %d ggplot facet grid from %d long-form rows",
    length(figure_ids),
    2,
    nrow(plot_data)
  )
)
