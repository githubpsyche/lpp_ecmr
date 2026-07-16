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
category_colours <- c(Negative = "#FF0000", Neutral = "#000000")

diagnostics <- read.csv(summary_path, stringsAsFactors = FALSE, check.names = FALSE)
plot_data <- diagnostics[diagnostics$source_id %in% figure_ids, ]
if (length(unique(plot_data$source_id)) != length(figure_ids)) {
  stop("The diagnostic summary does not contain every selected figure source")
}
plot_data$model_y <- unname(model_positions[plot_data$source_id])
plot_data$category <- factor(plot_data$category, levels = c("Negative", "Neutral"))

base_theme <- theme_classic(base_family = "Helvetica", base_size = 9) +
  theme(
    axis.line = element_line(colour = "#222222", linewidth = 0.35),
    axis.text = element_text(colour = "#111111", size = 7.5),
    axis.text.y = element_text(hjust = 1, margin = margin(0, 5, 0, 0)),
    axis.ticks = element_line(colour = "#222222", linewidth = 0.3),
    axis.ticks.length = unit(2.5, "pt"),
    legend.position = "top",
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.key.width = unit(13, "pt"),
    legend.key.height = unit(7, "pt"),
    legend.spacing.x = unit(5, "pt"),
    legend.margin = margin(0, 0, 3, 0),
    strip.background = element_blank(),
    strip.text.x = element_text(
      colour = "#111111",
      face = "bold",
      size = 9,
      hjust = 0.5,
      lineheight = 0.95,
      margin = margin(0, 0, 6, 0)
    ),
    plot.margin = margin(5, 7, 4, 6)
  )

recall_data <- plot_data[plot_data$metric == "recall_rate", ]
if (nrow(recall_data) != length(figure_ids) * 2) {
  stop(sprintf("Expected %d recall-rate rows, found %d", length(figure_ids) * 2, nrow(recall_data)))
}
recall_data$bar_y <- recall_data$model_y + ifelse(
  recall_data$category == "Negative",
  0.17,
  -0.17
)

negative_recall <- recall_data[recall_data$category == "Negative", c("source_id", "mean")]
neutral_recall <- recall_data[recall_data$category == "Neutral", c("source_id", "mean")]
names(negative_recall)[2] <- "negative_mean"
names(neutral_recall)[2] <- "neutral_mean"
recall_gaps <- merge(negative_recall, neutral_recall, by = "source_id")
recall_gaps$model_y <- unname(model_positions[recall_gaps$source_id])
recall_gaps$label <- sprintf("%+.2f", recall_gaps$negative_mean - recall_gaps$neutral_mean)

recall_figure <- ggplot(
  recall_data,
  aes(x = mean, y = bar_y, fill = category, colour = category)
) +
  geom_col(width = 0.28, linewidth = 0.35, orientation = "y") +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    width = 0.08,
    linewidth = 0.45,
    orientation = "y",
    show.legend = FALSE
  ) +
  geom_text(
    data = recall_gaps,
    aes(x = 0.695, y = model_y, label = label),
    inherit.aes = FALSE,
    hjust = 1,
    size = 2.55,
    colour = "#111111"
  ) +
  annotate(
    "text",
    x = 0.695,
    y = 5.52,
    label = "Negative - Neutral",
    hjust = 1,
    size = 2.55,
    fontface = "bold",
    colour = "#111111"
  ) +
  scale_x_continuous(
    breaks = seq(0, 0.6, by = 0.2),
    labels = label_number(accuracy = 0.1),
    limits = c(0, 0.71),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = unname(model_positions[figure_ids]),
    labels = unname(model_labels[figure_ids]),
    limits = c(0.55, 5.66),
    expand = expansion(add = 0)
  ) +
  scale_fill_manual(values = category_colours, name = NULL) +
  scale_colour_manual(values = category_colours, name = NULL) +
  guides(
    fill = guide_legend(override.aes = list(colour = category_colours, linewidth = 0.35)),
    colour = "none"
  ) +
  labs(x = "Recall rate", y = NULL) +
  base_theme

lpp_data <- plot_data[plot_data$metric == "within_list_centered_early_lpp", ]
if (nrow(lpp_data) != length(figure_ids) * 4) {
  stop(sprintf("Expected %d Early-LPP rows, found %d", length(figure_ids) * 4, nrow(lpp_data)))
}
lpp_data$recall_label <- ifelse(
  lpp_data$recall_status == "Remembered",
  "Recalled",
  "Unrecalled"
)
lpp_data$status_key <- factor(
  paste(lpp_data$category, lpp_data$recall_label, sep = "_"),
  levels = c(
    "Negative_Recalled",
    "Negative_Unrecalled",
    "Neutral_Recalled",
    "Neutral_Unrecalled"
  ),
  labels = c(
    "Negative: Recalled",
    "Negative: Unrecalled",
    "Neutral: Recalled",
    "Neutral: Unrecalled"
  )
)
lpp_offsets <- c(
  "Negative: Recalled" = 0.27,
  "Negative: Unrecalled" = 0.09,
  "Neutral: Recalled" = -0.09,
  "Neutral: Unrecalled" = -0.27
)
lpp_data$bar_y <- lpp_data$model_y + unname(lpp_offsets[lpp_data$status_key])
lpp_fills <- c(
  "Negative: Recalled" = "#FF0000",
  "Negative: Unrecalled" = "#FFFFFF",
  "Neutral: Recalled" = "#000000",
  "Neutral: Unrecalled" = "#FFFFFF"
)
lpp_colours <- c(
  "Negative: Recalled" = "#FF0000",
  "Negative: Unrecalled" = "#FF0000",
  "Neutral: Recalled" = "#000000",
  "Neutral: Unrecalled" = "#000000"
)

lpp_figure <- ggplot(
  lpp_data,
  aes(x = mean, y = bar_y, fill = status_key, colour = status_key)
) +
  geom_vline(xintercept = 0, colour = "#555555", linewidth = 0.35) +
  geom_col(width = 0.15, linewidth = 0.55, orientation = "y") +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    width = 0.045,
    linewidth = 0.45,
    orientation = "y",
    show.legend = FALSE
  ) +
  scale_x_continuous(
    breaks = c(-0.2, 0, 0.3, 0.6),
    labels = label_number(accuracy = 0.1),
    limits = c(-0.27, 0.64),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = unname(model_positions[figure_ids]),
    labels = unname(model_labels[figure_ids]),
    limits = c(0.55, 5.45),
    expand = expansion(add = 0)
  ) +
  scale_fill_manual(values = lpp_fills, name = NULL) +
  scale_colour_manual(values = lpp_colours, name = NULL) +
  guides(
    fill = guide_legend(
      nrow = 2,
      byrow = TRUE,
      override.aes = list(
        colour = unname(lpp_colours),
        linewidth = 0.55
      )
    ),
    colour = "none"
  ) +
  labs(x = "Early LPP", y = NULL) +
  base_theme +
  theme(
    legend.position = "top",
    legend.key.width = unit(13, "pt"),
    legend.spacing.x = unit(7, "pt")
  )

save_figure <- function(plot, stem, width, height) {
  output_stem <- file.path(package_dir, stem)
  ggsave(
    paste0(output_stem, ".pdf"),
    plot,
    device = pdf,
    width = width,
    height = height,
    units = "in",
    family = "Helvetica",
    useDingbats = FALSE
  )
  png(
    paste0(output_stem, ".png"),
    width = width,
    height = height,
    units = "in",
    res = 300,
    type = "quartz",
    family = "Helvetica",
    bg = "white"
  )
  print(plot)
  dev.off()

  svg_status <- system2(
    "pdftocairo",
    c("-svg", paste0(output_stem, ".pdf"), paste0(output_stem, ".svg"))
  )
  if (!identical(svg_status, 0L)) {
    stop(sprintf("pdftocairo failed to convert %s to SVG", stem))
  }
}

save_figure(recall_figure, "recall_rate_diagnostic_figure", 7.25, 4.25)
save_figure(lpp_figure, "early_lpp_diagnostic_figure", 7.25, 4.25)

message(
  sprintf(
    "Generated separate recall-rate and Early-LPP figures from %d selected summary rows",
    nrow(recall_data) + nrow(lpp_data)
  )
)
