# Analyses

`lpp_ecmr/analyses` has three notebook roles:

- Source notebooks in `analyses/` are the maintained entrypoints for rendering and project-side analysis work.
- Template notebooks in `analyses/templates/` are reusable execution notebooks that source notebooks render with project-specific parameters.
- Generated notebooks in `analyses/rendered/` are rendered artifacts produced from the source notebooks and templates.

## Source notebooks

- `data_preparation.ipynb`: prepare the Talmi EEG dataset for model-fitting and downstream analysis.
- `render_reference_analyses.ipynb`: render categorized descriptive analyses and preview the resulting reference figures.
- `render_model_fitting_single_context.ipynb`: render subject-level fitting notebooks for the single-context model family.
- `render_model_fitting_full_ecmr.ipynb`: render subject-level fitting notebooks for the full eCMR model family.
- `render_model_fitting_group_level.ipynb`: render pooled/group-level fitting notebooks for the selected manuscript-scale models.
- `render_model_comparison_full_set.ipynb`: render the comparison notebook for the full enabled model set.
- `render_model_comparison_manuscript_set.ipynb`: render the comparison notebook for the manuscript-focused model set.
- `render_parameter_sensitivity.ipynb`: render parameter-shifting notebooks and preview their figures.

## Templates

Templates are reusable notebook bodies that the source notebooks parameterize. Current project-local template coverage includes categorized SPC/LPP analyses and cross-validation. Parameter shifting now renders from the central `jaxcmr` template.

## Rendered prefixes

- `cat_`: rendered reference-analysis notebooks.
- `fitting_`: subject-level fitting notebooks.
- `group_fitting_`: group-level fitting notebooks.
- `model_comparison_`: rendered model-comparison notebooks.
- `parameter_shifting_`: rendered parameter-sensitivity notebooks.
