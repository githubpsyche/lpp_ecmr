# Pure-list simulations from pooled fits

This work package reuses the 16 pooled parameter fits from
`work/lpp_model_comparison` and changes only the inputs used for forward
simulation. No model is refit here.

The 342 observed list templates are divided into:

- 171 pure-negative lists (`condition == 1` at all 20 positions); and
- 171 pure-neutral lists (`condition == 2` at all 20 positions).

The fixed assignment is recorded in `pure_list_design.csv`. It preserves the
original subject and list identifiers, assigns each of the 38 participants
four lists of one type and five of the other, and exactly balances the number
of imposed recall events between the two list types. Early-LPP values, study
positions, list lengths, and the observed recall-count masks are retained from
the original templates.

Because the fitted models exclude termination, the simulations condition on
the observed number recalled from each template. They can therefore compare
which items are recalled and the Early-LPP-by-recall diagnostic, but they do
not predict differences in total recall between pure-negative and pure-neutral
lists.

Each `fit_*.ipynb` notebook:

1. loads its copied `*_best_of_3.json` fit;
2. constructs the same pure-list simulation dataset from
   `pure_list_design.csv`;
3. runs 200 simulations per list template;
4. writes a model-specific HDF5 file; and
5. writes category-recall and Early-LPP-by-recall figures for the pure-list
   simulation.

`prepare.py` reproducibly copies and adapts the source notebooks and fit JSONs.
`execute_all.py` executes the prepared notebooks sequentially with the shared
workspace Jupyter kernel.
