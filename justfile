default:
    just --list

# Launch JupyterLab
lab:
    uv run jupyter lab

# Execute a notebook in-place (usage: just run lessons/stats/01_distributions.ipynb)
run notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace {{notebook}}

# Add a package (usage: just add cvxpy)
add package:
    uv add {{package}}

# Launch the MutationEngine GUI notebook (ipywidgets, runs in Jupyter)
mutate:
    uv run jupyter lab lessons/backtesting/03_mutation_engine.ipynb

# Launch the standalone Tkinter MutationEngine GUI (requires: sudo pacman -S tk)
gui:
    uv run python DrakonixBacktester/mutationengine/tkgui.py
