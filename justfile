# Launch JupyterLab
lab:
    uv run jupyter lab

# Execute a notebook in-place (usage: just run lessons/stats/01_distributions.ipynb)
run notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace {{notebook}}

# Add a package (usage: just add cvxpy)
add package:
    uv add {{package}}
