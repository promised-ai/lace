import matplotlib
from plotly import io
import polars as pl
import os

# Disable logging that might interfere with tests
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYLACE_EXAMPLES_QUIET"] = "1"

# Disable plot outputs
matplotlib.use("Agg")  # Use the "Agg" (non-interactive) backend
io.renderers.render_on_display = False
io.renderers.default = "json"

# Make polar's tables consistent
pl.Config(tbl_rows=8)
