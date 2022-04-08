

```
python
import pandas as pd
from plotly import express as px

url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
penguins = penguins.dropna(subset = ["Body Mass (g)", "Sex"])
penguins["Species"] = penguins["Species"].str.split().str.get(0)
penguins = penguins[penguins["Sex"] != "."]

cols = ["Species", "Island", "Sex", "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
penguins = penguins[cols]
```

```
python
fig = px.scatter(data_frame = penguins, #dataset we want to visualize
                 title = "Culmen Size Distribution among Three Penguin Species", #figure title
                 x = "Culmen Length (mm)", #column from dataset to use for x axis
                 y = "Culmen Depth (mm)", #column from dataset to use for y axis
                 color = "Species", #column to use for color
                 size = "Body Mass (g)", #column to use for point size
                 symbol = "Island", #column to use for point symbol symbol
                 size_max = 8, # setting max point size
                 opacity = 0.7, #setting point opacity
                 hover_name = "Species", #column data to use as the title when hovering over data points
                 hover_data = ["Island", "Sex"], #column data to use when hovering over data points
                 width = 800, #width (in pixels) of the plot
                 height = 500, #height (in pixels) of the plot
                marginal_y = "violin", #specifying marginal plots gives supplementary summary statistics of x or y axes
                marginal_x = "violin")

# show the plot
fig.show()
```