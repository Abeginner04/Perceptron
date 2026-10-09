from flask import Flask, render_template
from To_plot import *

app = Flask(__name__)

# for Home Page
@app.route('/')
def index():
    return render_template('index.html')    

@app.route('/perceptron-and')
def perceptron_and():
    return plot_perceptron_and()

@app.route('/perceptron-separable')
def perceptron_separable():
    return plot_perceptron_separable()
    

# Generate Cumulative Grid Plot
@app.route('/cumulative-grid')
def cumulative_grid():
    grid_plot = CumulativeGridPlot("data.txt")  # Adjust file path as needed
    return grid_plot.create_figure()

@app.route('/sun_mon_present')
def sun_mon_present():  # Adjust file path as needed
    return plot_grids()

# Route to Original Data Plot
@app.route('/x_data')
def x_data():
    return get_original_graph_html()

# Step 1 Plot
@app.route('/x_data_t1')
def x_data_t1():
    return get_step1_graph_html()

# Step 2 Plot
@app.route('/x_data_t2')
def x_data_t2():
    return get_step2_graph_html()


if __name__ == '__main__':
    app.run(debug=True)
