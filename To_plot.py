
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
# Perceptron for AND Gate
class Perceptron:
    def __init__(self, lr=0.1):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn(1)
        self.lr = lr
    
    def predict(self, X):
        return np.heaviside(np.dot(X, self.weights) + self.bias, 0)
    
    def train(self, X, y, epochs=50):
        history = []
        for _ in range(epochs):
            error_sum = 0
            for xi, target in zip(X, y):
                pred = self.predict(xi[np.newaxis, :])
                error = target - pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                error_sum += abs(error)
            history.append((self.weights.copy(), self.bias.copy()))
            if error_sum == 0:
                break
        return history

def plot_perceptron_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate truth table

    perceptron = Perceptron(lr=0.1)
    history = perceptron.train(X, y)

    frames = []
    for i, (w, b) in enumerate(history):
        x_range = np.linspace(-0.5, 1.5, 100)
        y_range = -(w[0] * x_range + b) / w[1] if abs(w[1]) > 1e-10 else np.full_like(x_range, -b / w[1])

        frames.append(go.Frame(
            data=[
                go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, size=12)),
                go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color='red'))
            ],
            name=f'Epoch {i+1}'
        ))

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title='Perceptron Training on AND Gate',
            xaxis=dict(range=[-0.5, 1.5], title='X1'),
            yaxis=dict(range=[-0.5, 1.5], title='X2'),
            updatemenus=[
                {
                    'type': 'buttons',
                    'direction': 'right',
                    'x': 0.2, 'y': -0.3,
                    'buttons': [
                        {'label': 'Play', 'method': 'animate', 
                         'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': 'Pause', 'method': 'animate', 
                         'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
                    ]
                }
            ]
        ),
        frames=frames
    )

    return fig.to_html(include_plotlyjs='cdn')



class CumulativeGridPlot:
    def __init__(self, file_path = r"C:\\Users\\akash\\Mini Project\\perceptron_website\\Sun_Cloud_imgData.txt", intensity_factor=10.0):
        self.file_path = file_path
        self.intensity_factor = intensity_factor
        self.history_array = None
        self.vmin = None
        self.vmax = None
        self.x_vals, self.y_vals = np.meshgrid(np.arange(64), np.arange(64))
        self.x_vals = self.x_vals.flatten()
        self.y_vals = self.y_vals.flatten()
        self.load_and_process_data()
    
    def load_data_from_file(self):
        data = []
        with open(self.file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                features = np.array(list(map(int, parts[0].strip().split())))
                data.append(features)
        
        data = np.array(data)
        reshaped_data = data.reshape((data.shape[0], 64, 64))
        return reshaped_data

    def generate_history(self, arrays):
        num_steps = arrays.shape[0]
        cumulative_grid = np.zeros((64, 64))
        history = [cumulative_grid.copy()]
        
        for i in range(num_steps):
            operation = np.random.choice([1, -1])
            scaled_grid = arrays[i] * self.intensity_factor
            cumulative_grid += scaled_grid if operation == 1 else -scaled_grid
            history.append(cumulative_grid.copy())
        
        return np.array(history)
    
    def load_and_process_data(self):
        grids = self.load_data_from_file()
        self.history_array = self.generate_history(grids)
        self.vmin, self.vmax = np.min(self.history_array), np.max(self.history_array)

    def create_figure(self):
        scatter = go.Scatter(
            x=self.x_vals, 
            y=self.y_vals, 
            mode='markers',
            marker=dict(
                size=6,
                color=self.history_array[0].flatten(),  
                colorscale='RdBu',
                cmin=self.vmin,
                cmax=self.vmax
            )
        )

        fig = go.Figure(data=[scatter])

        frames = [
            go.Frame(
                data=[
                    go.Scatter(
                        x=self.x_vals, 
                        y=self.y_vals, 
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=self.history_array[i].flatten(),
                            colorscale='RdBu',
                            cmin=self.vmin,
                            cmax=self.vmax
                        )
                    )
                ],
                name=f"Step {i}"
            )
            for i in range(1, len(self.history_array))
        ]

        fig.update_layout(
            title="Cumulative Grid Visualization (Points)",
            height=500,  
            width=600,  
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                        "label": "▶ Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        "label": "⏸ Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.2,
                "xanchor": "right",
                "y": -0.15,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[f"Step {i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(self.history_array))
                ],
                "currentvalue": {"prefix": "Step: ", "font": {"size": 14}},
                "pad": {"b": 10, "t": 20},
            }]
        )

        fig.frames = frames
        return fig.to_html(include_plotlyjs='cdn')



def generate_data(n=100):
    X = np.random.rand(n, 2) * 2 - 1
    w_true = np.array([0.2, 0.5, -0.3])
    y = np.sign(w_true[0] + w_true[1] * X[:, 0] + w_true[2] * X[:, 1])
    return X, y

def perceptron(X, y, epochs=100):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_bias.shape[1])
    history = [w.copy()]
    misclassified_history = []

    for _ in range(epochs):
        misclassified = np.where(np.sign(X_bias @ w) != y)[0]
        misclassified_history.append(len(misclassified))
        if len(misclassified) == 0:
            break
        
        idx = np.random.choice(misclassified)
        w += y[idx] * X_bias[idx]
        history.append(w.copy())

    return history, misclassified_history

def plot_perceptron_separable():
    X, y = generate_data()
    weights, _ = perceptron(X, y)

    frames = []
    for i, w in enumerate(weights):
        x_vals = np.array([-1, 1])
        y_vals = -(w[0] + w[1] * x_vals) / w[2] if w[2] != 0 else np.zeros_like(x_vals)

        frame = go.Frame(
            data=[
                go.Scatter(x=X[y == 1][:, 0], y=X[y == 1][:, 1], mode='markers', marker=dict(color='blue')),
                go.Scatter(x=X[y == -1][:, 0], y=X[y == -1][:, 1], mode='markers', marker=dict(color='red')),
                go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='black', width=2))
            ],
            name=f'Iteration {i+1}'
        )
        frames.append(frame)

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title='Perceptron Learning Algorithm',
            xaxis=dict(range=[-1, 1], title='X1'),
            yaxis=dict(range=[-1, 1], title='X2'),
            updatemenus=[
                {
                    'type': 'buttons',
                    'direction': 'right',
                    'x': 0.2, 'y': -0.3,
                    'buttons': [
                        {'label': 'Play', 'method': 'animate', 
                         'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': 'Pause', 'method': 'animate', 
                         'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
                    ]
                }
            ]
        ),
        frames=frames
    )

    return fig.to_html(include_plotlyjs='cdn')




# Perceptron implementation
def perceptron(X, y, epochs=100):
    X_bias = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    w = np.zeros(X_bias.shape[1])
    history = [w.copy()]
    misclassified_history = []

    for epoch in range(epochs):
        misclassified = np.where(np.sign(X_bias @ w) != y)[0]
        misclassified_history.append(len(misclassified))
        if len(misclassified) == 0:
            break
        idx = np.random.choice(misclassified)
        w += y[idx] * X_bias[idx]
        history.append(w.copy())

    return history, misclassified_history

#  to create and return Plotly HTML
def create_animation(X, y, weights, misclassified_counts, title_prefix, x_label, y_label, x_range, y_range, x_idx, y_idx, boundary_func):
    frames = []
    total_points = len(X)


    x_boundary, y_boundary = boundary_func(weights[0])

    for i in range(len(weights)):
        x_boundary, y_boundary = boundary_func(weights[i])
        misclassified = misclassified_counts[i] if i < len(misclassified_counts) else 0
        
        frame = go.Frame(
            data=[
                go.Scatter(x=X[y == 1][:, x_idx], y=X[y == 1][:, y_idx], mode='markers', 
                           marker=dict(color='blue', size=10, line=dict(color='black', width=1)), name='Class +1'),
                go.Scatter(x=X[y == -1][:, x_idx], y=X[y == -1][:, y_idx], mode='markers', 
                           marker=dict(color='red', size=10, line=dict(color='black', width=1)), name='Class -1'),
                go.Scatter(x=x_boundary, y=y_boundary, mode='lines', line=dict(color='black', width=2), name='Decision Boundary')
            ],
            layout=go.Layout(title=f"{title_prefix} - Iteration {i+1}<br>Total Points: {total_points}, Misclassified: {misclassified}")
        )
        frames.append(frame)

    fig = go.Figure(
        data=[
            go.Scatter(x=X[y == 1][:, x_idx], y=X[y == 1][:, y_idx], mode='markers', 
                       marker=dict(color='blue', size=10, line=dict(color='black', width=1)), name='Class +1'),
            go.Scatter(x=X[y == -1][:, x_idx], y=X[y == -1][:, y_idx], mode='markers', 
                       marker=dict(color='red', size=10, line=dict(color='black', width=1)), name='Class -1'),
            go.Scatter(x=x_boundary, y=y_boundary, mode='lines', line=dict(color='black', width=2), name='Decision Boundary')
        ],
        layout=go.Layout(
            xaxis=dict(title=x_label, range=x_range),
            yaxis=dict(title=y_label, range=y_range),
            title=f"{title_prefix} - Iteration 1<br>Total Points: {total_points}, Misclassified: {misclassified_counts[0]}",
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "mode": "immediate"}])])],
            showlegend=True
        ),
        frames=frames
    )

    return fig.to_html(full_html=True)

# Function to generate and return HTML for the original data graph
def get_original_graph_html():
    np.random.seed(42)
    X = np.random.randn(200, 2) * 2
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    y = 2 * y - 1  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    weights, misclassified_counts = perceptron(X_scaled, y)

    def get_boundary(w):
        x1_vals = np.linspace(-3, 3, 100)
        w0, w1, w2 = w[0], w[1], w[2]
        if np.abs(w2) > 1e-6:  # Avoid division by zero
            y_vals = -(w0 + w1 * x1_vals) / w2
            return x1_vals, y_vals
        return x1_vals, np.zeros_like(x1_vals)

    return create_animation(X, y, weights, misclassified_counts, "Original Data", "X1", "X2", [-3, 3], [-3, 3], 0, 1, get_boundary)

def get_step1_graph_html():
    np.random.seed(42)
    X = np.random.randn(200, 2) * 2
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    y = 2 * y - 1  
    X_step1 = np.column_stack([X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_step1)

    weights, misclassified_counts = perceptron(X_scaled, y)

    def get_boundary(w):
        x1_vals = np.linspace(-3, 3, 100)
        w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
        if np.abs(w3) > 1e-6:  # Avoid division by zero
            y_vals = -(w0 + w1 * x1_vals) / w3
            y_vals = np.clip(y_vals, -9, 9) 
            return x1_vals, y_vals
        return x1_vals, np.zeros_like(x1_vals)

    return create_animation(X_step1, y, weights, misclassified_counts, "Step 1 (X1 vs X1*X2)", "X1", "X1 * X2", [-3, 3], [-9, 9], 0, 2, get_boundary)

def get_step2_graph_html():
    np.random.seed(42)
    X = np.random.randn(200, 2) * 2
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    y = 2 * y - 1  
    X_step1 = np.column_stack([X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])

    def relu(x):
        return np.maximum(0, x)

    X_step2 = np.column_stack([relu(X_step1[:, 0]), relu(X_step1[:, 1]), X_step1[:, 2]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_step2)

    weights, misclassified_counts = perceptron(X_scaled, y)

    def get_boundary(w):
        x1_vals = np.linspace(-1, 2, 100)
        w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
        if np.abs(w3) > 1e-6:  # Avoid division by zero
            y_vals = -(w0 + w1 * x1_vals) / w3
            y_vals = np.clip(y_vals, -3, 3)  # Clip to range
            return x1_vals, y_vals
        return x1_vals, np.zeros_like(x1_vals)

    return create_animation(X_step2, y, weights, misclassified_counts, "Step 2 (ReLU(X1) vs X1*X2)", "ReLU(X1)", "X1 * X2", [-1, 2], [-3, 3], 0, 2, get_boundary)



def load_data_from_file(file_path=r"C:\Users\akash\Downloads\frontend_final hai\frontend_final\frontend\data.txt"):
    data = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split by the comma to separate features and label
            parts = line.strip().split(',')
            
            # Convert the first part (features) into a list of integers
            features = np.array(list(map(int, parts[0].strip().split())))
            
            # The second part (after the comma) is the label, convert it to integer
            label = int(parts[1].strip())
            
            data.append(features)
            labels.append(label)
    
    # Convert data to a numpy array
    data = np.array(data)
    labels = np.array(labels)
    
    # Reshape each row (4096 elements) into a (64, 64) grid
    reshaped_data = data.reshape((data.shape[0], 64, 64))  # (200, 64, 64)
    
    return reshaped_data, labels

# Load the data


def plot_grids():
    file_path = r"C:\Users\akash\Downloads\frontend_final hai\frontend_final\frontend\data.txt"
    grids, y = load_data_from_file(file_path)
    # Ensure the input grid is in the correct shape
    assert grids.shape[0] == 200 and grids.shape[1] == 64 and grids.shape[2] == 64, "Input array must have shape (200, 64, 64)"

    # Create a figure
    fig = go.Figure()

    # Add all 200 grids as heatmap traces, initially only the first one is visible
    for i in range(grids.shape[0]):
        fig.add_trace(
            go.Heatmap(
                z=grids[i],  # 64x64 grid data
                colorscale='Gray',
                showscale=False,  # Hide color scale for cleaner look
                visible=(i == 0)  # Only the first grid is visible initially
            )
        )

    # Create steps for the slider
    steps = []
    for i in range(grids.shape[0]):
        step = dict(
            method="update",
            args=[{"visible": [False] * grids.shape[0]}, {"title": f"Grid {i+1}"}],
            label=f"{i+1}"
        )
        step["args"][0]["visible"][i] = True  # Make the current grid visible
        steps.append(step)

    # Add a slider to the figure
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Grid: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        title="Grid 1",
        width=600,
        height=600,
        sliders=sliders,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    # Return the figure as HTML for Flask
    return fig.to_html(include_plotlyjs='cdn')

# Example usage (for testing outside Flask)
# html_output = plot_grids(grids)
# with open("plot.html", "w") as f:
#     f.write(html_output)