function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
}

// Generate linearly separable data
function generateData(n = 100) {
    const X = [];
    const y = [];
    const w_true = [0.2, 0.5, -0.3];
    
    for (let i = 0; i < n; i++) {
        const x1 = Math.random() * 2 - 1;
        const x2 = Math.random() * 2 - 1;
        X.push([x1, x2]);
        const value = w_true[0] + w_true[1] * x1 + w_true[2] * x2;
        y.push(value >= 0 ? 1 : -1);
    }
    return { X, y };
}

// Perceptron Learning Algorithm
function perceptron(X, y, epochs = 100) {
    const X_bias = X.map(x => [1, ...x]);
    let w = [0, 0, 0];
    const history = [w.slice()];
    const misclassifiedHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
        const predictions = X_bias.map(x => Math.sign(x[0] * w[0] + x[1] * w[1] + x[2] * w[2]));
        const misclassified = predictions.map((p, i) => p !== y[i] ? i : -1).filter(i => i !== -1);
        
        misclassifiedHistory.push(misclassified.length);
        if (misclassified.length === 0) break;

        const idx = misclassified[Math.floor(Math.random() * misclassified.length)];
        w = w.map((wi, i) => wi + y[idx] * X_bias[idx][i]);
        history.push(w.slice());
    }
    return { history, misclassifiedHistory };
}

// Get decision boundary
function getDecisionBoundary(w) {
    const x_vals = [-1, 1];
    const y_vals = w[2] !== 0 ? x_vals.map(x => -(w[0] + w[1] * x) / w[2]) : [0, 0];
    return { x_vals, y_vals };
}

// Main demo functions
let data, weights, misclassifiedCounts;

function initializeDemo() {
    data = generateData();
    const result = perceptron(data.X, data.y);
    weights = result.history;
    misclassifiedCounts = result.misclassifiedHistory;
}

function createPlot() {
    const frames = weights.map((w, i) => {
        const { x_vals, y_vals } = getDecisionBoundary(w);
        return {
            name: `Iteration ${i + 1}`,
            data: [
                {
                    x: data.X.filter((_, i) => data.y[i] === 1).map(x => x[0]),
                    y: data.X.filter((_, i) => data.y[i] === 1).map(x => x[1]),
                    mode: 'markers',
                    marker: { color: 'blue' },
                    name: 'Class +1'
                },
                {
                    x: data.X.filter((_, i) => data.y[i] === -1).map(x => x[0]),
                    y: data.X.filter((_, i) => data.y[i] === -1).map(x => x[1]),
                    mode: 'markers',
                    marker: { color: 'red' },
                    name: 'Class -1'
                },
                {
                    x: x_vals,
                    y: y_vals,
                    mode: 'lines',
                    line: { color: 'black', width: 2 },
                    name: 'Decision Boundary'
                }
            ]
        };
    });

    const layout = {
        title: 'Perceptron Learning Algorithm',
        xaxis: { range: [-1, 1], title: 'X1' },
        yaxis: { range: [-1, 1], title: 'X2' },
        updatemenus: [{
            buttons: [
                { label: 'Play', method: 'animate', args: [null, { frame: { duration: 500, redraw: true }, fromcurrent: true }] },
                { label: 'Pause', method: 'animate', args: [[null], { frame: { duration: 0, redraw: true }, mode: 'immediate' }] }
            ],
            direction: 'left',
            pad: { r: 10, t: 87 },
            showactive: false,
            type: 'buttons',
            x: 0.1,
            xanchor: 'right',
            y: 0,
            yanchor: 'top'
        }],
        sliders: [{
            steps: frames.map(f => ({
                args: [[f.name], { frame: { duration: 500, redraw: true }, mode: 'immediate' }],
                label: f.name,
                method: 'animate'
            })),
            active: 0,
            yanchor: 'top',
            xanchor: 'left',
            currentvalue: { prefix: 'Iteration: ', font: { size: 20 } },
            pad: { b: 10, t: 50 },
            len: 0.9,
            x: 0.1,
            y: -0.2
        }]
    };

    Plotly.newPlot('plot-container', frames[0].data, layout, { responsive: true }).then(() => {
        Plotly.addFrames('plot-container', frames);
    });
}

function runDemo() {
    initializeDemo();
    createPlot();
}

function resetDemo() {
    Plotly.purge('plot-container');
    document.getElementById('plot-container').innerHTML = '<p>Click "Run Demo" to see the visualization</p>';
}

// Initialize on page load
window.onload = () => {
    resetDemo();
};