# 🧠 Perceptron Learning Visualizer

An interactive website that demonstrates how a **Perceptron works from scratch**. The project visually explains how a perceptron processes inputs, calculates an output, and learns by updating its weights during training.

## 📌 About the Project

A perceptron is one of the simplest artificial neural network models used for binary classification. This project aims to make its working process easier to understand through a visual and interactive demonstration.

Instead of just studying the mathematical equations, users can observe how inputs, weights, bias, and the activation function work together to produce predictions and how the model adjusts its parameters to learn from errors.

## ✨ Features

* **Input Processing:** Understand how input values are passed into a perceptron.
* **Weights and Bias:** Explore how weights and bias influence the model's output.
* **Activation Function:** See how the weighted sum is converted into a prediction.
* **Learning Process:** Understand how weights are updated during training.
* **Prediction Visualization:** Observe how the perceptron classifies input data.
* **Educational Demonstration:** Learn the fundamentals of neural networks through visual interaction.

## ⚙️ How a Perceptron Works

A perceptron follows these basic steps:

1. Receive input values.
2. Multiply each input by its corresponding weight.
3. Add the weighted inputs and bias.
4. Apply an activation function to generate a prediction.
5. Compare the prediction with the expected output.
6. Update the weights and bias when the prediction is incorrect.
7. Repeat the process during training.

### Mathematical Representation

**Weighted Sum**

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

**Step Activation Function**

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

Where:

* \(x_i\) = Input values
* \(w_i\) = Weights
* \(b\) = Bias
* \(z\) = Weighted sum
* \(\hat{y}\) = Predicted output

## 🎯 Project Objective

The main objective is to make the internal working of a perceptron easier to understand by demonstrating its computations, predictions, and learning process visually.

This project builds a foundation for understanding more advanced concepts in artificial intelligence, machine learning, and deep learning.

## 🛠️ Technologies Used

Add the technologies used in your actual implementation, such as:

* HTML
* CSS
* JavaScript

## 🚀 Getting Started

1. Clone this repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   ```

2. Navigate to the project directory:

   ```bash
   cd YOUR_REPOSITORY_NAME
   ```

3. Open `index.html` in your browser.

   If the project uses a development server or framework, follow the corresponding setup instructions instead.

## 📚 Learning Outcomes

* Understanding the basic architecture of a perceptron.
* Learning how weights and bias affect predictions.
* Understanding the role of activation functions.
* Exploring the relationship between prediction errors and learning.
* Building a foundation in neural networks through practical visualization.

## 🔮 Future Improvements

* Add more interactive training examples.
* Visualize changes in weights and bias during training.
* Demonstrate linearly separable datasets such as AND and OR gates.
* Add a decision-boundary visualization.
* Display training accuracy and prediction errors.

## 🤝 Contributing

Suggestions and improvements that make the perceptron learning process easier to understand are welcome.

## 📄 License

Add a suitable open-source license if you intend to allow others to reuse or modify this project.
