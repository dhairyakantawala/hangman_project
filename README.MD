# Hangman Project: Mathematical Approaches to Word Prediction

## Overview
This project implements advanced mathematical and machine learning approaches to solve the classic Hangman game. The system uses probability theory, deep learning, and reinforcement learning to predict missing letters in words.

## Mathematical Framework

### 1. Probability Distribution Modeling
We model letter distributions using conditional probability:

$$P(l_i | w_{1:i-1}, w_{i+1:n}) = \frac{P(w_{1:i-1}, l_i, w_{i+1:n})}{P(w_{1:i-1}, w_{i+1:n})}$$

Where:
- $l_i$ represents a letter at position $i$
- $w_{1:i-1}$ represents known letters before position $i$
- $w_{i+1:n}$ represents known letters after position $i$

### 2. Neural Network Architecture

#### 2.1 BiLSTM Implementation
The bidirectional LSTM model processes sequences in both directions:

$$\vec{h}_t = \text{LSTM}_{\text{forward}}(x_t, \vec{h}_{t-1})$$
$$\cev{h}_t = \text{LSTM}_{\text{backward}}(x_t, \cev{h}_{t+1})$$
$$h_t = [\vec{h}_t; \cev{h}_t]$$

Output probability distribution:
$$P(y|x) = \text{softmax}(W \cdot h_t + b)$$

#### 2.2 Attention Mechanism
Implemented using the equation:

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_x} \exp(e_{tk})}$$

Where $e_{tj} = a(s_{t-1}, h_j)$ is the alignment model scoring the match between input at position $j$ and output at position $t$.

Context vector:
$$c_t = \sum_{j=1}^{T_x} \alpha_{tj} h_j$$

### 3. Reinforcement Learning Approach

#### 3.1 Q-Learning Framework
Q-learning update rule:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

Where:
- $s_t$ is the current state (partially filled word)
- $a_t$ is the action (guessing a letter)
- $r_t$ is the immediate reward
- $\gamma$ is the discount factor
- $\alpha$ is the learning rate

#### 3.2 Neural Network Q-Function Approximation
The Q-function is approximated using a neural network:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

Loss function:
$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$$

## Implementation Steps

1. **Data Preparation**
   - Character-level encoding using one-hot vectors
   - Mathematical formulation: $x_i \in \{0,1\}^{|V|}$ where $|V|$ is vocabulary size

2. **Model Development**
   - BiLSTM with embedding dimension $d = 128$
   - Hidden state dimension $h = 512$
   - Multi-layer LSTM with dropout $p = 0.5$

3. **Training Process**
   - Cross-entropy loss: $L = -\sum_i y_i \log(\hat{y}_i)$
   - AdamW optimizer with learning rate $\alpha = 5 \times 10^{-4}$
   - Weight decay $\lambda = 10^{-2}$

4. **Reinforcement Learning**
   - State representation: $s_t = [k_t, i_t, w_t]$ where:
     - $k_t$ = known letter vector
     - $i_t$ = incorrect letter vector
     - $w_t$ = word state vector
   - Action space: $\mathcal{A} = \{a, b, c, ..., z\}$
   - Reward function: $r(s, a, s') = \begin{cases} 
      50 & \text{if word completed} \\
      -20 & \text{if game lost} \\
      2 & \text{if correct guess} \\
      -3 & \text{if incorrect guess}
      \end{cases}$

## Evaluation Metrics

1. **Letter Prediction Accuracy**
   $$\text{Accuracy} = \frac{\text{Correct letter predictions}}{\text{Total predictions}}$$

2. **Game Completion Rate**
   $$\text{Completion Rate} = \frac{\text{Games won}}{\text{Total games}}$$

3. **Average Guesses per Game**
   $$\text{Avg. Guesses} = \frac{\sum_{i=1}^{n} \text{Guesses in game}_i}{n}$$

## Results

The BiLSTM model with attention achieved:
- Training loss convergence to $0.35$ after 8 epochs
- Letter prediction accuracy of $76\%$ on test set
- Game completion rate of $65\%$ when using greedy selection

The reinforcement learning approach demonstrated:
- Improved performance with training
- Adaptive strategy development
- Superior performance in specific word classes

## Future Work

1. Incorporating transformer-based architectures with self-attention:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

2. Combining statistical and neural approaches using ensemble methods:
   $$P(l|w) = \alpha P_{statistical}(l|w) + (1-\alpha)P_{neural}(l|w)$$

3. Implementing adversarial training to improve robustness against challenging word patterns.

---