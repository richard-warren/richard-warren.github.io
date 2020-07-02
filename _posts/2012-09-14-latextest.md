---
title: 'latextest'
date: 2012-09-14
permalink: /blog/latextest/
tags:
  - reinforcement learning
  - machine learning
read_time: false
---


{: class="table-of-content"}
* TOC
{:toc}

A couple of exciting news in Artificial Intelligence (AI) has just happened in recent years.  AlphaGo defeated the best professional human player in the game of Go. Very soon the extended algorithm AlphaGo Zero beat AlphaGo by 100-0 without supervised learning on human knowledge. Top professional game players lost to the bot developed by OpenAI on DOTA2 1v1 competition. After knowing these, it is pretty hard not to be curious about the magic behind these algorithms --- Reinforcement Learning (RL). I'm writing this post to briefly go over the field. We will first introduce several fundamental concepts and then dive into classic approaches to solving RL problems. Hopefully, this post could be a good starting point for newbies, bridging the future study on the cutting-edge research.


## What is Reinforcement Learning?

Say, we have an agent in an unknown environment and this agent can obtain some rewards by interacting with the environment. The agent ought to take actions so as to maximize cumulative rewards. In reality, the scenario could be a bot playing a game to achieve high scores, or a robot trying to complete physical tasks with physical items; and not just limited to these.


![Illustration of a reinforcement learning problem]({{ '/assets/images/RL_illustration.png' | relative_url }})
{: style="width: 70%;" class="center"}
*Fig. 1. An agent interacts with the environment, trying to take smart actions to maximize cumulative rewards.*


The goal of Reinforcement Learning (RL) is to learn a good strategy for the agent from experimental trials and relative simple feedback received. With the optimal strategy, the agent is capable to actively adapt to the environment to maximize future rewards.


### Key Concepts

Now Let's formally define a set of key concepts in RL.

The agent is acting in an **environment**. How the environment reacts to certain actions is defined by a **model** which we may or may not know. The agent can stay in one of many **states** ($$s \in \mathcal{S}$$) of the environment, and choose to take one of many **actions** ($$a \in \mathcal{A}$$) to switch from one state to another. Which state the agent will arrive in is decided by transition probabilities between states ($$P$$). Once an action is taken, the environment delivers a **reward** ($$r \in \mathcal{R}$$) as feedback.

The model defines the reward function and transition probabilities. We may or may not know how the model works and this differentiate two circumstances:
- **Know the model**: planning with perfect information; do model-based RL. When we fully know the environment, we can find the optimal solution by [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP). Do you still remember "longest increasing subsequence" or "traveling salesmen problem" from your Algorithms 101 class? LOL. This is not the focus of this post though.
- **Does not know the model**: learning with incomplete information; do model-free RL or try to learn the model explicitly as part of the algorithm. Most of the following content serves the scenarios when the model is unknown.

The agent's **policy** ($$\pi(s)$$) provides the guideline on what is the optimal action to take in a certain state with <span style="color: #e01f1f;">**the goal to maximize the total rewards**</span>. Each state is associated with a **value** function ($$V(s)$$) predicting the expected amount of future rewards we are able to receive in this state. In other words, the value function quantifies how good a state is. Both policy and value functions are what we try to learn in reinforcement learning.


![Categorization of RL Algorithms]({{ '/assets/images/RL_algorithm_categorization.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. Summary of approaches in RL based on whether we want to model the value, policy, or the environment. (Image source: reproduced from David Silver's RL course [lecture 1](https://youtu.be/2pWv7GOvuf0).)*


The interaction between the agent and the environment involves a sequence of actions and observed rewards in time, $$t=1, 2, \dots, T$$. During the process, the agent accumulates the knowledge about the environment, learns the optimal policy, and makes decisions on which action to take next so as to efficiently learn the best policy. Let's label the state, action, and reward at time step t as $$S_t$$, $$A_t$$, and $$R_t$$, respectively. Thus the interaction sequence is fully described by:

$$
S_1, A_1, R_2, S_2, A_2, \dots, S_T
$$

$$S_T$$ is the terminal state.


Terms you will encounter a lot when diving into different categories of RL algorithms:
- **Model-based**: Rely on the model of the environment; either the model is known or the algorithm learns it explicitly.
- **Model-free**: No dependency on the model during learning.
- **On-policy**: Use the deterministic outcomes or samples from the target policy to train the algorithm.
- **Off-policy**: Training on a distribution of transitions or episodes produced by a different behavior policy rather than that produced by the target policy.


#### Model: Transition and Reward

The model is a descriptor of the environment. With the model, we can learn or infer how the environment would interact with and provide feedback to the agent. The model has two major parts, transition probability function $$P$$ and reward function $$R$$.

Let's say when we are in state s, we decide to take action a to arrive in the next state s' and obtain reward r. This is known as one **episode**, represented by a tuple (s, a, s', r).

The transition function P records the probability of transitioning from state s to s' after taking action a while obtaining reward r. We use $$\mathbb{P}$$ as a symbol of "probability".

$$
P(s', r \vert s, a)  = \mathbb{P} [S_{t+1} = s', R_{t+1} = r \vert S_t = s, A_t = a]
$$

Thus the state-transition function can be defined as a function of $$P(s', r \vert s, a)$$:

$$
P_{ss'}^a = P(s' \vert s, a)  = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} P(s', r \vert s, a)
$$

The reward function R predicts the next reward triggered by one action:

$$
R(s, a) = \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} P(s', r \vert s, a)
$$


#### Policy

Policy, as the agent's behavior function $$\pi$$, tells us which action to take in state s. It is a mapping from state s to action a and can be either deterministic or stochastic:
- Deterministic: $$\pi(s) = a$$.
- Stochastic: $$\pi(a \vert s) = \mathbb{P}_\pi [A=a \vert S=s]$$.


#### Value Function

Value function measures the goodness of a state or how rewarding a state or an action is by a prediction of future reward. The future reward, also known as **return**, is a total sum of discounted rewards going forward. Let's compute the return $$G_t$$ starting from time t:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The discounting factor $$\gamma \in [0, 1]$$ penalize the rewards in the future, because:
- The future rewards may have higher uncertainty; i.e. stock market.
- The future rewards do not provide immediate benefits; i.e. As human beings, we might prefer to have fun today rather than 5 years later ;).
- Discounting provides mathematical convenience; i.e., we don't need to track future steps forever to compute return.
- We don't need to worry about the infinite loops in the state transition graph.

The **state-value** of a state s is the expected return if we are in this state at time t, $$S_t = s$$:

$$
V_{\pi}(s) = \mathbb{E}_{\pi}[G_t \vert S_t = s]
$$

Similarly, we define the **action-value** ("Q-value"; Q as "Quality" I believe?) of a state-action pair as:

$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \vert S_t = s, A_t = a]
$$

Additionally, since we follow the target policy $$\pi$$, we can make use of the probility distribution over possible actions and the Q-values to recover the state-value:

$$
V_{\pi}(s) = \sum_{a \in \mathcal{A}} Q_{\pi}(s, a) \pi(a \vert s)
$$

The difference between action-value and state-value is the action **advantage** function ("A-value"):

$$
A_{\pi}(s, a) = Q_{\pi}(s, a) - V_{\pi}(s)
$$


#### Optimal Value and Policy

The optimal value function produces the maximum return:

$$
V_{*}(s) = \max_{\pi} V_{\pi}(s),
Q_{*}(s, a) = \max_{\pi} Q_{\pi}(s, a)
$$

The optimal policy achieves optimal value functions:

$$
\pi_{*} = \arg\max_{\pi} V_{\pi}(s),
\pi_{*} = \arg\max_{\pi} Q_{\pi}(s, a)
$$

And of course, we have $$V_{\pi_{*}}(s)=V_{*}(s)$$ and $$Q_{\pi_{*}}(s, a) = Q_{*}(s, a)$$.


### Markov Decision Processes

In more formal terms, almost all the RL problems can be framed as **Markov Decision Processes** (MDPs). All states in MDP has "Markov" property, referring to the fact that the future only depends on the current state, not the history:

$$
\mathbb{P}[ S_{t+1} \vert S_t ] = \mathbb{P} [S_{t+1} \vert S_1, \dots, S_t]
$$

Or in other words, the future and the past are **conditionally independent** given the present, as the current state encapsulates all the statistics we need to decide the future.


![Agent-Environment Interaction in MDP]({{ '/assets/images/agent_environment_MDP.png' | relative_url }})
{: style="width: 60%;" class="center"}
*Fig. 3. The agent-environment interaction in a Markov decision process. (Image source: Sec. 3.1 Sutton & Barto (2017).)*


A Markov deicison process consists of five elements $$\mathcal{M} = <\mathcal{S}, \mathcal{A}, P, R, \gamma>$$, where the symbols carry the same meanings as key conceps in the [previsous](#key-concepts) section, well aligned with RL problem settings:
- $$\mathcal{S}$$ - a set of states;
- $$\mathcal{A}$$ - a set of actions;
- $$P$$ - transition probability function;
- $$R$$ - reward function;
- $$\gamma$$ - discounting factor for future rewards.
In an unknown environment, we do not have perfect knowledge about $$P$$ and $$R$$.


![MDP example]({{ '/assets/images/mdp_example.jpg' | relative_url }})
{: class="center"}
*Fig. 4. A fun example of Markov decision process: a typical work day. (Image source: [randomant.net/reinforcement-learning-concepts](https://randomant.net/reinforcement-learning-concepts/))*


### Bellman Equations

Bellman equations refer to a set of equations that decompose the value function into the immediate reward plus the discounted future values.

$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$

Similarly for Q-value,

$$
Q(s) = \mathbb{E} [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \vert S_t = s, A_t = a]
$$
