---
title: 'reading notes'
date: 2020-03-30
permalink: /blog/reading_notes/
tags:
  - reinforcement learning
  - deep learning
  - machine learning
---

# blog todo
- rl intro
- rl builds
- info theory
- gaussian process models
- eigenstuff
- probabilistic graphical models
- control theory intro
- math bits
  - taylor series expansion proof, with multivariate and vector form

# bishop
- 1.0
  - supervised, unsupervised, rl
  - train, test, generalization, training vs heuristics
- 1.1
  - data generating process ('model')
  - model noise
  - linear models (linear in weights, not necessarily weighted terms...)
  - error functions, eg sum of squared errors
  - model selection // overfitting
  - regularization, ridge regression

# topics / papers
- DQN
- DPG
- DDPG
- Memory-based control with recurrent neural networks
  - problem
    - model-free methods can fail when world partially observed, e.g. noisy or incomplete state info, seeing something that is then occluded, etc.
    - can have beliefs about world that are updated, but requires model
  - solution
    - incorporate memory!
    - modifies DPG and SVG with recurrent networks
- convex optimization
  - interior point method
  - programs
    - linear
    - quadratic
    - least squares
- https://www.youtube.com/watch?v=wEevt2a4SKI&t=12s
  - LQR: cost function is minimized subject to constraint that is the environment dynamics! xdot = Ax
  - why is x=0 optimal?
- control theory bootcamp
  - 5
    - x = Ax + Bu
    - u = -Kx
    - x = (A - BK)x
    - controllable means can put eigenvalues of (A-BK) wherever you want
    - therefore you can put the state wherever you want
    - usually A and B are given // they are dynamics of system and dynamics of control system
    - there are hard core tests of controllability (ctrb(A,B) in matlab)
    - controllability matrix -> [B AB A^2B ... A^{n-1} B]
    - if full ROW rank then controllable (n independent cols)
    - if nonlinear system is linearized, it is potentially still controllable even if test says its not controllable
  - 6
    - equivalences:
      - system is controllabel =
      - can place pole/evals wherever you want for A-BK =
      - you can get to any state you want! the 'reachable set' is all vectors in R^n such that a control signal u can get us there
      - `place` command in matlab gives us K thaht places eigenvals wherever we want them!
      - real systems are not linear! we can't get to any state in reality, so this is an approximation that is useful locally
    - 7
      - understanding controllability matrix as impulse response to u=1 and x=0 at time 1, then letting things ring out (in descrete time)
      - the columns of the ctrb matrix represent the directions in state space that are reached as the systems rings out
    - questions
      - explain the equivalence of convolution and control signal addition
    - 9-11
      - there are several tests for controllability that i don't have notes on... not sure how important these will be for my purposes
      - 12-13
        - pendulum on cart
        - need to understand lagrangian mechanics and how they are derived for this system
        - A and B matrices for system are obtained by linearizing are fixed points! this is first step in describing system
        - system is controllable because controllability matrix has rank n
        - `place(A,B,eigs)` gives K such that eigs of systems are at eigs
        - question: if we linearize around fixed points, system is only linear near those points, so how are these A and B matrices valid elsewhere?
        - q: is K applied to (x-x*), or just x???
        - eigs too small slow to correct, but too bib pushes the system out of the linear spot in the dynamics
      - 14, lqr
        - finds 'optimal' eigs wrt a cost function
        - have cost associated with states and control effort: J = \integral x^TQx + u^TRu dt
        - Q gives state cost, R gives effort cost
        - LQR is the control law that minimizes J
