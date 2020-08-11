---
title: 'reading notes'
date: 2020-03-30
permalink: /blog/reading_notes/
tags:
  - reinforcement learning
  - deep learning
  - machine learning
---

# papers
- DQN
- DPG
- DDPG
- Memory-based control with recurrent neural networks
  - problem
    - model-free methods can fail when world partially observed, e.g. noisy or incomplete state info, seeing something that is then occluded, etc.
    - can have beliefs about world that are updated, but requires model
  - solution
    - incorporate memory!
