---
title: 'probabilistic graphical models: intro'
date: 2020-03-30
permalink: /blog/probabilistic_graphical_models/
tags:
  - probability
  - machine learning
---

# topics
- representation
  - bayesian networks
    - factorizing a graph over a DAG
    - d-separation
    - active paths (common parent, common child (explaining away))
    - I-map (don't really get this)
  - markov random fields
    - factors, normalizing constants, 'energy landscape'
    - cliques
    - moralization // maybe unnecessary
    - markov blanket // maybe unnecessary
    - conditional random fields
- inference
  - general idea // exact vs approximate
  - marginal
  - map
  - variable elimination (see double sum to product of sums)
    - dynamic programming connection
    - ordering
  - belief propagation (junction tree algorithm generalization)
    - message passing (summarizing all info affecting parents)
    - sum-product message passing
    - max-product message passing
