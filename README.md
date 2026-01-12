# Cascade Modelling in Networked Systems

## Overview

This repository contains an exploratory research project investigating cascade dynamics in networked systems.  
The work is motivated by questions relevant to security operations and system resilience, but is intentionally simplified and pedagogical in nature.

The goal is understanding behaviour, not producing a deployment-ready model.

---

## Motivation

In complex systems, local failures can propagate non-linearly through network structure, leading to disproportionate downstream impact.  
In security and SOC contexts, similar dynamics may arise through misconfiguration, dependency chains or partial compromise.

This project explores such dynamics abstractly, using toy models to reason about structure, heterogeneity and hardening.

---

## Structure of the repository

- **Notebook 0 – Preface and vision**  
  Conceptual framing, assumptions and motivation for the work.

- **Notebook 1 – Cascades, criticality and why they matter**  
  An introduction to self-organising, critical systems.

Additional notebooks may extend or refine these ideas.

Please note that some notebooks are summaries for recruiters and are so-labelled in the filename. These notebooks accompany equivalent "workbooks" which are my more technical working documents.

---

## Design philosophy

- Prefer simple, inspectable models over realism
- Use synthetic systems to reason about behaviour
- Make assumptions explicit
- Treat this as exploratory research, not operational tooling

The notebooks are written to be readable by others who are new to cascade modelling (I am).

---

## Limitations & boundaries

- Models are highly simplified
- No real network data is used
- Results are illustrative rather than predictive
- The work is exploratory and educational

No claims are made about direct applicability to real-world systems without further validation.

---

## Ethics & use

This work is defensive and analytical in intent.  
It does not model exploitation, attack paths or adversarial tradecraft.

