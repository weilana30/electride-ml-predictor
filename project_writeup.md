# Electride ML Predictor: Initial Project Plan

## Project objective

The goal of this project is to develop a machine-learning workflow for predicting whether a crystalline compound can host an electride state: an electronic state in which one or more electrons occupy interstitial regions of the crystal rather than being associated primarily with atomic nuclei.

The proposed strategy separates two physical questions:

1. **Geometric suitability:** Is there a void with enough space and the right topology to host an interstitial electron?
2. **Chemical/electronic suitability:** Given the elements surrounding that void, is there a plausible mechanism for an electron to occupy and remain localized or delocalized in that region?

This naturally motivates a two-model system centered on crystal voids.

## Inspiration from the attached paper

The project is conceptually related to Ryan, Lengyel, and Shatruk, *Crystal Structure Prediction via Deep Learning*. That work represents crystallographically unique atomic sites using multiperspective fingerprints describing their local coordination topology, and then uses a neural network to learn which elements are compatible with particular structural environments.

The proposed adaptation is to replace the atomic site with a **candidate void site**. Instead of asking which element is likely to occupy a structural site, the model would ask whether a geometrically defined interstitial site is capable of hosting an electron and, in a second stage, whether the surrounding chemistry makes that electron plausible.

## 1. Dataset strategy

The dataset should not be limited to one structure per compound. Electride behavior can depend strongly on polymorphism, pressure, dimensionality, and structural relaxation. Relevant polymorphs and pressure-induced phases should therefore be retained when they represent genuinely distinct structures.

At the same time, the dataset should not initially attempt to include every available structure snapshot or rely on a highly complex equivariant model. A staged approach is preferable:

- Collect experimentally known electrides.
- Include computationally proposed or ambiguous electride candidates.
- Add chemically and structurally matched non-electrides.
- Include difficult negative examples with large voids or high ELF but no genuine interstitial electron.
- Deduplicate nearly identical structures.
- Preserve meaningful polymorphs and distinct pressure phases.

The train/validation/test split must be grouped by composition, chemical family, or structural prototype. Randomly splitting related polymorphs or chemically similar compounds would lead to overly optimistic performance and could allow the model to memorize composition or structure families.

Because confirmed electrides are likely to be rare, unverified materials should not automatically be treated as negatives. The dataset should distinguish among confirmed electrides, strong computational candidates, ambiguous cases, and ordinary non-electrides. Positive-unlabeled learning, uncertainty-aware classification, or active learning may be useful as the dataset grows.

## 2. Candidate voids

The structural model should operate on candidate voids rather than directly on whole crystals. Candidate voids can be generated using periodic Voronoi or Delaunay geometry, followed by filtering and clustering of nearby equivalent points.

Each void should have a local environment descriptor containing quantities such as:

- Void volume and characteristic radius.
- Distances to the nearest nuclei and several neighbor shells.
- Coordination number and coordination-polyhedron geometry.
- Solid-angle distribution around the void.
- Local symmetry and geometric anisotropy.
- Bottleneck sizes connecting the void to neighboring voids.
- Connectivity and dimensionality of the void network.
- Periodic-image relationships.

For a strict structure-only model, the input should contain lattice vectors and atomic coordinates but no element identities. A separate ablation model may include fixed geometric radii, such as covalent or ionic radii, to test how much apparent structural information is actually chemical information in disguise.

## 3. Target values

Electride behavior should not initially be represented by one binary label. ELF alone can identify ordinary bonding or lone-pair regions, while an electron near the Fermi level is not required for every electride. Metallic, semiconducting, insulating, localized, and extended electrides should not be conflated.

The preferred approach is a hierarchy of void-level and crystal-level targets.

### Void-level targets

For each candidate void, calculate or predict as many of the following as practical:

- Presence or absence of a non-nuclear attractor.
- ELF maximum near the void.
- Integrated electron population in an interstitial basin.
- Distance between the density maximum and the nearest nucleus.
- Charge-density Laplacian at the attractor.
- Localization strength and basin connectivity.
- Whether the state is isolated, chain-like, layered, or three-dimensionally connected.

### Crystal-level targets

At the crystal level, retain separate outputs for:

- Probability of electride character.
- Confidence that the charge is genuinely interstitial.
- Localization dimensionality: 0D, 1D, 2D, or 3D.
- Metallic, semiconducting, or insulating behavior.
- Magnetic or nonmagnetic character.
- Thermodynamic and dynamical stability.

A continuous electride score is preferable to an immediate yes/no target. A possible score could combine evidence for an interstitial attractor, interstitial charge population, localization, and a penalty for ordinary bonding environments. The exact definition should be based on the available DFT data rather than chosen arbitrarily.

The labeling scheme should retain confidence tiers:

- Confirmed electride.
- Strong computational electride.
- Possible or ambiguous electride.
- Non-electride control.

## 4. Model architecture

### Model 1: structure-only geometric model

The first model should predict whether each candidate void is spatially capable of supporting an interstitial state. It should then aggregate void-level predictions into a crystal-level score.

```text
Crystal structure
        |
Candidate void generation
        |
Void-centered geometric encoder
        |
Geometric score for each void
        |
Crystal-level pooling
        |
Geometric electride likelihood
```

This model could begin with handcrafted descriptors and a gradient-boosted tree or small neural network. A graph or equivariant model can be introduced after the dataset and labels are reliable.

### Model 2: chemical/electronic model

The second model should also be void-centered. A global composition-only model would learn useful chemical trends but would not know which particular cavity is relevant.

The preferred representation is a crystal graph containing both atom nodes and candidate-void nodes. Nearby atoms exchange messages with the void node, allowing the model to learn how elemental identity, coordination, electron count, and local geometry influence electron hosting.

Useful chemical features include:

- Valence electron count and stoichiometric electron balance.
- Electronegativity and electronegativity differences.
- Ionization energies and electron affinities.
- Atomic and ionic radii.
- Polarizability.
- Estimated oxidation-state patterns.
- Local coordination and bond-length statistics.
- Sublattice charge-transfer descriptors.
- Band gap or metallicity when available.

A composition-only gradient-boosting baseline should be retained so that the value of the structural and void-centered information can be measured.

## 5. Interaction between the models

Initially, the models should be used as a cascade:

```text
Structure-only model
        |
Ranks geometrically plausible voids
        |
Chemical/electronic model
        |
Ranks electron-hosting propensity
        |
DFT validation
```

This preserves a useful physical interpretation:

- The geometric model asks, “Is there room?”
- The chemical model asks, “Can the chemistry place and stabilize an electron there?”

Later, the two branches can be joined into a multitask model. The geometric embedding or geometric score can be passed to the chemical branch, while both branches retain separate outputs. This makes it possible to distinguish geometrically plausible but chemically unfavorable structures from chemically promising structures that lack a suitable cavity.

## 6. Recommended first implementation

The first version should be deliberately modest:

1. Generate periodic candidate voids.
2. Build a void-centered geometric descriptor.
3. Train a structure-only model against DFT-derived void descriptors.
4. Add elemental information and train a chemical/electronic model on the same voids.
5. Compare against composition-only and structure-only baselines.
6. Use composition- and prototype-held-out testing.
7. Add an equivariant graph model only after establishing reliable labels and splits.

The first useful targets are continuous void-level quantities such as ELF maximum, interstitial charge population, or non-nuclear-attractor likelihood. The binary electride decision can then be calibrated as a downstream classification task.

## Central design principle

The core workflow should preserve the physical decomposition:

```text
Geometry -> candidate void
Chemistry -> electron-hosting propensity
Electronic descriptors -> electride confidence
Stability -> practical material priority
```

The largest early risk is not the choice of neural-network architecture. It is constructing labels and evaluation splits that properly represent the ambiguity of electride definitions and prevent leakage between closely related structures.
