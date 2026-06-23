# Glossary

## Stored unique weights

Weights actually saved in the model file/checkpoint.

## Effective materialized weights

Weights the model would have if every virtual layer's composed matrices were stored separately.

## Basis matrix

A shared learned matrix used as an ingredient to compose effective weights.

## Route coefficients

Tiny learned vectors that select/mix basis matrices for a virtual layer.

## Low-rank delta

A small correction matrix represented as `U @ V` where rank is much smaller than matrix width.

## Virtual layer

A computational depth step. It may reuse the same stored basis weights as other virtual layers.

## Block family

A reusable group of basis weights and parameters shared across multiple virtual layers.

## TCMDL

Future TinyCore Model file format for compositional low-bit models.
