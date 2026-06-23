# Codex Runtime Engineer Role

Do not begin with kernels. Begin with format inspection and a faithful reference loader.

Runtime phases:

1. Read manifest + checkpoint metadata.
2. Verify section shapes and architecture config.
3. Implement Python export to manifest layout.
4. Implement C++ loader stub.
5. Implement CPU reference inference for a tiny model.
6. Only then optimise low-bit basis matmul.
