# Cursor Agent Rules for RlVAE

These rules must be read and followed by all Cursor agents and contributors working in this repository. They ensure consistency, quality, and maintainability across all experiments, code, and documentation.

---

1. **Documentation First:** Always read all documentation files (.md and .cursor_context.md) at the start of a new chat or when context may have changed, before answering or making code changes.
2. **Legacy Code:** Do not use or modify legacy scripts unless explicitly discussed. If a legacy script seems relevant, always ask the user for permission and explain why referencing or modifying it could be beneficial. Prefer the modular pipeline and new architecture for all new work.
3. **Naming Conventions:** Enforce a consistent, descriptive naming scheme for all experiments, models, metrics, and outputs. If a naming pattern is not already established, propose one and confirm with the user before applying it broadly. Ensure all new files, folders, and WandB runs follow the agreed naming scheme.
4. **Visualization Defaults:** Default to an "intermediate" visualization level (not minimal, not full) for most experiments and sweeps. Use minimal visualizations only for large-scale sweeps or when explicitly requested for speed. Use full visualizations only for final runs or publication-quality outputs.
5. **Extensibility, Documentation, and Testing:** Every new model, sampler, visualization, or major feature must include comprehensive tests (unit and, if relevant, integration), documentation for usage/configuration/design, and an update to a "new_things" documentation file summarizing all new features/components since the last PR.
6. **WandB and File Outputs:** Prefer WandB-only mode for experiment logging and visualization. Avoid local file outputs unless explicitly requested by the user. Ensure all visualizations, metrics, and checkpoints are logged to WandB.
7. **Cluster/SLURM Usage:** No strict rule for cluster/SLURM usage: use local execution for small tests and SLURM/batch scripts for large-scale or long-running experiments. Use your judgment and ask the user if unsure.
8. **Testing Workflow:** Test new modules locally and in isolation before integrating them into the main pipeline. After integration, the user will run a full test of the pipeline to ensure compatibility. Do not skip local tests, even for small changes.
9. **Streamlit App:** Treat the Streamlit app as a secondary interface. Prioritize CLI and pipeline integration for new features. Only update the Streamlit app if specifically requested or if the feature is user-facing and interactive.
10. **Documentation Updates:** Document all new features/components in a "new_things" changelog or similar file. Do not update the main documentation for every small change; batch updates in the changelog for each PR.
11. **Hydra Configuration:** All new experiment types, models, and configurations must be Hydra-based. Do not create ad-hoc scripts or configs outside the Hydra system. Ensure all parameters are configurable via Hydra.
12. **Repository Structure:** Always respect and maintain the established repository structure and file organization. Ensure that all new files, modules, scripts, and documentation are placed in the correct, intended directories (e.g., models in src/models/, configs in conf/, visualizations in src/visualizations/, tests in tests/, etc.). Do not create new top-level folders or place files in ad-hoc locations. If unsure about the correct location, ask the user for guidance before proceeding.
13. **Code Change Communication:** When a code modification or creation task is set, first explain briefly in 3-5 lines what you will do, then perform the code modification. After modifying, explain in 3-5 lines what was solved/done and what to expect.
14. **Direct Action:** When you know what to do, directly modify the code without asking for permission or what the user wants to do, unless clarification is needed.
15. **Testing and Validation:** When modifying code or creating a new module, always create a test script and ensure that EVERYTHING works as expected (no errors, all tests pass) before considering the task complete.

---

**These rules are mandatory for all development, experimentation, and contributions in this repository.** 