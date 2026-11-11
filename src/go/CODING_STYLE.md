# C++ style

- C++17, four spaces, no tabs, and approximately 120 columns.
- Opening braces remain on the declaration/control-flow line.
- Types/classes use `PascalCase`; functions use `camelCase`.
- Variables and files use `snake_case`; private fields end in `_`.
- Preserve existing public identifiers; use `kName` for new typed constants.
- Keep includes explicit. Prefer value objects, RAII, and clear ownership.
- Use English for new comments, documentation, and user-visible diagnostics.
- Explain units, ownership, locking, invariants, and non-obvious formulas. Avoid comments that repeat a statement.
- Keep YAML parsing in `config_manager.cpp` and simulator glue in `GoNode`.
- Inject configuration into modules rather than adding scattered global constants.
- Test observable behavior and regressions. Do not claim GPU validation from CPU-only tests.

The existing `.clang-format` is the formatting source of truth. Avoid formatting unrelated lines during a functional fix. Historical implementation comments remain available while documentation is being consolidated.
