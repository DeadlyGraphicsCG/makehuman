# Status

Updated: 2026-08-17

## Face pipeline

- `chore/category-reorg` contains `c7cbbccf chore(face-pipeline): correct DG workspace paths`.
- The commit corrects DG_Brain and MakeHuman absolute-path references and adds the Sapiens plus MediaPipe Character Modeller planning document.
- `codex/face-pipeline-retopo-texture-bake` and `chore/category-reorg` shared the same tip before this path-correction commit. The face-pipeline work should use one PR, not two.
- Validation for the path-correction commit: `git diff --check` passed before staging the existing plan file; the staged plan has one trailing blank-line warning. The eleven changed Python files compiled with `C:\AI\apps\Core\DG_Brain\.venv\Scripts\python.exe -m py_compile`.
