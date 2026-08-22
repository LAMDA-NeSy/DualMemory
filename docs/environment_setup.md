# Environment Setup

Please follow the environment configuration in the StateAct repository:

https://github.com/ai-nikolai/StateAct

This repository uses the three StateAct benchmark environments:
ALFWorld, WebShop, and TextCraft. Large benchmark assets, including the
WebShop image files, are not redistributed here.

The published ALFWorld task files are:

- `tasks/alfworld_s1_tasks_suffix.json`: Stage 1, 50 `train` tasks.
- `tasks/alfworld_s3_tasks_suffix.json`: Stage 3, 134 `valid_unseen` tasks.

The combined task manifests are also public:

- `tasks/webshop_s1_fixed.json`: 50 fixed WebShop tasks.
- `tasks/webshop_s3_fixed.json`: 100 fixed WebShop tasks.
- `tasks/textcraft_s1_fixed.json`: TextCraft seeds `0..49`, with full recipe specifications.
- `tasks/textcraft_s3_fixed.json`: TextCraft seeds `50..149`, with full recipe specifications.

The public runners validate the task text returned by WebShop/TextCraft against
these manifests after reset, so use the same benchmark data version as StateAct.

The top-level `requirements.txt` installs method dependencies only. The
StateAct setup is still required to provide the ALFWorld package and assets,
the WebShop server/data, and the TextCraft package/data.
