
Supplementary Code Repository

This repository contains supplementary code for generating scene graphs and answers using multiple datasets. The folder structure is as follows:


supplementary/
├── MM-Bench/
│   ├── mm.py
│   ├── mm_vqa.py
│   └── query_mm.py
├── Whoops!/
│   ├── whoops.py
│   ├── whoops_vqa.py
│   └── query_whoops.py
├── SEED/
│   ├── seed.py
│   ├── seed_vqa.py
│   └── query_seed.py



Folder Descriptions

 MM-Bench
This folder contains code specific to the MM dataset.
- mm.py  
  Generates the objects list JSON file and the global scene graph JSON file.
- mm_vqa.py  
  Handles the final answer generation based on the scene graph data.
- query_mm.py  
  Creates the query-specific scene graph JSON file for targeted analysis.

 Whoops!
This folder contains code specific to the Whoops! dataset.
- whoops.py
  Generates the objects list JSON file and the global scene graph JSON file.
- whoops_vqa.py
  Handles the final answer generation based on the scene graph data.
- query_whoops.py 
  Creates the query-specific scene graph JSON file for targeted analysis.

SEED
This folder contains code specific to the SEED dataset.
- seed.py
  Generates the objects list JSON file and the global scene graph JSON file.
- seed_vqa.py
  Handles the final answer generation based on the scene graph data.
- query_seed.py
  Creates the query-specific scene graph JSON file for targeted analysis.


Each dataset follows the same structure to ensure consistency across different tasks. These scripts are designed to support scene graph generation and question answering workflows in a modular fashion.
