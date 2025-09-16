# Supplementary Code Repository
This repository contains supplementary code for our paper **"Zero-Shot Vision-Language Reasoning via Dual-layer Scene Graph Chain-of-Thoughts"**.  
It includes scripts for generating scene graphs and producing answers across multiple datasets.

---

## Repository Structure

```
supplementary/
├── MM-Bench/
│   ├── mm.py           - Generates the objects list JSON file and the global scene graph JSON file
│   ├── mm_vqa.py       - Produces final answers from scene graph data
│   └── query_mm.py     - Builds query-specific scene graph JSON
├── Whoops!/
│   ├── whoops.py       - Generates the objects list JSON file and the global scene graph JSON file
│   ├── whoops_vqa.py   - Produces final answers from scene graph data
│   └── query_whoops.py - Builds query-specific scene graph JSON
└── SEED/
    ├── seed.py         - Generates the objects list JSON file and the global scene graph JSON file
    ├── seed_vqa.py     - Produces final answers from scene graph data
    └── query_seed.py   - Builds query-specific scene graph JSON
```

---

## Folder Descriptions

### **MM-Bench**
Code specific to the **MM-Bench** dataset.  
- **mm.py** — Generates the objects list JSON file and the global scene graph JSON file.  
- **mm_vqa.py** — Produces final answers based on scene graph data.  
- **query_mm.py** — Builds query-specific scene graph JSON files for targeted analysis.  

### **Whoops!**
Code specific to the **Whoops!** dataset.  
- **whoops.py** — Generates the objects list JSON file and the global scene graph JSON file.  
- **whoops_vqa.py** — Produces final answers based on scene graph data.  
- **query_whoops.py** — Builds query-specific scene graph JSON files for targeted analysis.  

### **SEED**
Code specific to the **SEED** dataset.  
- **seed.py** — Generates the objects list JSON file and the global scene graph JSON file.  
- **seed_vqa.py** — Produces final answers based on scene graph data.  
- **query_seed.py** — Builds query-specific scene graph JSON files for targeted analysis.  

---

## Notes
- Each dataset follows the same modular structure for consistency.  
- The scripts support **scene graph generation** and **question answering workflows** across datasets.  

---
