# Third-Party Notices & Attributions

## LLM Architecture Gallery Dataset (`src/data/galleryModels.json`)

This project bundles model metadata derived from **Sebastian Raschka's LLM Architecture Gallery**:
- Source Repository: https://github.com/rasbt/llm-architecture-gallery
- Gallery Webpage: https://sebastianraschka.com/llm-architecture-gallery/
- License: **Apache License, Version 2.0**

Modifications made:
- Synced and normalized all **132 open-weight models** from Sebastian Raschka's LLM Architecture Gallery (`https://sebastianraschka.com/llm-architecture-gallery/` & `models.yml`) into typed JSON (`src/data/galleryModels.json`) with normalized numeric parameter, context-length, vocabulary, and KV-cache KiB fields for interactive filtering, Pareto plotting, and similarity scoring.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use the derived dataset except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
