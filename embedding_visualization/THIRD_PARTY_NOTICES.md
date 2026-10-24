# Third-party notices

The frontend includes code adapted from the projects below. Each section lists the files that come from the project and gives its license notice. The project itself is licensed under the Apache License 2.0: see `LICENSE` at the repository root, or https://www.apache.org/licenses/LICENSE-2.0.

## Embedding Atlas

Apple Inc., https://github.com/apple/embedding-atlas, MIT License. The files below are adapted, some with modifications:

- `app/utils/dynamicMapPlacement.ts`, `app/utils/priorityQueue.ts`: dynamic map label placement (the 2D zoom bands). `dynamicMapPlacement.ts` is modified for speed and gives the same placements. Its tests compare it with the unmodified original, kept in `app/utils/__tests__/fixtures/referenceDynamicMapPlacement.ts`.
- `app/utils/text_summarizer.ts`, `app/utils/stop_words.ts`: keyword summaries of plot regions.
- `app/utils/densityClusterLabels.ts`: the density-blob label pipeline, ported and adapted.
- `app/utils/density_clustering/`: the density clustering crate (Rust). Its WebAssembly build is in `public/wasm/` and `lib/density-clustering/pkg/`, wrapped by `lib/density-clustering/index.ts`.
- `lib/utils/densityRenderer/`: the 2D density overlay renderer, ported and adapted. That covers its shaders (`shaders.ts`), the renderer (`DensityRenderer.ts`), its maths (`densityMath.ts`) and its WebGL resources (`glResources.ts`).

```
The MIT License (MIT)

Copyright (c) 2025 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## plotly.js

Plotly Technologies Inc., https://github.com/plotly/plotly.js, MIT License.

- Forked and modified in `forked/plotly.js/`. The changes are listed in `forked/plotly_changes.md`.
- plotly.js bundles further open-source modules under their own licenses.

```
MIT License

Copyright (c) 2016-2024 Plotly Technologies Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## react-plotly.js

Plotly, Inc., https://github.com/plotly/react-plotly.js, MIT License. Forked and modified in `forked/react-plotly.js/`. The changes are listed in `forked/plotly_changes.md`.

```
The MIT License (MIT)

Copyright (c) 2017 Plotly, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## TensorBoard embedding projector

The TensorFlow Authors, https://github.com/tensorflow/tensorboard (`tensorboard/plugins/projector`), Apache License 2.0. The files below come from it, with modifications:

- `lib/utils/collisionGrid.ts`: the label collision grid, ported.
- `lib/utils/labels/pointLabels.ts`: the probe-first label placement of its canvas labels.

```
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
