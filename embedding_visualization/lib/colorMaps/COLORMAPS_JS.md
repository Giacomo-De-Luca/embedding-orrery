# Crameri Scientific Colormaps for JavaScript

Converted from [Fabio Crameri's perceptually uniform colormaps](https://www.fabiocrameri.ch/colourmaps/) (v8.0) for use with D3.js and Plotly.js.

These colormaps are designed to be perceptually uniform, colour-blind friendly, and suitable for scientific visualisation.

## Files

| File | Format | Description |
|------|--------|-------------|
| `colormaps/batlow.json` | JSON | Individual colormap file (~12KB each for 256 colours, ~4.5KB for categorical) |
| `colormaps/index.json` | JSON | Lightweight index — name, type, and colour count for all 60 colormaps |
| `colormaps.json` | JSON | All 60 colormaps in a single file (for convenience when bundle size isn't a concern) |
| `colormaps.js` | ES module | All colormaps with D3 `d3Interpolator` functions |
| `convert.mjs` | Node.js script | Regenerates everything from the source `.txt` files |

### Individual vs monolithic

- **`colormaps/batlow.json`** — load only the colormaps you need (~12KB each)
- **`colormaps.json`** — all 60 colormaps in one file (~1MB) if you need quick access to everything

## Available Colormaps (60 total)

### Sequential (21)
`acton` `bamako` `batlow` `batlowK` `batlowW` `bilbao` `buda` `davos` `devon` `glasgow` `grayC` `hawaii` `imola` `lajolla` `lapaz` `lipari` `navia` `nuuk` `oslo` `tokyo` `turku`

### Diverging (10)
`bam` `berlin` `broc` `cork` `lisbon` `managua` `roma` `tofino` `vanimo` `vik`

### Cyclic (5)
`bamO` `brocO` `corkO` `romaO` `vikO`

### Multi-Sequential (3)
`bukavu` `fes` `oleron`

### Categorical (21)
Every sequential colormap has a categorical variant with an `S` suffix:
`actonS` `bamakoS` `batlowS` `batlowKS` `batlowWS` `bilbaoS` `budaS` `davosS` `devonS` `glasgowS` `grayCS` `hawaiiS` `imolaS` `lajollaS` `lapazS` `lipariS` `naviaS` `nuukS` `osloS` `tokyoS` `turkuS`

Sequential, diverging, cyclic, and multi-sequential colormaps contain **256 colours**. Categorical colormaps contain **100 colours**.

## Usage

### Plotly.js

Load a single colormap:

```js
// Individual file (recommended — only loads what you need)
import batlow from './colormaps/batlow.json' with { type: 'json' };

Plotly.newPlot('div', [{
  z: data,
  type: 'heatmap',
  colorscale: batlow.plotly,
}]);
```

Or load all from the monolithic file:

```js
import colormaps from './colormaps.json';

Plotly.newPlot('div', [{
  x, y,
  mode: 'markers',
  marker: {
    color: values,
    colorscale: colormaps.vik.plotly,
  },
}]);
```

### D3.js

Import from the ES module and use `d3Interpolator` with `d3.scaleSequential`:

```js
import { batlow, vik } from './colormaps.js';

// Continuous scale
const scale = d3.scaleSequential(batlow.d3Interpolator)
  .domain([0, 100]);

scale(50); // => "rgb(130,130,49)"

// Diverging scale (centred at 0)
const diverging = d3.scaleDiverging(vik.d3Interpolator)
  .domain([-1, 0, 1]);
```

### Dynamic loading (fetch)

Load a colormap at runtime without bundling:

```js
const response = await fetch('./colormaps/batlow.json');
const batlow = await response.json();

// Use with Plotly
Plotly.restyle('div', { colorscale: [batlow.plotly] });

// Or build a D3 interpolator on the fly
const interpolator = (t) =>
  batlow.colors[Math.round(Math.max(0, Math.min(1, t)) * (batlow.colors.length - 1))];
const scale = d3.scaleSequential(interpolator).domain([0, 100]);
```

### Browsing available colormaps

Use the index to list colormaps without loading all the colour data:

```js
import index from './colormaps/index.json' with { type: 'json' };

// { acton: { type: "sequential", numColors: 256 }, ... }
Object.keys(index);  // all 60 names

// Filter by type
const diverging = Object.entries(index)
  .filter(([, info]) => info.type === 'diverging')
  .map(([name]) => name);
// => ["bam", "berlin", "broc", "cork", "lisbon", "managua", "roma", "tofino", "vanimo", "vik"]
```

### Accessing raw colours

```js
import { batlow } from './colormaps.js';

batlow.colors;        // ["rgb(1,25,89)", "rgb(2,27,90)", ..., "rgb(250,204,250)"]
batlow.colors.length; // 256
batlow.type;          // "sequential"
```

### Categorical colormaps

Categorical variants (`S` suffix) contain 100 distinct colours suitable for discrete categories:

```js
import { batlowS } from './colormaps.js';

batlowS.colors;        // 100 distinct colours
batlowS.type;          // "categorical"

// Use individual colours for category assignments
const categoryColours = batlowS.colors.slice(0, numCategories);
```

### Converting to hex

The colours are exported as `rgb(r,g,b)` strings. To convert to hex:

```js
function rgbToHex(rgb) {
  const [r, g, b] = rgb.match(/\d+/g).map(Number);
  return '#' + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');
}

rgbToHex('rgb(1,25,89)'); // => "#011959"
```

## JSON Structure

Each individual JSON file (and each entry in `colormaps.json`) has:

```json
{
  "name": "batlow",
  "type": "sequential",
  "colors": ["rgb(1,25,89)", "rgb(2,27,90)", "..."],
  "plotly": [[0, "rgb(1,25,89)"], [0.003922, "rgb(2,27,90)"], "...", [1, "rgb(250,204,250)"]]
}
```

- **name** — colormap identifier
- **type** — `"sequential"`, `"diverging"`, `"cyclic"`, `"multi-sequential"`, or `"categorical"`
- **colors** — array of `rgb(r,g,b)` strings (256 or 100 entries)
- **plotly** — array of `[position, color]` pairs where position ranges from 0 to 1

## Regenerating

If the source `.txt` files are updated, regenerate all outputs:

```bash
node convert.mjs
```

## Attribution

Colormaps by Fabio Crameri, distributed under the MIT licence.
Python wrapper by Callum Rollo. JavaScript conversion for this project.

Reference: Crameri, F. (2023). Scientific colour maps (Version 8.0). Zenodo. https://doi.org/10.5281/zenodo.8035877
