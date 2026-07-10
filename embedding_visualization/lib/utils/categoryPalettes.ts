/**
 * Categorical palette registry.
 *
 * Each palette is a named array of hex color strings.
 * To add a new palette, add an entry to CATEGORY_PALETTES.
 * The DEFAULT_PALETTE_KEY controls which palette generateCategoryColors uses
 * when no explicit palette name is passed.
 */

export interface CategoricalPalette {
  label: string;
  colors: readonly string[];
}

export const CATEGORY_PALETTES: Record<string, CategoricalPalette> = {
  cosmicGalaxy: {
    label: 'Cosmic Galaxy',
    colors: [
      '#0b7285', // deep nebula teal
      '#66d9e8', // dusty teal echo
      '#d4a017', // stellar gold
      '#f0d77b', // pale gold echo
      '#1971c2', // O-star blue
      '#74c0fc', // ice blue echo
      '#e8590c', // Carina coral
      '#ffa094', // salmon mist echo
      '#2f9e44', // nebula emerald
      '#8ce99a', // pale jade echo
      '#c92a2a', // red dwarf
      '#ffa8a8', // rose echo
      '#7048e8', // ionised violet
      '#b197fc', // lavender echo
      '#e67700', // galactic amber
      '#ffd8a8', // pale wheat echo
      '#d6336c', // supernova magenta
      '#faa2c1', // blush echo
      '#364fc7', // dark-matter indigo
      '#91a7ff', // silver-blue echo
    ],
  },
  // Light-mode sibling of Cosmic Galaxy: same ten hue families in the same
  // order, but every slot holds ≥3:1 WCAG contrast against white so points
  // and cluster-label titles stay readable outside dark mode. The pale
  // "echo" slots become deep siblings instead. Keep any new color ≥3:1 on
  // #ffffff. (#0b7285 sits below the OKLCH 0.10 chroma floor only because
  // sRGB cannot produce a more saturated deep teal — it is at the gamut edge.)
  cosmicGalaxyLight: {
    label: 'Cosmic Galaxy Light',
    colors: [
      '#0b7285', // deep nebula teal
      '#0d97ad', // bright lagoon teal
      '#a27900', // stellar gold, deepened for white
      '#7e5900', // old-star bronze
      '#1971c2', // O-star blue
      '#2294d3', // clear-sky azure
      '#d9480f', // Carina coral, deepened
      '#953b17', // iron-oxide rust
      '#2f9e44', // nebula emerald
      '#087f5b', // comet sea-green
      '#c92a2a', // red dwarf
      '#932a33', // garnet ember
      '#7048e8', // ionised violet
      '#9775fa', // bright lavender
      '#d9700f', // galactic amber, deepened
      '#905211', // burnt sienna
      '#d6336c', // supernova magenta
      '#972b5f', // deep plum-rose
      '#364fc7', // dark-matter indigo
      '#5c7cfa', // periwinkle flare
    ],
  },
  // Cosmic Galaxy extended to 32 colors: the first 20 are identical to
  // cosmicGalaxy (switching palettes keeps existing colors), followed by six
  // more main+echo families in the same dark-starfield style.
  cosmicGalaxyXL: {
    label: 'Cosmic Galaxy XL',
    colors: [
      '#0b7285', // deep nebula teal
      '#66d9e8', // dusty teal echo
      '#d4a017', // stellar gold
      '#f0d77b', // pale gold echo
      '#1971c2', // O-star blue
      '#74c0fc', // ice blue echo
      '#e8590c', // Carina coral
      '#ffa094', // salmon mist echo
      '#2f9e44', // nebula emerald
      '#8ce99a', // pale jade echo
      '#c92a2a', // red dwarf
      '#ffa8a8', // rose echo
      '#7048e8', // ionised violet
      '#b197fc', // lavender echo
      '#e67700', // galactic amber
      '#ffd8a8', // pale wheat echo
      '#d6336c', // supernova magenta
      '#faa2c1', // blush echo
      '#364fc7', // dark-matter indigo
      '#91a7ff', // silver-blue echo
      '#099268', // sea-green comet
      '#63e6be', // seafoam echo
      '#66a80f', // aurora lime
      '#c0eb75', // pale lime echo
      '#9c36b5', // quasar grape
      '#da77f2', // orchid echo
      '#a05f2f', // copper dust
      '#e3ab6a', // pale sand echo
      '#b22855', // deep-space wine
      '#f783ac', // rose quartz echo
      '#928e04', // citron flare
      '#d3d566', // pale citron echo
    ],
  },
  // Thematic single-family "galaxies": one hue family per palette, category
  // separation carried by strong lightness alternation (dark/bright) plus
  // sub-hue shifts. Tuned for the dark starfield first (all ≥3:1 on black),
  // mid-range enough to stay visible on white.
  emeraldGalaxy: {
    label: 'Emerald Galaxy',
    colors: [
      '#1f6f3d', // deep viridian
      '#6cd981', // bright spring
      '#007c59', // pine teal
      '#81b84d', // leaf green
      '#3a8f42', // emerald
      '#3db07c', // jade
      '#446a2c', // forest
      '#37cda0', // seafoam
      '#587c18', // moss
      '#3aba6a', // malachite
      '#0b764d', // deep sea green
      '#74c163', // light meadow
    ],
  },
  azureGalaxy: {
    label: 'Azure Galaxy',
    colors: [
      '#295ab9', // deep cobalt
      '#75caf2', // ice blue
      '#006eae', // royal blue
      '#26b7d3', // sky cyan
      '#417acc', // steel azure
      '#3ca2e0', // clear azure
      '#006b99', // deep ocean
      '#81b4f6', // moonlight blue
      '#007ba1', // cerulean
      '#739bf5', // cornflower
      '#1666aa', // midnight blue
      '#53b6eb', // pale azure
    ],
  },
  emberGalaxy: {
    label: 'Ember Galaxy',
    colors: [
      '#a1392c', // deep garnet
      '#e4b750', // pale gold
      '#ab4500', // rust
      '#df911a', // amber
      '#c64e31', // ember red
      '#dd7b2b', // flame orange
      '#8f4700', // burnt umber
      '#e4a339', // honey
      '#b64340', // brick
      '#e38305', // marmalade
      '#8a5600', // bronze
      '#cda629', // gold dust
    ],
  },
  violetGalaxy: {
    label: 'Violet Galaxy',
    colors: [
      '#6f47a7', // deep violet
      '#e0a4ee', // pale orchid
      '#993f94', // magenta plum
      '#ab93ed', // lavender
      '#9b5bb6', // orchid
      '#d073b3', // pink mauve
      '#81468f', // dark plum
      '#c39dee', // lilac
      '#6f5db9', // indigo violet
      '#cc7bd1', // bright orchid
      '#953d7c', // wine violet
      '#c692e6', // light violet
    ],
  },
  Galaxy: {
    label: 'Galaxy',
    colors: [
      '#9ca3af', // deep nebula teal
      '#7f7f7f', // dusty teal echo
      '#d4a017', // stellar gold
      '#f0d77b', // pale gold echo
      '#1971c2', // O-star blue
      '#74c0fc', // ice blue echo
      '#e8590c', // Carina coral
      '#ffa094', // salmon mist echo
      '#2f9e44', // nebula emerald
      '#8ce99a', // pale jade echo
      '#c92a2a', // red dwarf
      '#fbeaf1', // rose echo
      '#7048e8', // ionised violet
      '#b197fc', // lavender echo
      '#e67700', // galactic amber
      '#ffd8a8', // pale wheat echo
      '#d6336c', // supernova magenta
      '#faa2c1', // blush echo
      '#364fc7', // dark-matter indigo
      '#91a7ff', // silver-blue echo
    ],
  },
  category10: {
    label: 'D3 Category 10',
    colors: [
      '#1f77b4',
      '#ff7f0e',
      '#2ca02c',
      '#d62728',
      '#9467bd',
      '#8c564b',
      '#e377c2',
      '#7f7f7f',
      '#bcbd22',
      '#17becf',
    ],
  },
  category20: {
    label: 'D3 Category 20',
    colors: [
      '#1f77b4',
      '#aec7e8',
      '#ff7f0e',
      '#ffbb78',
      '#2ca02c',
      '#98df8a',
      '#d62728',
      '#ff9896',
      '#9467bd',
      '#c5b0d5',
      '#8c564b',
      '#c49c94',
      '#e377c2',
      '#f7b6d2',
      '#7f7f7f',
      '#c7c7c7',
      '#bcbd22',
      '#dbdb8d',
      '#17becf',
      '#9edae5',
    ],
  },
};

/** The palette key used when no palette is explicitly specified. */
export const DEFAULT_PALETTE_KEY = 'cosmicGalaxy';

/** All built-in palette names, for UI iteration. */
export const BUILTIN_PALETTE_NAMES = Object.keys(CATEGORY_PALETTES);

/** Get a palette's color array by name. */
export function getPaletteColors(name: string): readonly string[] | undefined {
  return CATEGORY_PALETTES[name]?.colors;
}
