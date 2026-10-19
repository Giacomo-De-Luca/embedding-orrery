/**
 * Categorical palette registry.
 *
 * Each palette is a named array of hex color strings.
 * To add a new palette, add an entry to CATEGORY_PALETTES.
 * The DEFAULT_PALETTE_KEY controls which palette generateCategoryColors uses
 * when no explicit palette name is passed.
 *
 * Categories take slots in order, so the first slots matter most: every
 * palette here is ordered so that its short prefixes are as distinct as the
 * design allows. Design notes, validation numbers and the regeneration
 * workflow are in documentation/CATEGORICAL_PALETTES.md.
 */

export interface CategoricalPalette {
  label: string;
  colors: readonly string[];
}

/** One hue family of the Cosmic Galaxy palettes: dark-starfield main + pale echo, and the light-mode pair. */
export interface CosmicFamily {
  /** Saturated main colour (dark mode). */
  main: string;
  /** Pale "echo" of the main (dark mode); slots 10–19. */
  echo: string;
  /** Main colour deepened where needed to hold ≥3:1 on white. */
  lightMain: string;
  /** Deep sibling replacing the echo in light mode (≥3:1 on white). */
  lightDeep: string;
}

export const COSMIC_FAMILIES = {
  teal: { main: '#0b7285', echo: '#66d9e8', lightMain: '#0b7285', lightDeep: '#0d97ad' }, // deep nebula teal / dusty teal / bright lagoon
  gold: { main: '#d4a017', echo: '#f0d77b', lightMain: '#a27900', lightDeep: '#7e5900' }, // stellar gold / pale gold / old-star bronze
  red: { main: '#c92a2a', echo: '#ffa8a8', lightMain: '#c92a2a', lightDeep: '#932a33' }, // red dwarf / rose / garnet ember
  violet: { main: '#7048e8', echo: '#b197fc', lightMain: '#7048e8', lightDeep: '#9775fa' }, // ionised violet / lavender / bright lavender
  coral: { main: '#e8590c', echo: '#ffa094', lightMain: '#d9480f', lightDeep: '#953b17' }, // Carina coral / salmon mist / iron-oxide rust
  green: { main: '#2f9e44', echo: '#8ce99a', lightMain: '#2f9e44', lightDeep: '#087f5b' }, // nebula emerald / pale jade / comet sea-green
  blue: { main: '#1971c2', echo: '#74c0fc', lightMain: '#1971c2', lightDeep: '#2294d3' }, // O-star blue / ice blue / clear-sky azure
  indigo: { main: '#364fc7', echo: '#91a7ff', lightMain: '#364fc7', lightDeep: '#5c7cfa' }, // dark-matter indigo / silver-blue / periwinkle
  magenta: { main: '#d6336c', echo: '#faa2c1', lightMain: '#d6336c', lightDeep: '#972b5f' }, // supernova magenta / blush / deep plum-rose
  amber: { main: '#e67700', echo: '#ffd8a8', lightMain: '#d9700f', lightDeep: '#905211' }, // galactic amber / pale wheat / burnt sienna
} satisfies Record<string, CosmicFamily>;

export type CosmicFamilyName = keyof typeof COSMIC_FAMILIES;

/**
 * Slot order of the ten families. Teal leads for identity; the rest is the
 * farthest-first order over the main colours (PaletteDesigner.prefixOrder),
 * so 2, 3, 4… categories always get the most distinct mains available. The
 * pale echoes follow in the same order: slot i + 10 is the echo of slot i.
 */
export const COSMIC_FAMILY_ORDER: readonly CosmicFamilyName[] = [
  'teal', 'gold', 'red', 'violet', 'coral', 'green', 'blue', 'indigo', 'magenta', 'amber',
];

/** All families' `mainKey` colours in order, then all their `echoKey` colours. */
const cosmicSlots = (families: readonly Pick<CosmicFamily, 'main' | 'echo'>[], mainKey: 'main' | 'echo' = 'main', echoKey: 'main' | 'echo' = 'echo'): string[] => [
  ...families.map((f) => f[mainKey]),
  ...families.map((f) => f[echoKey]),
];
const COSMIC_ORDERED = COSMIC_FAMILY_ORDER.map((f) => COSMIC_FAMILIES[f]);

const COSMIC_GALAXY = cosmicSlots(COSMIC_ORDERED);

/** Six extra families appended by Cosmic Galaxy XL (their mains, then their echoes). */
const COSMIC_XL_EXTRA_FAMILIES: readonly Pick<CosmicFamily, 'main' | 'echo'>[] = [
  { main: '#099268', echo: '#63e6be' }, // sea-green comet / seafoam
  { main: '#66a80f', echo: '#c0eb75' }, // aurora lime / pale lime
  { main: '#9c36b5', echo: '#da77f2' }, // quasar grape / orchid
  { main: '#a05f2f', echo: '#e3ab6a' }, // copper dust / pale sand
  { main: '#b22855', echo: '#f783ac' }, // deep-space wine / rose quartz
  { main: '#928e04', echo: '#d3d566' }, // citron flare / pale citron
];

export const CATEGORY_PALETTES: Record<string, CategoricalPalette> = {
  // Ten saturated mains first, their pale echoes after (slot i + 10 echoes
  // slot i). Tuned for the black starfield; echoes drop to 1.3–2.4:1 on white.
  cosmicGalaxy: {
    label: 'Cosmic Galaxy',
    colors: COSMIC_GALAXY,
  },
  // Light-mode sibling: same families in the same order, every slot ≥3:1 on
  // white so points and cluster-label titles stay readable outside dark mode.
  // The echo slots hold deep siblings instead of pale colours. (#0b7285 sits
  // below the OKLCH 0.10 chroma floor only because sRGB cannot produce a more
  // saturated deep teal — it is at the gamut edge.)
  cosmicGalaxyLight: {
    label: 'Cosmic Galaxy Light',
    colors: cosmicSlots(COSMIC_ORDERED.map((f) => ({ main: f.lightMain, echo: f.lightDeep }))),
  },
  // Cosmic Galaxy extended to 32: the first 20 are identical to cosmicGalaxy
  // (switching palettes keeps existing colors), then six more mains and their
  // echoes in the same dark-starfield style.
  cosmicGalaxyXL: {
    label: 'Cosmic Galaxy XL',
    colors: [...COSMIC_GALAXY, ...cosmicSlots(COSMIC_XL_EXTRA_FAMILIES)],
  },
  // ---- Generated palettes: do not hand-edit. Recipes live in paletteRecipes.ts;
  // ---- regenerate with `npm run palettes:design` (paletteDesign.test.ts enforces parity).
  // Hubble Deep Field: 64 saturated colours over the whole hue circle, picked
  // farthest-first so any prefix stays as distinct as possible. Opens with the
  // default's teal and gold. For maps with more than 20 topics.
  deepField: {
    label: 'Deep Field',
    colors: [
      '#0b7285', '#d4a017', '#fd78ff', '#8400f5', '#c90015', '#00e7b1', '#00a880', '#8b6fe8',
      '#b6d700', '#c86b35', '#ff6ba1', '#00afd8', '#f3b3ff', '#a900b7', '#ffa47a', '#c0006c',
      '#005dca', '#00cfd4', '#00d061', '#008bbf', '#ed6461', '#d93e61', '#9c44ff', '#007745',
      '#be76ff', '#ff3806', '#009233', '#d98940', '#00b6ae', '#5cb5ff', '#cd4300', '#9a4825',
      '#ff8bb7', '#2cdbff', '#cbb800', '#942f9e', '#3079d0', '#54d28c', '#b6324a', '#7e47e3',
      '#3d94e2', '#ffa5d8', '#c8397a', '#ac3181', '#ff4481', '#2663ac', '#00ac4f', '#b459ff',
      '#f08085', '#009dcb', '#007aae', '#f3ae52', '#769dec', '#9658e6', '#6038df', '#00c9b1',
      '#358d4f', '#f070cd', '#00e187', '#8e35c5', '#00e1e6', '#ef7a00', '#00c0ec', '#a27feb',
    ],
  },
  // Aurora: oxygen greens and limes, teal, and violet/magenta curtains.
  aurora: {
    label: 'Aurora',
    colors: [
      '#29d975', '#9200fe', '#cfa9ff', '#5d7f00', '#ba3580', '#ff66b3', '#9a75e9', '#00a880',
      '#60e7d7', '#abe500', '#7ca800', '#a237c0', '#00b1cd', '#929bff', '#aa4cff', '#009b3d',
    ],
  },
  // Hubble SHO false colour: golds, ambers and rusts against teals and blues.
  hubble: {
    label: 'Hubble Palette',
    colors: [
      '#efb225', '#0051e3', '#ac3400', '#1bc7ff', '#ff540a', '#008cbc', '#ed9377', '#3e7fff',
      '#00b5b2', '#2c64a5', '#769ee8', '#00eaed', '#b25f35', '#be8041', '#ffb27e', '#00cfd1',
    ],
  },
  // Pulsar: electric colours at the chroma ceiling, glowing on black.
  pulsar: {
    label: 'Pulsar',
    colors: [
      '#00e9f3', '#fa0013', '#8e5fff', '#ffb411', '#ff79a8', '#51a9ff', '#009f68', '#00fc8e',
      '#bba500', '#eee300', '#ff65e8', '#00c46d', '#ff9464', '#00e5af',
    ],
  },
  // ---- End of generated palettes.
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
  // Legacy variant of the original interleaved cosmicGalaxy with a grey
  // leading pair; kept for collections whose saved default names it.
  Galaxy: {
    label: 'Galaxy',
    colors: [
      '#9ca3af', // cool grey
      '#7f7f7f', // mid grey
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
