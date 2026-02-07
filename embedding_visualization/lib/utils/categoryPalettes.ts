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
