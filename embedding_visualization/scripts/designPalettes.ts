/**
 * Regenerate the computationally designed categorical palettes and print an
 * audit of every built-in palette.
 *
 *   npm run palettes:design            # all recipes + audit
 *   npm run palettes:design -- aurora  # one recipe
 *
 * Paste the printed arrays into lib/utils/categoryPalettes.ts; the audit
 * numbers go into documentation/CATEGORICAL_PALETTES.md.
 */

import { PaletteDesigner, PaletteAudit } from '../lib/utils/paletteDesign';
import { GENERATED_PALETTE_RECIPES } from '../lib/utils/paletteRecipes';
import { CATEGORY_PALETTES } from '../lib/utils/categoryPalettes';

const PREFIXES = [4, 8, 12, 16, 20, 24, 32, 48, 64];
const only = process.argv.slice(2);
const known = GENERATED_PALETTE_RECIPES.map((r) => r.key);
const unknown = only.filter((key) => !known.includes(key));
if (unknown.length) console.warn(`Unknown recipe key(s): ${unknown.join(', ')}. Known: ${known.join(', ')}`);
const recipes = only.length ? GENERATED_PALETTE_RECIPES.filter((r) => only.includes(r.key)) : GENERATED_PALETTE_RECIPES;

const formatArray = (colors: readonly string[]) => {
  const lines: string[] = [];
  for (let i = 0; i < colors.length; i += 8) lines.push('      ' + colors.slice(i, i + 8).map((h) => `'${h}'`).join(', ') + ',');
  return lines.join('\n');
};

for (const recipe of recipes) {
  const colors = PaletteDesigner.generate(recipe);
  const committed = CATEGORY_PALETTES[recipe.key]?.colors ?? [];
  const same = committed.length === colors.length && committed.every((c, i) => c === colors[i]);
  console.log(`\n// ${recipe.key}: ${colors.length} colours ${same ? '(matches the committed palette)' : '(DIFFERS from the committed palette — paste this in)'}`);
  console.log(formatArray(colors));
}

console.log('\n== Audit (all-pairs minima: plain OKLab ΔE×100 / worst protan-or-deutan ΔE) ==');
for (const [key, palette] of Object.entries(CATEGORY_PALETTES)) {
  const adjacent = PaletteAudit.pairStats(palette.colors, 'adjacent');
  const prefixes = PaletteAudit.prefixReport(palette.colors, PREFIXES)
    .map((p) => `${p.k}:${p.normal.toFixed(1)}/${p.cvd.toFixed(1)}`)
    .join('  ');
  console.log(`${key.padEnd(18)} n=${String(palette.colors.length).padStart(2)}  black ≥${PaletteAudit.minContrast(palette.colors, '#000000').toFixed(2)}  white ≥${PaletteAudit.minContrast(palette.colors, '#ffffff').toFixed(2)}  adjacent ${adjacent.normal.toFixed(1)}/${adjacent.cvd.toFixed(1)}`);
  console.log(`${''.padEnd(18)} prefixes ${prefixes}`);
}
