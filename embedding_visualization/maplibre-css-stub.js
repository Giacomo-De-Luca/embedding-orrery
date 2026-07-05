// Empty CommonJS module, aliased in next.config.ts's turbopack.resolveAlias in
// place of maplibre-gl/dist/maplibre-gl.css.
//
// forked/plotly.js/src/registry.js does `require('maplibre-gl/dist/maplibre-gl.css')`
// purely for its style side-effect when registering map traces (which this app
// never uses). Turbopack cannot instantiate a CSS module that is required from
// JS ("module factory is not available"), and that error aborts the ENTIRE
// Plotly load so the scatter plots never mount. Resolving the CSS request to an
// empty JS module sidesteps the CSS-module path (aliasing CSS->CSS does not
// work — the target must be a JS module). Webpack imports the real CSS fine, so
// this is Turbopack-only.
module.exports = {};
