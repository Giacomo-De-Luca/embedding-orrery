'use strict';

var Plotly = require('./core');

// [fork] The app only creates scattergl (2D) and scatter3d (3D) traces, so the entry
// registers just those. Upstream's full list (~48 traces plus the calendars component)
// also bundled two map libraries that load as soon as the module is evaluated. Core
// still registers scatter, modebar, select/lasso, legend and toImage. A new trace type
// must be registered here; forked/__tests__/plotlyEntry.test.ts fails until it is.
// See forked/plotly_changes.md, change 5.
Plotly.register([
    require('./scattergl'),
    require('./scatter3d'),
]);

module.exports = Plotly;
