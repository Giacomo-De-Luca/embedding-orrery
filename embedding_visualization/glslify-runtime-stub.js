/**
 * Runtime stub for glslify, aliased in next.config.ts.
 *
 * The regl-* packages (regl-scatter2d, regl-line2d, regl-error2d, gl-text)
 * ship pre-compiled shaders but still wrap them in a runtime call:
 *   markerOptions.frag = glslify(["...shader source..."])
 * At runtime glslify only needs to behave as a tagged-template join and
 * return the source string. Aliasing to `false` (empty module) makes that
 * call throw "glslify is not a function" inside createScatter(), which
 * react-plotly swallows silently — scattergl then never renders.
 */
module.exports = function glslify(strings) {
  if (Array.isArray(strings)) {
    var result = strings[0];
    for (var i = 1; i < arguments.length; i++) {
      result += String(arguments[i]) + strings[i];
    }
    return result;
  }
  return String(strings);
};
