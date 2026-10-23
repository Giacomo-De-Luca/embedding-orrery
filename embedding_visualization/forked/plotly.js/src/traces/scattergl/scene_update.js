'use strict';

var Lib = require('../../lib');

// regl components are first marked as needed with the placeholder `true`
// (see calc.js) and only replaced by real objects in the dirty plot call
// (see plot.js). A drag can call scene.update/draw inside that window,
// so treat the placeholder as "not ready yet".
function isReady(component) {
    return component && component !== true;
}

// [fork] change 6: scale each trace's markers by config.scatterglOpacityScale and
// config.scatterglSizeScale before a draw, always from the base values in markerOptions,
// so draws never compound. The opacity uniform and a single marker size are written to
// the group directly: scatter2d.update() re-parses every option alias per trace, which
// costs milliseconds per frame with thousands of traces. Per-point sizes live in a GPU
// buffer, so they go through update(), and only when their factor changes.
function applyMarkerScale(gd, scene) {
    var context = gd._context || {};
    var opacityScale = typeof context.scatterglOpacityScale === 'function' ? context.scatterglOpacityScale : null;
    var sizeScale = typeof context.scatterglSizeScale === 'function' ? context.scatterglSizeScale : null;
    if(!(opacityScale || sizeScale) || !isReady(scene.scatter2d)) return;

    var groups = scene.scatter2d.groups;
    var scaledSizes = scene.scaledSizes || (scene.scaledSizes = []);
    var sizeUpdates = null;
    var calcdata = gd.calcdata || [];
    for(var k = 0; k < calcdata.length; k++) {
        var cd0 = calcdata[k] && calcdata[k][0];
        var stash = cd0 && cd0.t;
        if(!stash || stash._scene !== scene) continue;

        var i = stash.index;
        var options = scene.markerOptions[i];
        var group = groups[i];
        if(!options || !group) continue;
        // A trace with a selection keeps plotly's selected/unselected opacity; its
        // selected points are drawn by select2d, so a single size is scaled there too.
        var selected = scene.selectBatch[i].length || scene.unselectBatch[i].length;

        var trace = cd0.trace;
        var xRange = trace._xA.range;
        var yRange = trace._yA.range;
        if(opacityScale && !selected) group.opacity = options.opacity * opacityScale(trace, xRange, yRange);
        if(!sizeScale) continue;

        var factor = sizeScale(trace, xRange, yRange);
        if(!Lib.isArrayOrTypedArray(options.sizes)) {
            group.size = options.size * factor;
            var selectedGroup = selected && isReady(scene.select2d) && scene.select2d.groups[i];
            if(selectedGroup) {
                var selectedOptions = scene.markerSelectedOptions[i];
                selectedGroup.size = ((selectedOptions && selectedOptions.size) || options.size) * factor;
            }
            continue;
        }
        // select2d's per-point sizes are not scaled, so a selection keeps these at the base to match
        if(selected) factor = 1;
        if(appliedSizeFactor(scaledSizes[i], group, options) !== factor) {
            var sizes = factor === 1 ? options.sizes : scaleSizes(options.sizes, factor);
            sizeUpdates = sizeUpdates || Lib.repeat(undefined, groups.length);
            sizeUpdates[i] = {size: sizes};
            scaledSizes[i] = {base: options.sizes, sizes: sizes, factor: factor};
        }
    }
    if(sizeUpdates) scene.scatter2d.update(sizeUpdates);
}

// The factor a group's per-point sizes carry: 1 while it holds the base sizes (plotly
// resets it on every plot call), else the factor last uploaded, if those sizes are current.
function appliedSizeFactor(scaled, group, options) {
    if(group.size === options.sizes) return 1;
    return scaled && scaled.sizes === group.size && scaled.base === options.sizes ? scaled.factor : NaN;
}

function scaleSizes(sizes, factor) {
    var out = new Array(sizes.length);
    for(var j = 0; j < sizes.length; j++) out[j] = sizes[j] * factor;
    return out;
}

// make sure scene exists on subplot, return it
module.exports = function sceneUpdate(gd, subplot) {
    var scene = subplot._scene;

    var resetOpts = {
        // number of traces in subplot, since scene:subplot -> 1:1
        count: 0,
        // whether scene requires init hook in plot call (dirty plot call)
        dirty: true,
        // last used options
        lineOptions: [],
        fillOptions: [],
        markerOptions: [],
        markerSelectedOptions: [],
        markerUnselectedOptions: [],
        errorXOptions: [],
        errorYOptions: [],
        textOptions: [],
        textSelectedOptions: [],
        textUnselectedOptions: [],
        // selection batches
        selectBatch: [],
        unselectBatch: []
    };

    // regl- component stubs, initialized in dirty plot call
    var initOpts = {
        fill2d: false,
        scatter2d: false,
        error2d: false,
        line2d: false,
        glText: false,
        select2d: false
    };

    if(!subplot._scene) {
        scene = subplot._scene = {};

        scene.init = function init() {
            Lib.extendFlat(scene, initOpts, resetOpts);
        };

        scene.init();

        // apply new option to all regl components (used on drag)
        scene.update = function update(opt) {
            var opts = Lib.repeat(opt, scene.count);

            if(isReady(scene.fill2d)) scene.fill2d.update(opts);
            if(isReady(scene.scatter2d)) scene.scatter2d.update(opts);
            if(isReady(scene.line2d)) scene.line2d.update(opts);
            if(isReady(scene.error2d)) scene.error2d.update(opts.concat(opts));
            if(isReady(scene.select2d)) scene.select2d.update(opts);
            if(isReady(scene.glText)) {
                for(var i = 0; i < scene.count; i++) {
                    scene.glText[i].update(opt);
                }
            }
        };

        // draw traces in proper order
        scene.draw = function draw() {
            var count = scene.count;
            var fill2d = scene.fill2d;
            var error2d = scene.error2d;
            var line2d = scene.line2d;
            var scatter2d = scene.scatter2d;
            var glText = scene.glText;
            var select2d = scene.select2d;
            var selectBatch = scene.selectBatch;
            var unselectBatch = scene.unselectBatch;

            applyMarkerScale(gd, scene);

            for(var i = 0; i < count; i++) {
                if(isReady(fill2d) && scene.fillOrder[i]) {
                    fill2d.draw(scene.fillOrder[i]);
                }
                if(isReady(line2d) && scene.lineOptions[i]) {
                    line2d.draw(i);
                }
                if(isReady(error2d)) {
                    if(scene.errorXOptions[i]) error2d.draw(i);
                    if(scene.errorYOptions[i]) error2d.draw(i + count);
                }
                if(isReady(scatter2d) && scene.markerOptions[i]) {
                    if(unselectBatch[i].length) {
                        var arg = Lib.repeat([], scene.count);
                        arg[i] = unselectBatch[i];
                        scatter2d.draw(arg);
                    } else if(!selectBatch[i].length) {
                        scatter2d.draw(i);
                    }
                }
                if(isReady(glText) && glText[i] && scene.textOptions[i]) {
                    glText[i].render();
                }
            }

            if(isReady(select2d)) {
                select2d.draw(selectBatch);
            }

            scene.dirty = false;
        };

        // remove scene resources
        scene.destroy = function destroy() {
            if(scene.fill2d && scene.fill2d.destroy) scene.fill2d.destroy();
            if(scene.scatter2d && scene.scatter2d.destroy) scene.scatter2d.destroy();
            if(scene.error2d && scene.error2d.destroy) scene.error2d.destroy();
            if(scene.line2d && scene.line2d.destroy) scene.line2d.destroy();
            if(scene.select2d && scene.select2d.destroy) scene.select2d.destroy();
            if(scene.glText) {
                scene.glText.forEach(function(text) {
                    if(text.destroy) text.destroy();
                });
            }

            scene.lineOptions = null;
            scene.fillOptions = null;
            scene.markerOptions = null;
            scene.markerSelectedOptions = null;
            scene.markerUnselectedOptions = null;
            scene.errorXOptions = null;
            scene.errorYOptions = null;
            scene.textOptions = null;
            scene.textSelectedOptions = null;
            scene.textUnselectedOptions = null;

            scene.selectBatch = null;
            scene.unselectBatch = null;

            // we can't just delete _scene, because `destroy` is called in the
            // middle of supplyDefaults, before relinkPrivateKeys which will put it back.
            subplot._scene = null;
        };
    }

    // in case if we have scene from the last calc - reset data
    if(!scene.dirty) {
        Lib.extendFlat(scene, resetOpts);
    }

    return scene;
};
