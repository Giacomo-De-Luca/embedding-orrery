'use client';

import { useEffect, useState } from 'react';
import { LabelFonts, type TextMeasurer } from '../utils/labels/labelStyle';

/**
 * A text measurer for label placement, replaced by a new one once the label fonts
 * have loaded: text measured before then used a fallback font. Placements depend
 * on it, so they are measured again with the real font; plots that measure on
 * their own canvas list it to redraw then.
 */
export function useLabelMeasurer(): TextMeasurer | null {
  const [measurer, setMeasurer] = useState(() => LabelFonts.createMeasurer());
  useEffect(() => {
    if (LabelFonts.isReady()) return;
    let active = true;
    LabelFonts.whenReady().then(() => {
      if (active) setMeasurer(LabelFonts.createMeasurer());
    });
    return () => {
      active = false;
    };
  }, []);
  return measurer;
}
