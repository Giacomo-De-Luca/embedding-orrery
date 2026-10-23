import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { RadioGroup, RadioGroupItem } from '@/lib/ui-primitives/radio-group';
import { Slider } from '@/lib/ui-primitives/slider';

/** The class list of the first element carrying `data-slot="<slot>"`. */
function slotClasses(html: string, slot: string): string[] {
  const match = html.match(new RegExp(`data-slot="${slot}"[^>]*class="([^"]*)"`));
  return (match?.[1] ?? '').split(' ').filter(Boolean);
}

describe('dot controls', () => {
  it('a checked checkbox shows the dot and no tick icon', () => {
    const html = renderToStaticMarkup(createElement(Checkbox, { checked: true }));
    expect(html).toContain('role="checkbox"');
    expect(html).toContain('aria-checked="true"');
    expect(html).toContain('data-slot="dot-indicator"');
    expect(html).not.toContain('<svg');
  });

  it('an unchecked checkbox shows no dot', () => {
    const html = renderToStaticMarkup(createElement(Checkbox, { checked: false }));
    expect(html).toContain('aria-checked="false"');
    expect(html).not.toContain('data-slot="dot-indicator"');
  });

  it('checkbox and radio item draw with the same classes', () => {
    const checkbox = renderToStaticMarkup(createElement(Checkbox, { checked: true }));
    const radio = renderToStaticMarkup(
      createElement(RadioGroup, { value: 'a' }, createElement(RadioGroupItem, { value: 'a' })),
    );
    expect(radio).toContain('role="radio"');
    expect(radio).toContain('data-slot="dot-indicator"');
    const look = (classes: string[]) => classes.filter((c) => c !== 'peer').sort();
    expect(look(slotClasses(radio, 'radio-group-item'))).toEqual(look(slotClasses(checkbox, 'checkbox')));
    expect(slotClasses(checkbox, 'checkbox')).toContain('rounded-full');
  });
});

describe('slider', () => {
  it('keeps a 24px target over a hairline track whose fill gradient spans the track', () => {
    const html = renderToStaticMarkup(createElement(Slider, { value: [30], min: 0, max: 100 }));
    expect(html).toContain('role="slider"');
    expect(slotClasses(html, 'slider')).toContain('h-6');
    expect(slotClasses(html, 'slider-track')).toEqual(
      expect.arrayContaining(['@container', 'bg-foreground/12', 'data-[orientation=horizontal]:h-1']),
    );
    expect(slotClasses(html, 'slider-range')).toEqual(
      expect.arrayContaining(['bg-linear-to-r', 'from-primary/55', '[background-size:100cqw_100%]']),
    );
    expect(slotClasses(html, 'slider-thumb')).toEqual(
      expect.arrayContaining(['size-3.5', 'ring-offset-2', 'ring-offset-background']),
    );
  });
});
