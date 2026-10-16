import { describe, it, expect } from 'vitest';
import { buildPageNavItems } from '../pageNavItems';
import { DEMO_DISABLED_MESSAGE } from '../demoMode';

describe('buildPageNavItems', () => {
  it('lists Explore, Collections and SAE in order, all enabled outside the demo', () => {
    const items = buildPageNavItems({ isDemo: false });
    expect(items.map((i) => i.id)).toEqual(['explore', 'collections', 'sae']);
    expect(items.every((i) => !i.disabled)).toBe(true);
  });

  it('matches Explore exactly so "/" does not claim every route', () => {
    const explore = buildPageNavItems({ isDemo: false })[0];
    expect(explore.href).toBe('/');
    expect(explore.match).toBe('exact');
  });

  it('defaults the SAE link to /sae and carries a context deep link through', () => {
    expect(buildPageNavItems({ isDemo: false })[2].href).toBe('/sae');
    const deepLink = '/sae?modelId=gemma-3-4b-it&saeId=9-gemmascope-2-res-16k';
    expect(buildPageNavItems({ isDemo: false, saeHref: deepLink })[2].href).toBe(deepLink);
  });

  it('keeps Collections visible but disabled, with the shared reason, in the demo', () => {
    const items = buildPageNavItems({ isDemo: true });
    const collections = items.find((i) => i.id === 'collections');
    expect(collections?.disabled).toBe(true);
    expect(collections?.disabledReason).toBe(DEMO_DISABLED_MESSAGE);
    // The SAE explorer is live in the demo.
    expect(items.filter((i) => i.disabled).map((i) => i.id)).toEqual(['collections']);
  });
});
