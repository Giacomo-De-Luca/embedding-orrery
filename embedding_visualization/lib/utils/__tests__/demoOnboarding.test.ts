import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getOnboardingAction,
  warmEmotionSearch,
  resetWarmEmotionSearchForTests,
  INTRO_STORAGE_KEY,
  TOUR_STORAGE_KEY,
  TOUR_MIN_VIEWPORT,
} from '../demoOnboarding';

const base = { isDemo: true, search: '', introSeen: false, viewportWidth: 1200 };

describe('getOnboardingAction', () => {
  it('auto-opens the intro on a bare demo first visit', () => {
    expect(getOnboardingAction(base)).toBe('intro');
  });

  it('never auto-opens outside demo builds or after being seen', () => {
    expect(getOnboardingAction({ ...base, isDemo: false })).toBeNull();
    expect(getOnboardingAction({ ...base, introSeen: true })).toBeNull();
  });

  it.each(['?collection=emotion', '?colorBy=label', '?preset=emnlp-topics'])(
    'deep link %s suppresses the auto-intro',
    (search) => {
      expect(getOnboardingAction({ ...base, search })).toBeNull();
    },
  );

  it('?intro=1 opens the dialog in any build, ignoring storage', () => {
    expect(getOnboardingAction({ ...base, search: '?intro=1', isDemo: false, introSeen: true }))
      .toBe('intro');
  });

  it('?tour=1 starts the tour in any build', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=1', isDemo: false, introSeen: true }))
      .toBe('tour');
  });

  it('?tour=1 downgrades to intro on narrow viewports', () => {
    expect(
      getOnboardingAction({ ...base, search: '?tour=1', viewportWidth: TOUR_MIN_VIEWPORT - 1 }),
    ).toBe('intro');
  });

  it('?tour=1 wins over ?intro=1', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=1&intro=1' })).toBe('tour');
  });
});

describe('storage keys', () => {
  it('are the versioned constants', () => {
    expect(INTRO_STORAGE_KEY).toBe('orrery.demo-intro.v1');
    expect(TOUR_STORAGE_KEY).toBe('orrery.demo-tour.v1');
  });
});

describe('warmEmotionSearch', () => {
  beforeEach(resetWarmEmotionSearchForTests);

  it('no-ops outside demo builds', () => {
    const query = vi.fn().mockResolvedValue({});
    warmEmotionSearch({ query }, false);
    expect(query).not.toHaveBeenCalled();
  });

  it('fires once against emotion only, then never again', () => {
    const query = vi.fn().mockResolvedValue({});
    warmEmotionSearch({ query }, true);
    warmEmotionSearch({ query }, true);
    expect(query).toHaveBeenCalledTimes(1);
    const { variables, fetchPolicy } = query.mock.calls[0][0];
    expect(variables.collectionName).toBe('emotion');
    expect(variables.nResults).toBe(1);
    expect(fetchPolicy).toBe('no-cache');
  });
});
