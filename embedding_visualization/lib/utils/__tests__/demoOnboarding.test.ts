import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getOnboardingAction,
  shouldShowMobileNotice,
  warmEmotionSearch,
  resetWarmEmotionSearchForTests,
  INTRO_STORAGE_KEY,
  TOUR_STORAGE_KEY,
  SAE_TOUR_STORAGE_KEY,
  MOBILE_NOTICE_STORAGE_KEY,
  TOUR_MIN_VIEWPORT,
} from '../demoOnboarding';

const base = {
  isDemo: true,
  search: '',
  introSeen: false,
  viewportWidth: 1200,
  mobileNoticeSeen: false,
};

/** A phone-sized viewport that has already dismissed the desktop notice. */
const narrowSeen = { viewportWidth: TOUR_MIN_VIEWPORT - 1, mobileNoticeSeen: true };

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
    expect(getOnboardingAction({ ...base, ...narrowSeen, search: '?tour=1' })).toBe('intro');
  });

  it('?tour=1 wins over ?intro=1', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=1&intro=1' })).toBe('tour');
  });

  it('?tour=sae routes to the SAE tour in any build, at any viewport', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=sae' })).toBe('sae-tour');
    expect(
      getOnboardingAction({ ...base, search: '?tour=sae', isDemo: false, introSeen: true }),
    ).toBe('sae-tour');
    // No downgrade here — the /sae page owns the viewport floor.
    expect(getOnboardingAction({ ...base, ...narrowSeen, search: '?tour=sae' })).toBe('sae-tour');
  });

  it('?tour=sae wins over ?intro=1', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=sae&intro=1' })).toBe('sae-tour');
  });

  it('unknown ?tour values fall through to the ordinary gating', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=nope' })).toBeNull();
  });

  it('shows the mobile notice on a phone-sized demo first visit', () => {
    expect(
      getOnboardingAction({ ...base, viewportWidth: TOUR_MIN_VIEWPORT - 1 }),
    ).toBe('mobile-notice');
  });

  it.each(['?tour=1', '?intro=1', '?tour=sae'])(
    'the mobile notice pre-empts %s',
    (search) => {
      expect(
        getOnboardingAction({ ...base, search, viewportWidth: TOUR_MIN_VIEWPORT - 1 }),
      ).toBe('mobile-notice');
    },
  );

  it('once dismissed, narrow viewports resume the ordinary gating', () => {
    expect(getOnboardingAction({ ...base, ...narrowSeen })).toBe('intro');
    expect(getOnboardingAction({ ...base, ...narrowSeen, introSeen: true })).toBeNull();
  });
});

describe('shouldShowMobileNotice', () => {
  const narrow = { isDemo: true, viewportWidth: 390, mobileNoticeSeen: false };

  it('fires on an unseen phone-sized demo viewport', () => {
    expect(shouldShowMobileNotice(narrow)).toBe(true);
  });

  it('is demo-only — self-hosted builds are never gated', () => {
    expect(shouldShowMobileNotice({ ...narrow, isDemo: false })).toBe(false);
  });

  it('does not fire at or above the viewport floor', () => {
    expect(shouldShowMobileNotice({ ...narrow, viewportWidth: TOUR_MIN_VIEWPORT })).toBe(false);
    expect(shouldShowMobileNotice({ ...narrow, viewportWidth: TOUR_MIN_VIEWPORT - 1 })).toBe(true);
  });

  it('does not fire once dismissed', () => {
    expect(shouldShowMobileNotice({ ...narrow, mobileNoticeSeen: true })).toBe(false);
  });
});

describe('storage keys', () => {
  it('are the versioned constants', () => {
    expect(INTRO_STORAGE_KEY).toBe('orrery.demo-intro.v1');
    expect(TOUR_STORAGE_KEY).toBe('orrery.demo-tour.v1');
    expect(SAE_TOUR_STORAGE_KEY).toBe('orrery.demo-sae-tour.v1');
    expect(MOBILE_NOTICE_STORAGE_KEY).toBe('orrery.demo-mobile-notice.v1');
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
